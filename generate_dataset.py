#!/usr/bin/env python
"""
generate_dataset.py  --  Mass-produce four-bar linkage samples.

Abuses ``simulate_batch_no_grad``'s vectorisation: one PyTorch call simulates
BATCH mechanisms in parallel, then valid samples are collected and saved.

Dataset format  (dataset.npz  by default)
-----------------------------------------
  configs    (M, 5, 2)   PSlice -- raw joint positions. Joint 0 is fixed at
                          the origin; joints are in original index order.
  paths      (M, T, 2)   Path-node trajectory, arc-length resampled to T pts.
  fractions  (M,)        Fraction of the full 2-pi sweep that is reachable
                          (1.0 = full rotation, i.e. Grashof crank).
  modes      (M,)        0 = full, 1 = wrap-around arc, 2 = open arc.

Each row i is one sample:
    initial_config = configs[i]   # shape (5, 2)
    path           = paths[i]     # shape (T, 2)

Usage
-----
  python generate_dataset.py                        # 100k samples -> dataset.npz
  python generate_dataset.py --n 500000 --batch 16384 --out big.npz
  python generate_dataset.py --n 50000 --device cuda --seed 7
  python generate_dataset.py --n 10000 --out_t 100  # 100-point paths

Options
-------
  --n          int    Target number of valid samples          [100 000]
  --batch      int    Mechanisms simulated per GPU call       [8 192]
  --out        str    Output file path (*.npz)                [dataset.npz]
  --out_t      int    Output path length (resampled)          [200]
  --sim_t      int    Theta samples per simulation            [200]
  --seed       int    NumPy random seed                       [42]
  --device     str    'cpu' | 'cuda' | 'auto'                 [auto]
  --min_frac   float  Minimum reachable arc fraction          [0.05]
  --min_link   float  Minimum link length (avoids degeneracy) [0.2]
  --lo  --hi   float  Sampling box for joint positions        [-6, 6]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# -- resolve the Argentina directory -----------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from simulator import sort_mechanism, simulate_batch_no_grad, reachable_arc


# ===========================================================================
# Fixed four-bar topology  (adjacency matrix + roles)
# ===========================================================================

JJ = np.array(
    [
        [0, 1, 0, 0, 0],
        [1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0],
    ],
    dtype=np.float64,
)

MOTOR       = [0, 1]
FIXED_NODES = [0, 2]
PATH_NODE   = 4
N_JOINTS    = 5

# Connected edge pairs (original joint indices) -- used for link-length filter.
EDGES = [(0, 1), (1, 3), (2, 3), (1, 4), (3, 4)]

# Mode encoding
MODE_INT = {"full": 0, "wrap": 1, "arc": 2}

# -- Pre-compute topology sort (same for every sample) -----------------------
# sort_mechanism depends only on JJ structure, not on PSlice values.
_ps0 = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.5, 1.0], [0.5, 0.5]]
)
JJ_S, _, _, FN_S, ORD_ = sort_mechanism(JJ, _ps0, MOTOR, FIXED_NODES)
INV_ORD = np.empty_like(ORD_)
INV_ORD[ORD_] = np.arange(N_JOINTS)
PATH_IDX = int(INV_ORD[PATH_NODE])   # column of path node in the sorted sol

# Fixed-node type vector: shape (N, 1), 1.0 for ground joints
NT = np.zeros((N_JOINTS, 1))
NT[FN_S] = 1.0


# ===========================================================================
# Helpers
# ===========================================================================

def resample_arc(arc, n_out):
    """
    Arc-length reparameterise arc (M, 2) -> (n_out, 2).
    Returns None if the arc is degenerate (total length < 1e-9).
    """
    if len(arc) < 2:
        return None
    diffs    = np.diff(arc, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen   = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total    = cumlen[-1]
    if total < 1e-9:
        return None
    cumlen /= total
    t = np.linspace(0.0, 1.0, n_out)
    return np.stack([np.interp(t, cumlen, arc[:, c]) for c in range(2)], axis=1)


def sample_pslices(B, rng, lo=-6.0, hi=6.0, ground_lo=0.5, ground_hi=8.0):
    """
    Sample B random PSlice arrays -> (B, 5, 2).

    Normalisation applied here:
      * Joint 0 (motor pivot)  fixed at the origin.
      * Joint 2 (other ground) placed at a random distance in
        [ground_lo, ground_hi] from the origin, random direction.
      * Joints 1, 3, 4         uniform in [lo, hi]^2.
    """
    ps = rng.uniform(lo, hi, (B, N_JOINTS, 2))
    ps[:, 0] = 0.0                           # joint 0 at origin

    # Joint 2 at a controlled ground span (avoids near-zero ground links).
    ang2 = rng.uniform(0.0, 2.0 * np.pi, B)
    r2   = rng.uniform(ground_lo, ground_hi, B)
    ps[:, 2, 0] = r2 * np.cos(ang2)
    ps[:, 2, 1] = r2 * np.sin(ang2)

    return ps


def filter_min_link(ps_batch, min_link):
    """
    Return boolean mask (B,) -- True when every link length >= min_link.
    Fully vectorised: no Python loops over B.
    """
    a_idx = [a for a, _ in EDGES]
    b_idx = [b for _, b in EDGES]
    diffs = ps_batch[:, a_idx] - ps_batch[:, b_idx]   # (B, E, 2)
    lens  = np.hypot(diffs[:, :, 0], diffs[:, :, 1])  # (B, E)
    return (lens >= min_link).all(axis=1)               # (B,)


# ===========================================================================
# Core simulation + collection loop
# ===========================================================================

def simulate_and_collect(ps_batch, thetas_t, device, out_t, min_frac, min_link):
    """
    Simulate ps_batch (B, 5, 2) in one vectorised call, return valid samples.

    Returns four parallel lists:
        configs   list of (5, 2) np.ndarray  -- original (unsorted) PSlice
        paths     list of (out_t, 2) np.ndarray  -- arc-length resampled path
        fractions list of float
        modes     list of int   (0=full, 1=wrap, 2=arc)
    """
    B = ps_batch.shape[0]

    # -- geometric pre-filter ------------------------------------------------
    valid_mask = filter_min_link(ps_batch, min_link)
    if not valid_mask.any():
        return [], [], [], []

    keep_idx = np.where(valid_mask)[0]
    ps_keep  = ps_batch[keep_idx]           # (Bf, 5, 2)
    Bf       = len(keep_idx)

    # -- sort joint order for the solver (ORD_ is [0,1,2,3,4] for this topo) -
    ps_sorted = ps_keep[:, ORD_, :]         # (Bf, 5, 2)

    # -- build batch tensors -------------------------------------------------
    JJs_t = torch.as_tensor(
        np.broadcast_to(JJ_S[None], (Bf, N_JOINTS, N_JOINTS)).copy(),
        dtype=torch.float64, device=device,
    )
    PSls_t = torch.as_tensor(ps_sorted, dtype=torch.float64, device=device)
    NTs_t  = torch.as_tensor(
        np.broadcast_to(NT[None], (Bf, N_JOINTS, 1)).copy(),
        dtype=torch.float64, device=device,
    )

    # -- one batched forward pass --------------------------------------------
    sol    = simulate_batch_no_grad(JJs_t, PSls_t, NTs_t, thetas_t)
    # sol : (Bf, N, T, 2)
    traces = sol[:, PATH_IDX].cpu().numpy()  # (Bf, T, 2)

    # -- post-process each trace ---------------------------------------------
    configs, paths, fractions, modes = [], [], [], []

    for i, bi in enumerate(keep_idx):
        arc, info = reachable_arc(traces[i])

        if info["fraction"] < min_frac or len(arc) < 3:
            continue

        path = resample_arc(arc, out_t)
        if path is None:
            continue

        configs.append(ps_batch[bi].copy())    # original (unsorted) PSlice
        paths.append(path)
        fractions.append(float(info["fraction"]))
        modes.append(MODE_INT.get(info["mode"], 2))

    return configs, paths, fractions, modes


# ===========================================================================
# Main
# ===========================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Mass-produce four-bar linkage samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n",        type=int,   default=100_000, help="Target valid samples")
    p.add_argument("--batch",    type=int,   default=8_192,   help="Mechanisms per GPU call")
    p.add_argument("--out",      type=str,   default="dataset.npz", help="Output .npz path")
    p.add_argument("--out_t",    type=int,   default=200,     help="Path length after resampling")
    p.add_argument("--sim_t",    type=int,   default=200,     help="Theta samples per simulation")
    p.add_argument("--seed",     type=int,   default=42,      help="NumPy RNG seed")
    p.add_argument("--device",   type=str,   default="auto",  help="cpu | cuda | auto")
    p.add_argument("--min_frac", type=float, default=0.05,    help="Min reachable arc fraction")
    p.add_argument("--min_link", type=float, default=0.2,     help="Min link length")
    p.add_argument("--lo",       type=float, default=-6.0,    help="Sampling box lower bound")
    p.add_argument("--hi",       type=float, default=6.0,     help="Sampling box upper bound")
    return p


def main():
    args = build_parser().parse_args()

    # -- device --------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    # -- shared theta tensor (reused every batch) ----------------------------
    thetas_t = torch.linspace(
        0.0, 2.0 * torch.pi, args.sim_t + 1, dtype=torch.float64, device=device
    )[: args.sim_t]

    # -- accumulators --------------------------------------------------------
    all_configs   = []
    all_paths     = []
    all_fractions = []
    all_modes     = []

    rng       = np.random.default_rng(args.seed)
    collected = 0
    simulated = 0
    batch_no  = 0
    t0_total  = time.perf_counter()

    print(
        f"Target : {args.n:,} valid samples   "
        f"(batch={args.batch}, sim_T={args.sim_t}, out_T={args.out_t}, "
        f"min_frac={args.min_frac}, min_link={args.min_link})"
    )
    print("-" * 72)

    while collected < args.n:
        batch_no += 1
        t0 = time.perf_counter()

        ps_batch = sample_pslices(args.batch, rng, lo=args.lo, hi=args.hi)

        cfgs, paths, fracs, mods = simulate_and_collect(
            ps_batch, thetas_t, device,
            out_t    = args.out_t,
            min_frac = args.min_frac,
            min_link = args.min_link,
        )

        n_new      = len(cfgs)
        simulated += args.batch
        collected += n_new

        all_configs.extend(cfgs)
        all_paths.extend(paths)
        all_fractions.extend(fracs)
        all_modes.extend(mods)

        elapsed = time.perf_counter() - t0
        rate    = n_new / elapsed if elapsed > 0 else float("inf")
        accept  = 100.0 * n_new / args.batch

        print(
            f"  batch {batch_no:>4d} | "
            f"yield {n_new:>5d}/{args.batch}  ({accept:4.1f}%)  |  "
            f"{elapsed:5.2f}s  ({rate:6.0f} smp/s)  |  "
            f"total {min(collected, args.n):>8,}/{args.n:,}"
        )

    # Trim to exactly args.n
    all_configs   = all_configs[: args.n]
    all_paths     = all_paths[: args.n]
    all_fractions = all_fractions[: args.n]
    all_modes     = all_modes[: args.n]

    # -- stack and save ------------------------------------------------------
    configs_arr   = np.stack(all_configs,   axis=0).astype(np.float32)
    paths_arr     = np.stack(all_paths,     axis=0).astype(np.float32)
    fractions_arr = np.array(all_fractions, dtype=np.float32)
    modes_arr     = np.array(all_modes,     dtype=np.int8)

    out_path = (
        os.path.join(HERE, args.out)
        if not os.path.isabs(args.out)
        else args.out
    )
    np.savez_compressed(
        out_path,
        configs   = configs_arr,    # (M, 5, 2)
        paths     = paths_arr,      # (M, T, 2)
        fractions = fractions_arr,  # (M,)
        modes     = modes_arr,      # (M,)  0=full 1=wrap 2=arc
    )

    total_time = time.perf_counter() - t0_total
    size_mb    = os.path.getsize(out_path) / 1024 / 1024

    print("-" * 72)
    print(f"Saved  -> {out_path}")
    print(f"  configs  : {configs_arr.shape}    float32")
    print(f"  paths    : {paths_arr.shape}  float32")
    print(f"  fractions: {fractions_arr.shape}          float32")
    print(f"  modes    : {modes_arr.shape}          int8   (0=full 1=wrap 2=arc)")
    print(f"  Accept   : {100.0 * args.n / simulated:.1f}%  ({args.n:,}/{simulated:,})")
    print(f"  File     : {size_mb:.1f} MB")
    print(f"  Time     : {total_time:.1f} s")
    print()
    print("Load with:")
    print(f"  data = np.load('{os.path.basename(out_path)}')")
    print( "  configs, paths = data['configs'], data['paths']")
    print( "  # configs[i]: (5, 2)  initial joint positions")
    print( "  # paths[i]:   (T, 2)  path-node trajectory")


if __name__ == "__main__":
    main()
