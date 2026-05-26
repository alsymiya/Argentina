#!/usr/bin/env python
"""
novelty.py -- Novelty scoring for four-bar mechanism datasets.

Feature vector (12 values per mechanism)
-----------------------------------------
  [0 : 5]   Link lengths for each of the 5 connected edge pairs.
  [5 : 12]  2D cross-products for every pair of bars meeting at the
             same joint.  Joints 1, 3, 4 have 3, 3, 1 bar-pairs
             respectively, giving 7 cross-product scalars.

Cross-product  (va x vb)  at joint k:
  va = ps[a] - ps[k],  vb = ps[b] - ps[k]
  cross = va.x * vb.y - va.y * vb.x  =  |va| |vb| sin(theta)
Encodes both the enclosed angle and the product of the two link lengths,
so it is a richer shape descriptor than either alone.

Novelty score
-------------
  novelty[i] = mean Euclidean distance from sample i to its k nearest
               neighbours in (standardised) feature space.

RAM budget
----------
  Features matrix  :  M * 12 * 4 bytes      (~9.6 MB  for M=100k)
  Peak per chunk   :  chunk * M * 4 * 3      (~300 MB  for chunk=256, M=100k)
  (The factor of 3 accounts for dots, dists_sq, and the np.partition copy.)
  Total stays << 1 GB for any dataset up to ~1 M samples with chunk <= 512.

Usage
-----
  python novelty.py dataset.npz                # scores + summary stats
  python novelty.py dataset.npz --k 10 --chunk 512
  python novelty.py dataset.npz --out scores.npy
  python novelty.py dataset.npz --plot         # histogram (requires matplotlib)
  python novelty.py dataset.npz --topk 20      # print 20 most/least novel
"""

import argparse
import os
import sys
import time

import numpy as np


# ===========================================================================
# Topology constants  (must match generate_dataset.py)
# ===========================================================================

# 5 connected edge pairs -> 5 link lengths
EDGES = [(0, 1), (1, 3), (2, 3), (1, 4), (3, 4)]

# For each joint with >= 2 bars: (pivot_joint, endpoint_a, endpoint_b)
# At joint k the cross product is (ps[a]-ps[k]) x (ps[b]-ps[k]).
#
#  joint 1 connects to joints 0, 3, 4  ->  3 pairs
#  joint 3 connects to joints 1, 2, 4  ->  3 pairs
#  joint 4 connects to joints 1, 3     ->  1 pair
CROSS_PAIRS = [
    (1, 0, 3),
    (1, 0, 4),
    (1, 3, 4),
    (3, 1, 2),
    (3, 1, 4),
    (3, 2, 4),
    (4, 1, 3),
]

N_FEATS = len(EDGES) + len(CROSS_PAIRS)   # 12


# ===========================================================================
# Feature extraction  (fully vectorised, no Python loops over M)
# ===========================================================================

def mechanism_features(configs):
    """
    Build the 12-d descriptor for every mechanism in one NumPy pass.

    Parameters
    ----------
    configs : (M, 5, 2)  float array  -- raw PSlice values

    Returns
    -------
    feats : (M, 12)  float64
        Columns 0-4  : link lengths   (EDGES order, always >= 0)
        Columns 5-11 : cross-products (CROSS_PAIRS order, signed)
    """
    M = configs.shape[0]
    feats = np.empty((M, N_FEATS), dtype=np.float64)

    # -- link lengths --------------------------------------------------------
    for fi, (a, b) in enumerate(EDGES):
        d = configs[:, a] - configs[:, b]          # (M, 2)
        feats[:, fi] = np.hypot(d[:, 0], d[:, 1])

    # -- cross products ------------------------------------------------------
    for fi, (k, a, b) in enumerate(CROSS_PAIRS):
        va = configs[:, a] - configs[:, k]         # (M, 2)
        vb = configs[:, b] - configs[:, k]         # (M, 2)
        feats[:, 5 + fi] = va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0]

    return feats


# ===========================================================================
# Standardisation
# ===========================================================================

def standardise(feats):
    """
    Shift to zero mean, unit std per dimension.

    Returns (feats_std, mu, sigma).
    Constant dimensions (std < 1e-12) are left as zero rather than
    generating NaN -- they carry no discriminative information anyway.
    """
    mu    = feats.mean(axis=0)           # (D,)
    sigma = feats.std(axis=0)            # (D,)
    sigma[sigma < 1e-12] = 1.0          # safe divide
    return (feats - mu) / sigma, mu, sigma


# ===========================================================================
# Chunked k-NN Euclidean distance  (RAM-safe)
# ===========================================================================

def novelty_scores(feats_std, k=5, chunk=256, verbose=True):
    """
    Novelty[i] = mean distance from sample i to its k nearest neighbours.

    Avoids the full (M, M) distance matrix by processing `chunk` query
    rows at a time.  Uses the identity

        ||a - b||^2  =  ||a||^2  +  ||b||^2  -  2 (a . b)

    so the only large allocation is the (chunk, M) dot-product matrix.

    Parameters
    ----------
    feats_std : (M, D)  standardised float array
    k         : int     number of nearest neighbours
    chunk     : int     query batch size  -- tune to fit your RAM budget.
                        Peak usage ~ chunk * M * 12 bytes  (float32, 3 temps).
    verbose   : bool    print progress

    Returns
    -------
    scores : (M,)  float32  -- higher = more novel
    """
    M, D = feats_std.shape

    # float32 cuts peak RAM in half vs float64 with negligible precision loss
    F = feats_std.astype(np.float32)
    norms_sq = (F ** 2).sum(axis=1)       # (M,)  precomputed once

    scores = np.empty(M, dtype=np.float32)

    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        q   = F[start:end]               # (C, D)
        C   = end - start

        # -- squared distances in one BLAS dgemm call ------------------------
        # dots  : (C, M)
        dots = q @ F.T                                   # (C, M)

        # reuse 'dots' in-place: avoids a second (C, M) allocation
        # dists_sq = q_norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        q_norms_sq = (q ** 2).sum(axis=1)               # (C,)
        dots *= -2.0
        dots += q_norms_sq[:, None]
        dots += norms_sq[None, :]
        np.clip(dots, 0.0, None, out=dots)               # kill float32 noise

        # -- mask self-distances  (diagonal block) ---------------------------
        for ci in range(C):
            dots[ci, start + ci] = np.inf

        # -- mean of k-nearest distances  (np.partition = O(M) not O(M logM)) -
        knn_sq = np.partition(dots, k, axis=1)[:, :k]   # (C, k)
        scores[start:end] = np.sqrt(knn_sq).mean(axis=1)

        if verbose and (start == 0 or end == M or (start // chunk) % 20 == 0):
            pct = 100.0 * end / M
            print(f"  novelty  {end:>8,}/{M:,}  ({pct:5.1f}%)", end="\r", flush=True)

    if verbose:
        print()

    return scores


# ===========================================================================
# CLI
# ===========================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Novelty scoring for four-bar mechanism datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset",         help="Path to dataset.npz produced by generate_dataset.py")
    p.add_argument("--k",    type=int,   default=5,    help="k-NN neighbourhood size")
    p.add_argument("--chunk",type=int,   default=256,  help="Query chunk size (RAM knob)")
    p.add_argument("--out",  type=str,   default=None, help="Save scores as .npy (default: <dataset>_scores.npy)")
    p.add_argument("--plot", action="store_true",      help="Show histogram of novelty scores")
    p.add_argument("--topk", type=int,   default=0,    help="Print indices of top-k most/least novel samples")
    return p


def main():
    args = build_parser().parse_args()

    # -- load ----------------------------------------------------------------
    print(f"Loading {args.dataset} ...")
    data = np.load(args.dataset)
    configs = data["configs"]            # (M, 5, 2)
    M = configs.shape[0]
    print(f"  {M:,} mechanisms loaded")

    # estimated peak RAM in MB
    peak_mb = args.chunk * M * 12 / 1e6
    print(f"  k={args.k}, chunk={args.chunk}  "
          f"(estimated peak RAM ~ {peak_mb:.0f} MB)")

    # -- features ------------------------------------------------------------
    t0 = time.perf_counter()
    feats = mechanism_features(configs.astype(np.float64))
    print(f"  Features computed: {feats.shape}  ({time.perf_counter()-t0:.2f}s)")

    print("\nFeature statistics (before standardisation):")
    names = ([f"len_{a}-{b}" for a,b in EDGES]
             + [f"cross_j{k}_{a}{b}" for k,a,b in CROSS_PAIRS])
    w = max(len(n) for n in names)
    print(f"  {'name':<{w}}   mean      std       min       max")
    for i, n in enumerate(names):
        col = feats[:, i]
        print(f"  {n:<{w}}  {col.mean():+8.3f}  {col.std():8.3f}  "
              f"{col.min():+8.3f}  {col.max():+8.3f}")

    # -- standardise ---------------------------------------------------------
    feats_std, mu, sigma = standardise(feats)

    # -- novelty scores ------------------------------------------------------
    print(f"\nComputing {args.k}-NN novelty scores ...")
    t0 = time.perf_counter()
    scores = novelty_scores(feats_std, k=args.k, chunk=args.chunk)
    elapsed = time.perf_counter() - t0

    print(f"\nNovelty score statistics  ({elapsed:.1f}s):")
    print(f"  mean   {scores.mean():.4f}")
    print(f"  std    {scores.std():.4f}")
    print(f"  min    {scores.min():.4f}   (most similar / redundant)")
    print(f"  median {float(np.median(scores)):.4f}")
    print(f"  max    {scores.max():.4f}   (most novel / outlier)")
    print(f"  p5     {float(np.percentile(scores,  5)):.4f}")
    print(f"  p95    {float(np.percentile(scores, 95)):.4f}")

    if args.topk > 0:
        order = np.argsort(scores)
        print(f"\nTop-{args.topk} MOST novel (highest score):")
        for idx in order[-args.topk:][::-1]:
            print(f"  [{idx:>7d}]  novelty={scores[idx]:.4f}")
        print(f"\nTop-{args.topk} LEAST novel (most redundant):")
        for idx in order[:args.topk]:
            print(f"  [{idx:>7d}]  novelty={scores[idx]:.4f}")

    # -- save ----------------------------------------------------------------
    out_path = args.out
    if out_path is None:
        base = os.path.splitext(args.dataset)[0]
        out_path = base + "_scores.npy"
    np.save(out_path, scores)
    print(f"\nScores saved -> {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")

    # -- optional plot -------------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].hist(scores, bins=80, color="#1f77b4", edgecolor="none", alpha=0.85)
        axes[0].axvline(scores.mean(), color="red", lw=1.5, label=f"mean={scores.mean():.3f}")
        axes[0].set_xlabel("novelty score (mean k-NN distance)")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"Novelty distribution  (k={args.k}, N={M:,})")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # scatter: first two PCA dims of features vs novelty
        try:
            U, S, Vt = np.linalg.svd(feats_std, full_matrices=False)
            pc = U[:, :2] * S[:2]
            sc = axes[1].scatter(pc[:, 0], pc[:, 1], c=scores,
                                 cmap="viridis", s=2, alpha=0.5, rasterized=True)
            plt.colorbar(sc, ax=axes[1], label="novelty")
            axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
            axes[1].set_title("Feature PCA coloured by novelty")
            axes[1].grid(alpha=0.3)
        except Exception as e:
            axes[1].set_title(f"PCA unavailable: {e}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
