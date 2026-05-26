"""
simulator.py — Differentiable dyadic-linkage simulator (PyTorch).

Argentina demo, self-contained: nothing in this file imports from outside the
Argentina/ folder. Optimizer and visualizer both import from here.

Public API
----------
    find_path(JJ, motor, fixed_nodes)             -> (path, ok)
    get_order(JJ, motor, fixed_nodes)             -> permutation
    sort_mechanism(JJ, PSlice, motor, fixed_nodes)
        -> (JJ_s, PSlice_s, motor_s, fixed_s, ord_)

    simulate_batch(JJs, PSlices, node_types, thetas, distance_to_locking=False)
    simulate_batch_no_grad(JJs, PSlices, node_types, thetas, distance_to_locking=False)
        -> x          : (B, N, T, 2)
           [cos_phis] : (B, N-3, T)   if distance_to_locking

    solve_single_mechanism(JJ, PSlice, motor, fixed_nodes,
                           thetas=None, device='cpu', differentiable=False)
        -> sol        : (N, T, 2)
           ord_       : (N,)

    reachable_arc(trace)
        Given a (T, 2) per-timestep joint trajectory whose unreachable theta
        samples are NaN, return the longest contiguous reachable arc as ONE
        curve — with wrap-around joining (e.g. samples [100, T-1] + [0, 10]
        become one continuous arc in cyclic theta order) — plus a small dict
        describing the range and mode ('full' / 'wrap' / 'arc' / 'none').
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Topology helpers (NumPy — only run once per mechanism)
# ---------------------------------------------------------------------------

def find_path(JJ, motor=[0, 1], fixed_nodes=[0, 1]):
    """Return the dyadic solve order as a list of [k, i, j] triples."""
    path = []
    JJ, fixed_nodes, motor = np.array(JJ), np.array(fixed_nodes), np.array(motor)

    unknowns = np.array(list(range(JJ.shape[0])))
    knowns = np.concatenate([fixed_nodes, [motor[-1]]])
    unknowns = unknowns[np.logical_not(np.isin(unknowns, knowns))]

    counter = 0
    while unknowns.shape[0] != 0:
        if counter == unknowns.shape[0]:
            return [], False
        n = unknowns[counter]
        neighbors = np.where(JJ[n])[0]
        known_neighbors = knowns[np.isin(knowns, neighbors)]

        if known_neighbors.shape[0] == 2:
            path.append([n, known_neighbors[0], known_neighbors[1]])
            counter = 0
            knowns = np.concatenate([knowns, [n]])
            unknowns = unknowns[unknowns != n]
        elif known_neighbors.shape[0] > 2:
            return [], False
        else:
            counter += 1

    return np.array(path), True


def get_order(JJ, motor=[0, 1], fixed_nodes=[0, 1]):
    """[motor, other fixed, passive joints in dyad order]."""
    path, status = find_path(JJ, motor, fixed_nodes)
    fixed_nodes = np.array(fixed_nodes)
    if status:
        return np.concatenate(
            [motor, fixed_nodes[fixed_nodes != motor[0]], path[:, 0]]
        )
    raise Exception("Non-dyadic or DOF larger than 1")


def sort_mechanism(JJ, PSlice, motor=[0, 1], fixed_nodes=[0, 1]):
    """Permute JJ + PSlice into canonical solving order."""
    ord_ = get_order(JJ, motor, fixed_nodes)
    n_t = np.zeros(JJ.shape[0]); n_t[fixed_nodes] = 1

    JJ_s = JJ[ord_, :][:, ord_]
    n_t_s = n_t[ord_]
    return JJ_s, PSlice[ord_], np.array([0, 1]), np.where(n_t_s)[0], ord_


# ---------------------------------------------------------------------------
# Core solver (PyTorch, vectorized over a batch of mechanisms)
# ---------------------------------------------------------------------------

def _solve_core(JJs, PSlices, node_types, thetas,
                distance_to_locking=False, safe_acos=False):
    """Vectorized dyadic solver. See module docstring for input format.

    safe_acos
        If True, cos_phi is clamped to [-1, 1] before ``torch.acos``. This
        keeps the forward finite (boundary-saturated for non-rotatable theta
        samples) and keeps the backward well-defined (``clamp``'s gradient
        is 1 inside the range, 0 outside — so unreachable samples contribute
        zero gradient instead of NaN). Use for gradient-based optimization
        on non-Grashof mechanisms. Default False preserves the original
        NaN-marks-unreachable behavior, which is what the visualizer's
        ``reachable_arc`` relies on.
    """
    B, N, _ = PSlices.shape
    T = thetas.shape[0]
    device = PSlices.device

    D = torch.cdist(PSlices, PSlices)

    # Fixed/padded joints stay at PSlice; moving joints start at zero.
    x = (node_types * PSlices).unsqueeze(2).expand(-1, -1, T, -1).contiguous()

    # Driven crank tip (joint 1).
    cos_t = torch.cos(thetas)
    sin_t = torch.sin(thetas)
    crank_dir = torch.stack([cos_t, sin_t], dim=-1)
    m = x[:, 0] + crank_dir.unsqueeze(0) * D[:, 0, 1].view(B, 1, 1)
    x = torch.cat([x[:, 0:1], m.unsqueeze(1), x[:, 2:]], dim=1)

    batch_idx = torch.arange(B, device=device)
    cos_list = []

    for k in range(3, N):
        inds = torch.argsort(JJs[:, k, 0:k], dim=-1)[:, -2:]
        i_idx, j_idx = inds[:, 0], inds[:, 1]

        xi = x[batch_idx, i_idx]
        xj = x[batch_idx, j_idx]

        l_ij = torch.linalg.norm(xj - xi, dim=-1)

        g_ik = D[batch_idx, i_idx, k].unsqueeze(-1)
        g_jk = D[batch_idx, j_idx, k].unsqueeze(-1)

        # Law of cosines: cos(phi) at vertex i of the (i, j, k) dyad triangle.
        cos_phi = (l_ij ** 2 + g_ik ** 2 - g_jk ** 2) / (2 * l_ij * g_ik)

        is_fixed = (node_types[:, k] != 0.0)
        cos_phi = torch.where(is_fixed, torch.zeros_like(cos_phi), cos_phi)
        cos_list.append(cos_phi.unsqueeze(1))

        # Assembly-mode sign from the initial geometry.
        PSlice_i = PSlices[batch_idx, i_idx]
        PSlice_j = PSlices[batch_idx, j_idx]
        PSlice_k = PSlices[:, k]
        s = torch.sign(
            (PSlice_i[:, 1] - PSlice_k[:, 1]) * (PSlice_i[:, 0] - PSlice_j[:, 0])
            - (PSlice_i[:, 1] - PSlice_j[:, 1]) * (PSlice_i[:, 0] - PSlice_k[:, 0])
        ).unsqueeze(-1)

        # Clamp before acos in the differentiable path so out-of-range
        # cos_phi (non-rotatable theta) doesn't produce NaN forward/grad.
        # cos_list above keeps the unclamped value for the locking diag.
        cos_phi_acos = cos_phi.clamp(-1.0, 1.0) if safe_acos else cos_phi
        phi = s * torch.acos(cos_phi_acos)

        cos_p, sin_p = torch.cos(phi), torch.sin(phi)
        R = torch.stack([
            torch.stack([cos_p, -sin_p], dim=-1),
            torch.stack([sin_p,  cos_p], dim=-1),
        ], dim=-2)

        scaled_ij = (xj - xi) / l_ij.unsqueeze(-1) * g_ik.unsqueeze(-1)
        x_k_computed = (R @ scaled_ij.unsqueeze(-1)).squeeze(-1) + xi

        x_k_contrib = torch.where(
            is_fixed.unsqueeze(-1),
            torch.zeros_like(x_k_computed),
            x_k_computed,
        )

        new_xk = x[:, k] + x_k_contrib
        x = torch.cat([x[:, :k], new_xk.unsqueeze(1), x[:, k + 1:]], dim=1)

    if distance_to_locking:
        cos_phis = torch.cat(cos_list, dim=1)
        return x, cos_phis
    return x


def simulate_batch(JJs, PSlices, node_types, thetas,
                   distance_to_locking=False, safe_acos=False):
    """Differentiable solver (autograd graph built). ``safe_acos=True``
    clamps cos_phi to [-1, 1] so non-rotatable thetas don't NaN out the
    gradient — see ``_solve_core``."""
    return _solve_core(JJs, PSlices, node_types, thetas,
                       distance_to_locking, safe_acos=safe_acos)


@torch.no_grad()
def simulate_batch_no_grad(JJs, PSlices, node_types, thetas,
                           distance_to_locking=False, safe_acos=False):
    """Forward-only solver — for visualization, benchmarking, FD optimizers.

    Default ``safe_acos=False`` keeps NaN markers so the visualizer's
    ``reachable_arc`` can identify non-rotatable arcs."""
    return _solve_core(JJs, PSlices, node_types, thetas,
                       distance_to_locking, safe_acos=safe_acos)


# ---------------------------------------------------------------------------
# Convenience: solve a single (unsorted) mechanism end-to-end
# ---------------------------------------------------------------------------

def solve_single_mechanism(JJ, PSlice, motor=[0, 1], fixed_nodes=[0, 1],
                           thetas=None, device='cpu', differentiable=False):
    """Topo-sort + solve one mechanism. Returns (sol, ord_)."""
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 201)[:200]

    JJ_s, PSlice_s, _, fn_s, ord_ = sort_mechanism(JJ, PSlice, motor, fixed_nodes)

    n_t = np.zeros([JJ_s.shape[0], 1]); n_t[fn_s] = 1
    JJs    = torch.as_tensor(JJ_s[None],     dtype=torch.float64, device=device)
    PSlices = torch.as_tensor(PSlice_s[None], dtype=torch.float64, device=device)
    nts    = torch.as_tensor(n_t[None],      dtype=torch.float64, device=device)
    ths    = torch.as_tensor(np.asarray(thetas), dtype=torch.float64, device=device)

    solver = simulate_batch if differentiable else simulate_batch_no_grad
    sol = solver(JJs, PSlices, nts, ths)
    return sol[0], ord_


# ---------------------------------------------------------------------------
# Non-rotatable mechanism support: pick the largest contiguous reachable arc
# from a cyclically-sampled trace, joining wrap-around runs into one curve.
# ---------------------------------------------------------------------------

def reachable_arc(trace):
    """Longest contiguous reachable arc on the cyclic theta grid.

    The simulator sweeps ``theta`` over ``[0, 2*pi)``. For a non-rotatable
    mechanism, the dyad triangle is unsolvable on part of that interval and
    the corresponding rows of ``trace`` come back as NaN. This helper picks
    the largest contiguous block of finite samples and — crucially — treats
    the index axis as cyclic: a run like indices [100..T-1] followed by
    [0..10] is one open arc that happens to straddle the theta=0 cut, and we
    rejoin it into a single curve in cyclic order.

    Inputs:
        trace : (T, 2) array. NaN rows mark unreachable theta samples.

    Returns:
        arc  : (M, 2) ndarray of the reachable arc in cyclic order.
        info : dict with keys
            'mode'     : 'full' | 'wrap' | 'arc' | 'none'
                         'full' = every theta sample is finite (closed loop).
                         'wrap' = open arc straddling the theta=0 cut.
                         'arc'  = open arc inside the grid (no wrap).
                         'none' = mechanism unreachable everywhere.
            'start'    : original (pre-wrap) index where the arc begins.
            'end'      : original (pre-wrap) index where the arc ends.
            'count'    : number of finite samples in the arc (= len(arc)).
            'fraction' : count / T.
            'indices'  : (M,) int ndarray of original indices in cyclic order
                         — handy for grabbing the matching ``theta`` values.
    """
    trace = np.asarray(trace)
    T = trace.shape[0]
    # finite[t] = True if every coordinate of trace[t] is finite.
    if trace.ndim == 1:
        finite = np.isfinite(trace)
    else:
        finite = np.isfinite(trace).all(axis=tuple(range(1, trace.ndim)))

    if not finite.any():
        return trace[:0].copy(), {
            'mode': 'none', 'start': None, 'end': None,
            'count': 0, 'fraction': 0.0,
            'indices': np.array([], dtype=int),
        }
    if finite.all():
        return trace.copy(), {
            'mode': 'full', 'start': 0, 'end': T - 1,
            'count': T, 'fraction': 1.0,
            'indices': np.arange(T),
        }

    # At least one False is present. Roll the array so the new index 0 sits
    # immediately AFTER the first False; this guarantees no True-run wraps
    # the boundary of the rolled array, so we can find all runs linearly.
    first_false = int(np.where(~finite)[0][0])
    roll = -(first_false + 1)
    rolled = np.roll(trace, roll, axis=0)
    finite_r = np.roll(finite, roll)

    # Enumerate contiguous True runs in the rolled array.
    runs = []
    i = 0
    while i < T:
        if finite_r[i]:
            j = i
            while j < T and finite_r[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1

    if not runs:
        return trace[:0].copy(), {
            'mode': 'none', 'start': None, 'end': None,
            'count': 0, 'fraction': 0.0,
            'indices': np.array([], dtype=int),
        }

    a, b = max(runs, key=lambda r: r[1] - r[0] + 1)
    arc = rolled[a:b + 1].copy()
    indices = (np.arange(a, b + 1) - roll) % T
    orig_start = int(indices[0])
    orig_end = int(indices[-1])
    mode = 'wrap' if orig_start > orig_end else 'arc'
    return arc, {
        'mode': mode, 'start': orig_start, 'end': orig_end,
        'count': int(len(arc)), 'fraction': float(len(arc)) / T,
        'indices': indices,
    }


# ---------------------------------------------------------------------------
# CLI sanity check — runs the user's four-bar example + a non-rotatable demo.
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # 5 joints: 0 motor pivot (0,0); 1 driven crank tip (1,2);
    #           2 other ground (4,0); 3 moving joint (3,2); 4 path node (2,3).
    JJ = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0],
    ])
    PSlice = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [4.0, 0.0],
        [3.0, 2.0],
        [2.0, 3.0],
    ])
    sol, ord_ = solve_single_mechanism(
        JJ, PSlice, motor=[0, 1], fixed_nodes=[0, 2],
    )
    print("=== Wishlist 4-bar ===")
    print("sorted order:", ord_.tolist())
    print("sol shape   :", tuple(sol.shape), "(N, T, 2)")
    print("finite?     :", bool(torch.isfinite(sol).all()))

    trace = sol[ord_.tolist().index(4)].cpu().numpy()  # path node, in sorted slot
    arc, info = reachable_arc(trace)
    print("reachable   :", info['mode'], "count =", info['count'],
          "start =", info['start'], "end =", info['end'])
