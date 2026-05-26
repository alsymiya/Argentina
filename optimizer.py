"""
optimizer.py — Backward / optimization toolkit for the dyadic linkage.

Argentina demo, self-contained: imports only from ``simulator`` (sibling
module in this folder).

Public API
----------
    optimize_mechanism(JJ, PSlice, target,
                       motor=..., fixed_nodes=..., path_node=...,
                       method='lbfgs', metric='ordered_l2', **kwargs)

    fit_to_expected_path(JJ, PSlice, expected_path,
                         motor=..., fixed_nodes=..., path_node=...)
        Thin wrapper meant for the visualizer's "Optimize" button. Defaults
        to soft-chamfer + temperature annealing + L-BFGS, because that handles
        arbitrary user-drawn target paths well without forcing the user to
        sample exactly T points.

Methods
-------
    'gd', 'adam', 'lbfgs', 'lbfgs_nc', 'lm', 'lm_fd'

    'lbfgs_nc' is a hand-rolled L-BFGS that does NOT use PyTorch's closure
    pattern. One forward+backward per outer step, plus a few forward-only
    evaluations for Armijo backtracking. Set ``max_ls=0`` for a constant-
    step variant.

Metrics
-------
    'ordered_l2', 'mse', 'chamfer', 'hausdorff', 'soft_chamfer'
"""

import numpy as np
import torch

from simulator import (
    simulate_batch,
    simulate_batch_no_grad,
    sort_mechanism,
)

DTYPE = torch.float64


# ===========================================================================
# Scalar metrics
# ===========================================================================

def ordered_l2_loss(curve, target):
    """Mean point-wise L2 distance (assumes 1-to-1 sample correspondence)."""
    return torch.linalg.norm(curve - target, dim=-1).mean()


def mse_loss(curve, target):
    """Mean of squared distances."""
    return ((curve - target) ** 2).sum(dim=-1).mean()


def chamfer_loss(curve, target):
    """Bidirectional nearest-neighbor mean distance."""
    dist = torch.cdist(curve, target)
    return dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()


def hausdorff_loss(curve, target):
    """One-sided Hausdorff distance summed both ways."""
    dist = torch.cdist(curve, target)
    return dist.min(dim=1).values.max() + dist.min(dim=0).values.max()


def soft_chamfer_loss(curve, target, tau=0.1):
    """Softmin-based chamfer — differentiable everywhere."""
    diff = curve.unsqueeze(1) - target.unsqueeze(0)
    dist_sq = (diff ** 2).sum(-1)
    w_fwd = torch.softmax(-dist_sq / tau, dim=1)
    w_bwd = torch.softmax(-dist_sq / tau, dim=0)
    target_matched = w_fwd @ target
    pred_matched = w_bwd.t() @ curve
    fwd_err = ((curve - target_matched) ** 2).sum(-1).mean()
    bwd_err = ((pred_matched - target) ** 2).sum(-1).mean()
    return 0.5 * (fwd_err + bwd_err)


def _soft_dtw_raw(a, b, gamma=0.1):
    """Raw, unnormalized Cuturi (2017) soft-DTW value R[N, M].

    Implementation: vectorized anti-diagonal sweep. Cell R[i, j] only
    depends on cells with smaller i+j, so a whole anti-diagonal can be
    computed as a single 1D tensor op from the previous two diagonals.
    That turns the Python loop from O(N*M) iterations into O(N+M) — ~60x
    fewer per-op autograd overheads for our ~200-sample curves. Same
    math, same softmin (log-sum-exp with running-min for stability);
    only the iteration order changes.
    """
    N, M = a.shape[0], b.shape[0]
    D = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)   # (N, M)
    dtype, device = a.dtype, a.device
    INF = float('inf')

    # Rolling 1D anti-diagonals: pp = diagonal k-2, p = diagonal k-1.
    pp = torch.zeros(1, dtype=dtype, device=device)                  # diag 0: (0,0) = 0
    p = torch.full((2,), INF, dtype=dtype, device=device)            # diag 1: both INF
    a_pp, a_p = 0, 0

    for k in range(2, N + M + 1):
        a_k = max(0, k - M)
        b_k = min(N, k)
        n_k = b_k - a_k + 1
        i_lo = max(1, a_k)
        i_hi = min(N, k - 1)

        if i_lo > i_hi:
            cur = torch.full((n_k,), INF, dtype=dtype, device=device)
        else:
            ii = torch.arange(i_lo, i_hi + 1, device=device)
            r1 = pp[(ii - 1) - a_pp]
            r2 = p[(ii - 1) - a_p]
            r3 = p[ii - a_p]
            rmin = torch.minimum(torch.minimum(r1, r2), r3)
            sm = rmin - gamma * torch.log(
                torch.exp(-(r1 - rmin) / gamma)
                + torch.exp(-(r2 - rmin) / gamma)
                + torch.exp(-(r3 - rmin) / gamma)
            )
            inner = D[ii - 1, (k - ii) - 1] + sm

            n_left = i_lo - a_k
            n_right = b_k - i_hi
            parts = []
            if n_left:
                parts.append(torch.full((n_left,), INF, dtype=dtype, device=device))
            parts.append(inner)
            if n_right:
                parts.append(torch.full((n_right,), INF, dtype=dtype, device=device))
            cur = torch.cat(parts) if len(parts) > 1 else inner

        a_pp = a_p
        a_p = a_k
        pp = p
        p = cur

    return p[0]   # R[N, M], unnormalized


def soft_dtw_loss(curve, target, gamma=0.1, divergence=True):
    """Soft-DTW loss — by default the **divergence** form (always >= 0).

    Cuturi's raw soft-DTW
        R[i, j] = D[i, j] + softmin_g(R[i-1, j-1], R[i-1, j], R[i, j-1])
    is a differentiable relaxation of Dynamic Time Warping. The softmin
    underestimates the hard min by up to ``g * log(n)`` per DP step, so
    the raw value picks up a negative bias proportional to the path
    length — meaning ``soft_dtw_raw(X, Y)`` can be **negative** when X
    and Y match closely. The bias is constant in the curve geometry so
    the gradient is unaffected, but the displayed loss number is
    misleading.

    The Blondel-Mensch-Vert (2020) divergence cures this:
        D_sdtw(X, Y) = sdtw(X, Y) - 0.5 * sdtw(X, X) - 0.5 * sdtw(Y, Y)
    which is provably >= 0, equals 0 iff X = Y, and removes the
    log(n)-per-step bias by subtracting the two self-similarity
    baselines. Compute cost is ~3x the raw form; for the Argentina
    demo's 90-200 sample curves that's still sub-second per call.

    Pass ``divergence=False`` to recover the raw Cuturi value (useful if
    you want to reproduce papers that cite the unmodified form).

    Returns the result normalized by ``(N + M)`` so different curve
    lengths give comparable numbers.
    """
    N, M = curve.shape[0], target.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((), dtype=curve.dtype, device=curve.device)

    sxy = _soft_dtw_raw(curve, target, gamma)
    if not divergence:
        return sxy / float(N + M)
    sxx = _soft_dtw_raw(curve, curve, gamma)
    syy = _soft_dtw_raw(target, target, gamma)
    return (sxy - 0.5 * sxx - 0.5 * syy) / float(N + M)


METRICS = {
    'ordered_l2':   ordered_l2_loss,
    'mse':          mse_loss,
    'chamfer':      chamfer_loss,
    'hausdorff':    hausdorff_loss,
    'soft_chamfer': soft_chamfer_loss,
    'soft_dtw':     soft_dtw_loss,
}


def make_composite_loss(fns, weights=None):
    """Sum a list of metric callables into a single composite loss.

    Each fn is called as ``fn(curve, target)``; the composite returns the
    weighted sum. If any sub-fn exposes ``set_step`` / ``tau`` (the
    annealed-tau interface), the composite proxies those to every sub-fn
    that has them — so passing ``['soft_chamfer', 'hausdorff']`` with
    ``tau_anneal=True`` Just Works.
    """
    if weights is None:
        weights = [1.0] * len(fns)
    if len(weights) != len(fns):
        raise ValueError(
            f"weights length {len(weights)} != fns length {len(fns)}"
        )

    def composite(curve, target):
        total = None
        for w, fn in zip(weights, fns):
            term = w * fn(curve, target)
            total = term if total is None else (total + term)
        return total

    sub_set_steps = [getattr(f, 'set_step', None) for f in fns]
    if any(s is not None for s in sub_set_steps):
        def set_step(k):
            for s in sub_set_steps:
                if s is not None:
                    s(k)
        composite.set_step = set_step
        for f in fns:
            if hasattr(f, 'tau'):
                composite.tau = f.tau
                break
    return composite


# ===========================================================================
# Vector residuals (LM family)
# ===========================================================================

def ordered_l2_residual(curve, target):
    return (curve - target).flatten()


def mse_residual(curve, target):
    return (curve - target).flatten()


def soft_chamfer_residual(curve, target, tau=0.1):
    diff = curve.unsqueeze(1) - target.unsqueeze(0)
    dist_sq = (diff ** 2).sum(-1)
    w_fwd = torch.softmax(-dist_sq / tau, dim=1)
    w_bwd = torch.softmax(-dist_sq / tau, dim=0)
    target_matched = w_fwd @ target
    pred_matched = w_bwd.t() @ curve
    N, M = curve.shape[0], target.shape[0]
    r_fwd = (curve - target_matched).flatten() / float(N) ** 0.5
    r_bwd = (pred_matched - target).flatten() / float(M) ** 0.5
    return torch.cat([r_fwd, r_bwd])


RESIDUALS = {
    'ordered_l2':   ordered_l2_residual,
    'mse':          mse_residual,
    'soft_chamfer': soft_chamfer_residual,
}


def make_annealed(fn, tau_init=0.5, tau_final=0.01, n_steps=200, schedule='exp'):
    """Wrap a tau-parameterized loss/residual with an annealing schedule."""
    if n_steps < 2:
        n_steps = 2
    if schedule == 'exp':
        decay = (tau_final / tau_init) ** (1.0 / (n_steps - 1))
        tau_of_step = lambda k: tau_init * (decay ** min(k, n_steps - 1))
    elif schedule == 'linear':
        tau_of_step = lambda k: (
            tau_init
            + (tau_final - tau_init) * min(k, n_steps - 1) / (n_steps - 1)
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")

    state = {'step': 0}

    def wrapped(curve, target):
        return fn(curve, target, tau=tau_of_step(state['step']))

    wrapped.set_step = lambda k: state.update({'step': int(k)})
    wrapped.tau = lambda: tau_of_step(state['step'])
    wrapped.tau_init = tau_init
    wrapped.tau_final = tau_final
    return wrapped


# ===========================================================================
# Curve normalization & locking barrier
# ===========================================================================

def normalize_curve(curve):
    """Translate min to origin, scale x-extent to 1, center."""
    c = curve - curve.min(dim=0, keepdim=True).values
    sc = c[:, 0].max().clamp(min=1e-12)
    c = c / sc
    c = c - c.max(dim=0, keepdim=True).values / 2
    return c


def locking_barrier(cos_phis, eps=1e-3):
    """Diverges as the mechanism approaches a singular dyad triangle."""
    dl = (1.0 - cos_phis ** 2).clamp(min=1e-12)
    return (torch.log(dl / eps) ** 2).mean()


# ===========================================================================
# Forward builder
# ===========================================================================

def _drop_nan_rows(c):
    """Safety net: drop rows of a (T, 2) curve whose entries are NaN/Inf.

    With ``safe_acos=True`` in the differentiable simulator (which
    ``build_forward`` enables) the forward should never produce NaN — every
    theta sample gets a finite, boundary-saturated position. This helper
    therefore acts as a defensive guard for unanticipated NaN sources
    (extreme numerical conditions, future solver changes, etc.). The
    all-NaN branch returns a sentinel that *retains* a graph link back to
    ``c`` via ``0 * c.nan_to_num().sum()`` — without that link, the loss
    has no grad_fn and ``backward()`` raises "element 0 of tensors does
    not require grad."
    """
    finite = torch.isfinite(c).all(dim=-1)
    if not finite.any():
        zero_link = 0.0 * c.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).sum()
        return torch.zeros((1, c.shape[-1]),
                           dtype=c.dtype, device=c.device) + zero_link
    return c[finite]


def _reachable_arc_torch(curve, reach_mask):
    """Torch version of ``simulator.reachable_arc`` — autograd-friendly.

    With ``safe_acos=True`` the simulator returns finite positions on the
    *unreachable* part of the theta sweep, just clamped at the dyad's
    swing limits. Those frozen-boundary samples are physically meaningless
    and pollute the loss. This helper drops them using a boolean
    reachability mask (one ``cos_phi`` per passive joint per timestep —
    a sample is reachable iff every passive joint's unclamped
    ``|cos_phi| <= 1``), then rolls cyclic theta order so the returned
    arc starts immediately after the first gap. That matches exactly what
    the visualizer's ``reachable_arc`` shows, so the optimizer fits the
    same curve the user sees.

    Autograd flows through the surviving samples. The roll uses
    ``.item()`` to get an integer shift, but the shifted values themselves
    keep their grad graph. Empty (fully unreachable) is handled with a
    graph-connected zero sentinel so ``backward()`` doesn't choke.
    """
    T = curve.shape[0]
    if reach_mask.all():
        return curve
    if not reach_mask.any():
        zero_link = 0.0 * curve.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).sum()
        return torch.zeros((1, curve.shape[-1]),
                           dtype=curve.dtype, device=curve.device) + zero_link
    # Roll so the curve starts just after the first unreachable sample;
    # then a single contiguous reachable arc shows up at the front of the
    # rolled array, and boolean indexing extracts it in cyclic order.
    first_false = int((~reach_mask).long().argmax().item())
    roll = -(first_false + 1)
    c_rolled = torch.roll(curve, roll, dims=0)
    m_rolled = torch.roll(reach_mask, roll, dims=0)
    return c_rolled[m_rolled]


def build_forward(JJ_s, fn_s, thetas, device,
                  coupler_slot=-1, with_locking=False, no_grad=False,
                  normalize=True):
    """f(PSlice_flat) -> (normalized) coupler curve at slot ``coupler_slot``.

    NaN samples produced by non-rotatable configurations are dropped before
    normalization, so the returned curve is always finite. Its length may
    therefore be less than ``len(thetas)`` for non-Grashof mechanisms —
    metrics like soft-chamfer handle that fine; ordered_l2 / mse require
    matched sample counts and should be paired with a Grashof mechanism or
    a 1-to-1 target curve.
    """
    N = JJ_s.shape[0]
    n_t = np.zeros([N, 1]); n_t[fn_s] = 1
    JJs_b = torch.as_tensor(JJ_s[None], dtype=DTYPE, device=device)
    nts_b = torch.as_tensor(n_t[None], dtype=DTYPE, device=device)
    sim = simulate_batch_no_grad if no_grad else simulate_batch
    # safe_acos=True keeps the gradient finite on non-rotatable mechanisms.
    # We ALSO need cos_phis to identify which samples are physically
    # reachable (|cos_phi| <= 1 for every passive joint) — without that
    # filter, the loss sees ~T_unreachable boundary-saturated samples that
    # pollute the fit (especially for order-sensitive metrics like DTW).
    sim_kwargs = {'safe_acos': True, 'distance_to_locking': True}

    def forward(PSlice_flat):
        PSlices = PSlice_flat.reshape(N, 2).unsqueeze(0)
        sol, cos_phis = sim(JJs_b, PSlices, nts_b, thetas, **sim_kwargs)
        c = sol[0, coupler_slot]
        # Reachability: each row of cos_phis[0] is (N-3, T); fixed-joint
        # rows are zeroed by the solver, so they don't falsely mark
        # samples as unreachable. A sample is reachable iff every
        # passive joint's UNCLAMPED |cos_phi| <= 1 at that theta.
        reach_mask = (cos_phis[0].abs() <= 1.0).all(dim=0)
        c = _reachable_arc_torch(c, reach_mask)
        # Defensive NaN safety net (should be a no-op with safe_acos).
        c = _drop_nan_rows(c)
        if with_locking:
            return (normalize_curve(c) if normalize else c), cos_phis[0]
        return normalize_curve(c) if normalize else c

    return forward


# ===========================================================================
# Optimizers
# ===========================================================================

def optimize_gd(forward_fn, x0_flat, target, loss_fn,
                n_steps=5000, lr=1e-4,
                locking=None, lock_weight=0.05, lock_eps=1e-3,
                verbose=False, report_every=500):
    x = x0_flat.clone().detach().requires_grad_(True)
    history = []
    for step in range(n_steps):
        if hasattr(loss_fn, 'set_step'):
            loss_fn.set_step(step)
        if x.grad is not None:
            x.grad.zero_()
        if locking:
            curve, cos_phis = forward_fn(x)
            loss = loss_fn(curve, target) + lock_weight * locking_barrier(cos_phis, lock_eps)
        else:
            curve = forward_fn(x)
            loss = loss_fn(curve, target)
        if not torch.isfinite(loss):
            break
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
        history.append(loss.item())
        if verbose and (step % report_every == 0):
            print(f"  step {step:5d}  loss = {loss.item():.6e}")
    return x.detach(), history


def optimize_adam(forward_fn, x0_flat, target, loss_fn,
                  n_steps=2000, lr=1e-2,
                  locking=None, lock_weight=0.05, lock_eps=1e-3,
                  verbose=False, report_every=200):
    x = x0_flat.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    history = []
    for step in range(n_steps):
        if hasattr(loss_fn, 'set_step'):
            loss_fn.set_step(step)
        optimizer.zero_grad()
        if locking:
            curve, cos_phis = forward_fn(x)
            loss = loss_fn(curve, target) + lock_weight * locking_barrier(cos_phis, lock_eps)
        else:
            curve = forward_fn(x)
            loss = loss_fn(curve, target)
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if verbose and (step % report_every == 0):
            print(f"  step {step:5d}  loss = {loss.item():.6e}")
    return x.detach(), history


def optimize_lbfgs(forward_fn, x0_flat, target, loss_fn,
                   n_outer=50, lr=0.5, max_iter=20,
                   locking=None, lock_weight=0.05, lock_eps=1e-3,
                   verbose=False, report_every=5):
    """L-BFGS using PyTorch's closure API (Strong-Wolfe line search)."""
    x = x0_flat.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([x], lr=lr, max_iter=max_iter,
                                  line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad()
        if locking:
            curve, cos_phis = forward_fn(x)
            loss = loss_fn(curve, target) + lock_weight * locking_barrier(cos_phis, lock_eps)
        else:
            curve = forward_fn(x)
            loss = loss_fn(curve, target)
        if torch.isfinite(loss):
            loss.backward()
        return loss

    for step in range(n_outer):
        if hasattr(loss_fn, 'set_step'):
            loss_fn.set_step(step)
        loss = optimizer.step(closure)
        if not torch.isfinite(loss):
            break
        history.append(loss.item())
        if verbose and (step % report_every == 0):
            print(f"  outer {step:3d}  loss = {loss.item():.6e}")
    return x.detach(), history


def optimize_lbfgs_nc(forward_fn, x0_flat, target, loss_fn,
                      n_steps=200, lr=1.0, history_size=10,
                      max_ls=20, ls_c1=1e-4, ls_shrink=0.5,
                      tol_g=1e-10, tol_step=1e-14,
                      locking=None, lock_weight=0.05, lock_eps=1e-3,
                      verbose=False, report_every=10):
    """L-BFGS without PyTorch's closure pattern.

    Implements the classical two-loop recursion explicitly. Each outer step
    costs one forward + one backward pass through the simulator; the optional
    Armijo backtracking line search costs up to ``max_ls`` extra forward-only
    passes (no autograd). Set ``max_ls=0`` to disable the line search and
    take a fixed step of size ``lr`` (faster, less robust).

    The two-loop recursion uses the standard scaled-identity initial Hessian
    ``H_0 = gamma * I`` with ``gamma = (s_last . y_last) / (y_last . y_last)``
    (Nocedal & Wright, eq. 7.20). ``history_size`` is the number of (s, y)
    pairs kept in memory.

    Convergence: stop when ``|g| < tol_g`` (gradient near zero) or when the
    accepted step has norm below ``tol_step`` (no progress).

    The simulator is differentiable end-to-end (``simulate_batch`` builds the
    autograd graph), so the single ``loss.backward()`` per outer step is
    sufficient — exactly the same access pattern ``optimize_gd`` and
    ``optimize_adam`` use, just with a smarter descent direction.
    """
    x = x0_flat.clone().detach().requires_grad_(True)
    history = []
    s_list, y_list, rho_list = [], [], []
    x_prev = None
    g_prev = None

    def eval_loss_and_grad():
        if x.grad is not None:
            x.grad.zero_()
        if locking:
            curve, cos_phis = forward_fn(x)
            loss = loss_fn(curve, target) + lock_weight * locking_barrier(cos_phis, lock_eps)
        else:
            curve = forward_fn(x)
            loss = loss_fn(curve, target)
        if not torch.isfinite(loss):
            return loss, None
        loss.backward()
        return loss, x.grad.detach().clone()

    @torch.no_grad()
    def eval_loss_only(x_trial):
        if locking:
            curve, cos_phis = forward_fn(x_trial)
            return (loss_fn(curve, target)
                    + lock_weight * locking_barrier(cos_phis, lock_eps))
        return loss_fn(forward_fn(x_trial), target)

    for step in range(n_steps):
        if hasattr(loss_fn, 'set_step'):
            loss_fn.set_step(step)

        loss, g = eval_loss_and_grad()
        if g is None:
            if verbose:
                print(f"  step {step}: non-finite loss — stopping")
            break
        f_cur = loss.item()
        history.append(f_cur)

        g_norm = g.norm().item()
        if verbose and (step % report_every == 0):
            mem = len(s_list)
            print(f"  step {step:4d}  loss = {f_cur:.6e}  |g| = {g_norm:.2e}  mem = {mem}")
        if g_norm < tol_g:
            if verbose:
                print(f"  step {step}: |g|={g_norm:.2e} < tol_g — converged")
            break

        # ---- update (s, y, rho) memory from the previous step ----
        if x_prev is not None:
            s = x.detach() - x_prev
            y = g - g_prev
            ys = (y * s).sum().item()
            if ys > 1e-10:                  # curvature condition
                if len(s_list) >= history_size:
                    s_list.pop(0); y_list.pop(0); rho_list.pop(0)
                s_list.append(s)
                y_list.append(y)
                rho_list.append(1.0 / ys)

        # ---- two-loop recursion: r = H_k * g ----
        q = g.clone()
        alpha = []
        for s_i, y_i, rho_i in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho_i * (s_i * q).sum().item()
            alpha.append(a)
            q = q - a * y_i
        alpha.reverse()

        if s_list:
            s_last, y_last = s_list[-1], y_list[-1]
            yy = (y_last * y_last).sum().item()
            gamma = (s_last * y_last).sum().item() / yy if yy > 1e-12 else 1.0
            r = gamma * q
        else:
            r = q                            # first step: H_0 = I → gradient descent

        for s_i, y_i, rho_i, a_i in zip(s_list, y_list, rho_list, alpha):
            b = rho_i * (y_i * r).sum().item()
            r = r + (a_i - b) * s_i

        d = -r                               # descent direction

        # ---- Armijo backtracking (no closure: we just call forward_fn) ----
        with torch.no_grad():
            dir_dot_g = (d * g).sum().item()
            if dir_dot_g >= 0:               # not a descent dir → reset to -g
                d = -g
                dir_dot_g = -(g * g).sum().item()

            x_base = x.detach().clone()
            if max_ls <= 0:
                alpha_t = lr                  # fixed step (constant-step L-BFGS)
            else:
                alpha_t = lr
                accepted = False
                for _ in range(max_ls + 1):
                    x_trial = x_base + alpha_t * d
                    f_trial = eval_loss_only(x_trial).item()
                    if (np.isfinite(f_trial)
                            and f_trial <= f_cur + ls_c1 * alpha_t * dir_dot_g):
                        accepted = True
                        break
                    alpha_t *= ls_shrink
                if not accepted:
                    alpha_t = 0.0             # stagnate but don't blow up

            step_norm = (alpha_t * d).norm().item() if alpha_t > 0 else 0.0
            x_prev = x_base
            g_prev = g
            x.data = x_base + alpha_t * d

        if step_norm < tol_step:
            if verbose:
                print(f"  step {step}: step norm {step_norm:.2e} < tol_step — stopping")
            break

    return x.detach(), history


def _lm_step(J, r, lam):
    JtJ = J.T @ J
    Jtr = J.T @ r
    diag = torch.diag(torch.diagonal(JtJ).clamp(min=1e-12))
    try:
        return -torch.linalg.solve(JtJ + lam * diag, Jtr)
    except RuntimeError:
        return None


def optimize_lm(forward_fn, x0_flat, target, residual_fn,
                n_steps=200, lam0=1e-3, lam_max=1e10,
                verbose=False, report_every=10):
    x = x0_flat.clone().detach()
    lam = lam0
    history = []

    def res(xx):
        return residual_fn(forward_fn(xx), target)

    for step in range(n_steps):
        if hasattr(residual_fn, 'set_step'):
            residual_fn.set_step(step)
        r = res(x)
        loss = 0.5 * (r ** 2).sum().item()
        history.append(loss)
        if not np.isfinite(loss):
            break
        if verbose and (step % report_every == 0):
            print(f"  step {step:4d}  loss = {loss:.6e}  lam = {lam:.2e}")
        J = torch.autograd.functional.jacobian(res, x, vectorize=True)
        dx = _lm_step(J, r, lam)
        if dx is None:
            lam = min(lam * 10, lam_max); continue
        x_new = x + dx
        r_new = res(x_new)
        if (r_new ** 2).sum() < (r ** 2).sum():
            x = x_new; lam = max(lam / 10, 1e-12)
        else:
            lam = min(lam * 10, lam_max)
            if lam >= lam_max:
                break
    return x.detach(), history


def optimize_lm_fd(forward_fn, x0_flat, target, residual_fn,
                   n_steps=200, lam0=1e-3, lam_max=1e10, h=1e-5,
                   verbose=False, report_every=5):
    x = x0_flat.clone().detach()
    n = x.numel()
    lam = lam0
    history = []
    with torch.no_grad():
        for step in range(n_steps):
            if hasattr(residual_fn, 'set_step'):
                residual_fn.set_step(step)
            r = residual_fn(forward_fn(x), target)
            loss = 0.5 * (r ** 2).sum().item()
            history.append(loss)
            if not np.isfinite(loss):
                break
            if verbose and (step % report_every == 0):
                print(f"  step {step:4d}  loss = {loss:.6e}  lam = {lam:.2e}")
            m = r.numel()
            J = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for i in range(n):
                xp = x.clone(); xp[i] += h
                xm = x.clone(); xm[i] -= h
                rp = residual_fn(forward_fn(xp), target)
                rm = residual_fn(forward_fn(xm), target)
                J[:, i] = (rp - rm) / (2 * h)
            dx = _lm_step(J, r, lam)
            if dx is None:
                lam = min(lam * 10, lam_max); continue
            x_new = x + dx
            r_new = residual_fn(forward_fn(x_new), target)
            if (r_new ** 2).sum() < (r ** 2).sum():
                x = x_new; lam = max(lam / 10, 1e-12)
            else:
                lam = min(lam * 10, lam_max)
                if lam >= lam_max:
                    break
    return x.detach(), history


# ===========================================================================
# Top-level dispatcher
# ===========================================================================

def optimize_mechanism(
    JJ, PSlice, target,
    motor=[0, 1], fixed_nodes=[0, 1], path_node=None,
    method='lbfgs', metric='ordered_l2', metric_weights=None, residual=None,
    thetas=None, coupler_slot=None,
    device='cpu',
    locking=False, lock_weight=0.05, lock_eps=1e-3,
    tau_anneal=False, tau_init=0.5, tau_final=0.01,
    tau_schedule='exp', tau_n_steps=None,
    normalize_target=True, normalize_curve_=True,
    verbose=False, **method_kwargs,
):
    """End-to-end mechanism optimization.

    ``metric`` may be a single name/callable OR a list/tuple of them. When
    a list is given the losses are summed (use ``metric_weights`` to weight
    them; defaults to equal weights). Soft-chamfer entries inside the list
    still pick up ``tau_anneal=True``. LM-family methods don't support
    composite metrics; use a single metric in ``RESIDUALS`` for those.
    """
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 201)[:200]
    thetas_t = torch.as_tensor(np.asarray(thetas), dtype=DTYPE, device=device)
    target_t = torch.as_tensor(np.asarray(target), dtype=DTYPE, device=device)
    if normalize_target:
        target_t = normalize_curve(target_t)

    JJ_s, PSlice_s, _, fn_s, ord_ = sort_mechanism(JJ, PSlice, motor, fixed_nodes)
    if coupler_slot is None:
        if path_node is not None:
            coupler_slot = int(np.where(ord_ == path_node)[0][0])
        else:
            coupler_slot = -1

    # ---- resolve metric(s) into a single loss_fn ---------------------------
    metric_list = list(metric) if isinstance(metric, (list, tuple)) else [metric]
    n_anneal = (tau_n_steps
                or method_kwargs.get('n_steps')
                or method_kwargs.get('n_outer')
                or 200)
    resolved = []
    for m in metric_list:
        if isinstance(m, str):
            if m not in METRICS:
                raise ValueError(f"Unknown metric: {m!r}; choose from {list(METRICS)}")
            if tau_anneal and m == 'soft_chamfer':
                resolved.append(make_annealed(soft_chamfer_loss,
                                              tau_init=tau_init, tau_final=tau_final,
                                              n_steps=n_anneal, schedule=tau_schedule))
            else:
                resolved.append(METRICS[m])
        else:
            resolved.append(m)
    if len(resolved) == 1:
        loss_fn = resolved[0]
        metric_label = metric_list[0] if isinstance(metric_list[0], str) else 'custom'
    else:
        loss_fn = make_composite_loss(resolved, metric_weights)
        metric_label = '+'.join(m if isinstance(m, str) else 'custom' for m in metric_list)

    if method in ('lm', 'lm_fd'):
        if isinstance(metric, (list, tuple)):
            raise ValueError(
                f"method='{method}' doesn't support composite metrics; "
                f"use a single residual-compatible metric in {list(RESIDUALS)}"
            )
        if callable(residual):
            res_fn = residual
        elif isinstance(residual, str):
            res_fn = RESIDUALS[residual]
        elif isinstance(metric, str) and metric in RESIDUALS:
            res_fn = RESIDUALS[metric]
        else:
            raise ValueError(
                f"method='{method}' needs a residual; pass residual=... or "
                f"use metric in {list(RESIDUALS)}"
            )
        if tau_anneal and (
            metric == 'soft_chamfer'
            or (isinstance(residual, str) and residual == 'soft_chamfer')
            or res_fn is soft_chamfer_residual
        ):
            n_anneal = tau_n_steps or method_kwargs.get('n_steps') or 200
            res_fn = make_annealed(soft_chamfer_residual,
                                   tau_init=tau_init, tau_final=tau_final,
                                   n_steps=n_anneal, schedule=tau_schedule)
        if locking:
            raise ValueError("locking barrier is not supported with LM.")

    use_no_grad = (method == 'lm_fd')
    forward_fn = build_forward(JJ_s, fn_s, thetas_t, device,
                               coupler_slot=coupler_slot,
                               with_locking=bool(locking),
                               no_grad=use_no_grad,
                               normalize=normalize_curve_)

    x0_flat = torch.as_tensor(PSlice_s.flatten(), dtype=DTYPE, device=device)
    common = dict(verbose=verbose)
    lock_kw = dict(locking=bool(locking), lock_weight=lock_weight, lock_eps=lock_eps)

    if method == 'gd':
        x_final, history = optimize_gd(forward_fn, x0_flat, target_t, loss_fn,
                                       **common, **lock_kw, **method_kwargs)
    elif method == 'adam':
        x_final, history = optimize_adam(forward_fn, x0_flat, target_t, loss_fn,
                                         **common, **lock_kw, **method_kwargs)
    elif method == 'lbfgs':
        x_final, history = optimize_lbfgs(forward_fn, x0_flat, target_t, loss_fn,
                                          **common, **lock_kw, **method_kwargs)
    elif method == 'lbfgs_nc':
        x_final, history = optimize_lbfgs_nc(forward_fn, x0_flat, target_t, loss_fn,
                                             **common, **lock_kw, **method_kwargs)
    elif method == 'lm':
        x_final, history = optimize_lm(forward_fn, x0_flat, target_t, res_fn,
                                       **common, **method_kwargs)
    elif method == 'lm_fd':
        x_final, history = optimize_lm_fd(forward_fn, x0_flat, target_t, res_fn,
                                          **common, **method_kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    N = JJ_s.shape[0]
    x_opt_sorted = x_final.reshape(N, 2).cpu().numpy()
    inv = np.empty_like(ord_)
    inv[ord_] = np.arange(N)
    x_opt_original = x_opt_sorted[inv]

    return {
        'x_optimized':         x_opt_original,
        'x_optimized_sorted':  x_opt_sorted,
        'x0_initial_sorted':   PSlice_s,
        'JJ_sorted':           JJ_s,
        'fixed_sorted':        fn_s,
        'ord':                 ord_,
        'history':             history,
        'method':              method,
        'metric':              metric_label,
        'coupler_slot':        coupler_slot,
    }


# ===========================================================================
# Visualizer-facing convenience wrapper
# ===========================================================================

def fit_to_expected_path(
    JJ, PSlice, expected_path,
    motor=[0, 1], fixed_nodes=[0, 1], path_node=None,
    method='lbfgs', metric='soft_chamfer', metric_weights=None,
    n_outer=80, lr=0.3, tau_anneal=True,
    normalize_target=True, normalize_curve_=True,
    verbose=False,
):
    """One-shot "Optimize" call for the visualizer button.

    ``metric`` accepts a single name or a list (e.g. ``['soft_chamfer',
    'hausdorff']``); ``metric_weights`` is an optional per-metric scaling.
    """
    expected_path = np.asarray(expected_path, dtype=float)
    if expected_path.ndim != 2 or expected_path.shape[1] != 2 or len(expected_path) < 3:
        raise ValueError("expected_path must be (M, 2) with M >= 3")

    kw = dict(
        motor=motor, fixed_nodes=fixed_nodes, path_node=path_node,
        method=method, metric=metric, metric_weights=metric_weights,
        tau_anneal=tau_anneal,
        normalize_target=normalize_target,
        normalize_curve_=normalize_curve_,
        verbose=verbose,
    )
    if method == 'lbfgs':
        kw['n_outer'] = n_outer
        kw['lr'] = lr
    elif method == 'lbfgs_nc':
        kw['n_steps'] = n_outer
        kw['lr'] = lr
    elif method in ('gd', 'adam'):
        kw['n_steps'] = n_outer
        kw['lr'] = lr
    elif method in ('lm', 'lm_fd'):
        kw['n_steps'] = n_outer

    return optimize_mechanism(JJ, PSlice, expected_path, **kw)
