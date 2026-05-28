"""
normalize.py - Geometric curve normalization for the Argentina workshop.

Pipeline (Geometric Invariant Curve Normalization, plus PCA alignment):

    1. center           : subtract the centroid
    2. scale            : rescale (either RMS-to-1 or max-distance-to-`scaling`)
    3. rotate_pca       : rotate so the greatest principal axis aligns with +X
    4. reflect_moments  : flip x/y signs (and optionally swap axes) so the
                          third-order moments are non-negative

Each step returns a 3x3 homogeneous transform matrix. The composite
M = M4 @ M3 @ M2 @ M1 maps the input curve into the canonical frame:

    [nc_x  nc_y  1].T == M @ [x  y  1].T

`normalize(curve, return_stages=True)` returns the intermediate curves so
you can plot each stage. Use `apply_transform(curve, T)` to reapply M to
any 2D point set.

No torch, no image rasterization - this is geometry only. See the original
backends for the 64x64 binary-image step.

Reference:
    "Geometric Invariant Curve and Surface Normalization" (the reflection step).
"""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(phi: float) -> np.ndarray:
    """3x3 homogeneous rotation by phi radians."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _rotate_points(curve: np.ndarray, phi: float) -> np.ndarray:
    """Rotate (N,2) points by phi radians. Scales up before the rotation and
    back down afterward; the original backend code did this for numerical
    accuracy on very small curves and we keep it for parity."""
    s = 100.0
    x = curve[:, 0] * s
    y = curve[:, 1] * s
    c, sn = np.cos(phi), np.sin(phi)
    xr = x * c - y * sn
    yr = x * sn + y * c
    return np.column_stack([xr, yr]) / s


# ---------------------------------------------------------------------------
# Stage primitives
# ---------------------------------------------------------------------------

def center(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate so the centroid lies at the origin.

    Returns
    -------
    out : (N, 2) translated curve
    T   : 3x3 translation matrix (out_h = T @ in_h)
    """
    m = np.mean(curve, axis=0)
    T = np.array([[1.0, 0.0, -m[0]],
                  [0.0, 1.0, -m[1]],
                  [0.0, 0.0, 1.0]])
    return curve - m, T


def scale(curve: np.ndarray, scaling: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Rescale the curve.

    Parameters
    ----------
    scaling : float
        - 0.0 (default): normalize by sqrt(var_x + var_y) so the RMS distance
          from the centroid is 1.
        - > 0: rescale so the maximum distance from the origin equals `scaling`.

    Returns
    -------
    out : (N, 2) scaled curve
    T   : 3x3 scale matrix
    """
    if scaling == 0:
        denom = float(np.sqrt(np.var(curve[:, 0]) + np.var(curve[:, 1])))
        if denom <= 0 or not np.isfinite(denom):
            denom = 1.0
        k = 1.0 / denom
    else:
        max_dist = float(np.max(np.linalg.norm(curve, axis=1)))
        if max_dist <= 0 or not np.isfinite(max_dist):
            max_dist = 1.0
        k = scaling / max_dist
    T = np.array([[k,   0.0, 0.0],
                  [0.0, k,   0.0],
                  [0.0, 0.0, 1.0]])
    return curve * k, T


def rotate_pca(curve: np.ndarray,
               tol: float = 1e-4,
               randinit: bool = False,
               rng: np.random.Generator | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the curve so its greatest principal axis aligns with +X.

    If `randinit` is True the curve is first rotated by a random angle to
    break ties in near-isotropic covariances; the returned transform
    composes both rotations.

    Returns
    -------
    out : (N, 2) rotated curve
    T   : 3x3 rotation matrix (composite if randinit was used)
    """
    phi_init = 0.0
    if randinit:
        rng = rng if rng is not None else np.random.default_rng()
        phi_init = float(rng.random()) * 2.0 * math.pi
    R_init = _rotation_matrix(phi_init)
    X0 = _rotate_points(curve, phi_init)

    cx = float(np.mean(X0[:, 0]))
    cy = float(np.mean(X0[:, 1]))
    dx = X0[:, 0] - cx
    dy = X0[:, 1] - cy
    n = X0.shape[0]
    cov_xx = float(np.dot(dx, dx) / n)
    cov_xy = float(np.dot(dx, dy) / n)
    cov_yy = float(np.dot(dy, dy) / n)
    cov = np.array([[cov_xx, cov_xy],
                    [cov_xy, cov_yy]])

    if abs(np.linalg.det(cov)) < tol:
        phi = 0.0
    else:
        eig_val, eig_vec = np.linalg.eig(cov)
        # match the original backend's convention: enforce a positive
        # cross-product between the eigenvectors
        if np.linalg.det(eig_vec) > 0:
            eig_vec[0, :] = -eig_vec[0, :]
        if eig_val[0] > eig_val[1]:
            phi = float(np.arctan2(eig_vec[1, 0], eig_vec[0, 0]))
        else:
            phi = float(np.arctan2(eig_vec[1, 1], eig_vec[0, 1]))

    out = _rotate_points(X0, phi)
    R = _rotation_matrix(phi)
    return out, R @ R_init


def reflect_moments(curve: np.ndarray, eps: float = 1e-5
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Fix the reflection ambiguity left by PCA using third-order moments.

    m12 = sum(x * y * y)  -> its sign decides whether to flip y
    m21 = sum(x * x * y)  -> its sign decides whether to flip x

    If |m12| > |m21|, the dominant moment lives on the wrong axis; we swap
    x and y before applying the sign correction.

    Returns
    -------
    out : (N, 2) reflected (and possibly axis-swapped) curve
    T   : 3x3 transform encoding the reflection/swap
    """
    x = curve[:, 0]
    y = curve[:, 1]
    m12 = float(np.sum(x * y * y))
    m21 = float(np.sum(x * x * y))
    s12 = 1.0 if abs(m12) < eps else float(np.sign(m12))
    s21 = 1.0 if abs(m21) < eps else float(np.sign(m21))
    R2 = np.array([[s12, 0.0],
                   [0.0, s21]])
    if abs(m12) > abs(m21):
        R2 = np.array([[0.0, 1.0], [1.0, 0.0]]) @ R2
    reflected = curve @ R2.T
    T = np.array([[R2[0, 0], R2[0, 1], 0.0],
                  [R2[1, 0], R2[1, 1], 0.0],
                  [0.0,      0.0,      1.0]])
    return reflected, T


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def normalize(curve: np.ndarray,
              scaling: float = 3.5,
              tol: float = 1e-8,
              max_iter: int = 2,
              return_stages: bool = False,
              rng: np.random.Generator | None = None):
    """Run center -> scale -> rotate_pca -> reflect_moments.

    Parameters
    ----------
    curve : (N, 2) array
    scaling : float
        See `scale`. Default 3.5 matches the MotionGen backend.
    tol : float
        Minimum allowed |det(M)| before considering the result degenerate.
    max_iter : int
        Number of random-init retries if the first pass is degenerate.
    return_stages : bool
        If True, also return a dict of intermediate (curve, T) pairs keyed
        by stage name. Useful for visualization.

    Returns
    -------
    nc      : (N, 2) normalized curve
    T       : 3x3 composite transform such that
              `apply_transform(curve, T) == nc` (up to float precision)
    success : True if |det(T)| > tol
    stages  : (when return_stages=True) dict
              {'input', 'centered', 'scaled', 'rotated', 'reflected'} ->
              (curve_at_stage, T_of_just_this_stage)
    """
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] < 2:
        eye = np.eye(3)
        if return_stages:
            return curve, eye, False, {'input': (curve, eye)}
        return curve, eye, False

    def _one_pass(X, randinit=False):
        Xc, M1 = center(X)
        Xs, M2 = scale(Xc, scaling=scaling)
        Xr, M3 = rotate_pca(Xs, randinit=randinit, rng=rng)
        Xf, M4 = reflect_moments(Xr)
        return (Xc, Xs, Xr, Xf), (M1, M2, M3, M4)

    (Xc, Xs, Xr, Xf), (M1, M2, M3, M4) = _one_pass(curve, randinit=False)
    M = M4 @ M3 @ M2 @ M1

    # Retry with random pre-rotations if the transform is degenerate
    if abs(np.linalg.det(M)) * max(scaling, 1.0) < tol:
        current = Xf
        for _ in range(max_iter):
            (Xc, Xs, Xr, Xf), (M1, M2, M3, M4) = _one_pass(current, randinit=True)
            M = (M4 @ M3 @ M2 @ M1) @ M
            current = Xf
            if abs(np.linalg.det(M)) > tol:
                break

    success = abs(np.linalg.det(M)) > tol

    if return_stages:
        stages = {
            'input':     (curve, np.eye(3)),
            'centered':  (Xc, M1),
            'scaled':    (Xs, M2),
            'rotated':   (Xr, M3),
            'reflected': (Xf, M4),
        }
        return Xf, M, success, stages
    return Xf, M, success


def apply_transform(curve: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to an (N, 2) curve."""
    pts = np.asarray(curve, dtype=float).reshape(-1, 2)
    h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (T @ h.T).T[:, :2]
    return out


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """Return the inverse of a 3x3 homogeneous transform. Convenience wrapper."""
    return np.linalg.inv(T)


# ---------------------------------------------------------------------------
# Resampling helper (useful for fixed-length downstream pipelines)
# ---------------------------------------------------------------------------

def resample_polyline(curve: np.ndarray, n: int) -> np.ndarray:
    """Resample an (N, 2) polyline to exactly `n` points by arclength."""
    curve = np.asarray(curve, dtype=float)
    if curve.shape[0] < 2:
        return np.tile(curve, (n, 1))[:n]
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arclen[-1])
    if total <= 0:
        return np.tile(curve[:1], (n, 1))
    t = np.linspace(0.0, total, n)
    x = np.interp(t, arclen, curve[:, 0])
    y = np.interp(t, arclen, curve[:, 1])
    return np.column_stack([x, y])
