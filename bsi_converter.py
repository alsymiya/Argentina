"""
bsi_converter.py - Convert a MotionGen path-synthesis backend response
(BSI dict + joint positions) into the Argentina simulator's format
(JJ, PSlice, motor, fixed_nodes, path_node).

Scope: RRRR four-bars with a rotary actuator.

Backend response shape (per candidate):
    {
        "B": [[0/1, ...], ...],  # n_bodies x n_joints  -- body membership
        "S": [[...], ...],       # slots ([] for RRRR; non-empty == prismatic)
        "I": [[from, at, to, act_type]],  # actuator. act_type 0 = rotary, 1 = linear.
        "c": [0/1, ...],         # n_joints -- 1 at the coupler / path node
        "p": [[x, y], ...],      # joint positions in user frame
        "error": <float>         # k-NN distance from the query in latent space
    }

Argentina format (mirrors visualizer.load_config_txt):
    {
        "JJ":          np.ndarray(int)   (n, n)   1 = joints share a moving link
        "PSlice":      np.ndarray(float) (n, 2)
        "motor":       [pivot_joint, driven_joint]
        "fixed_nodes": [ground_joint_idx, ...]
        "path_node":   int
    }
"""

from __future__ import annotations
import numpy as np


def is_rrrr(bsi):
    """5-joint mechanism with no slots = RRRR four-bar."""
    return not bsi["S"] and len(bsi["c"]) == 5


def actuator_is_rotary(bsi):
    """I[0][3] = 0 -> rotary; 1 -> linear."""
    return int(bsi["I"][0][3]) == 0


def bsi_to_argentina(bsi):
    """Convert one server-side BSI candidate to the Argentina linkage dict."""
    if not is_rrrr(bsi):
        raise ValueError("bsi_to_argentina: candidate is not RRRR")
    if not actuator_is_rotary(bsi):
        raise ValueError("bsi_to_argentina: actuator is linear; only rotary supported")

    B      = np.asarray(bsi["B"], dtype=int)
    PSlice = np.asarray(bsi["p"], dtype=float)
    c      = bsi["c"]
    n      = len(c)

    path_node = int(np.argmax(c))

    motor_from, motor_at, motor_to, _ = bsi["I"][0]
    motor_from, motor_at, motor_to = int(motor_from), int(motor_at), int(motor_to)
    motor = [motor_at, motor_to]

    # Ground body = the row of B that contains both motor_at and motor_from.
    ground_body_idx = next(
        k for k in range(B.shape[0])
        if B[k, motor_at] and B[k, motor_from]
    )
    fixed_nodes = sorted(int(j) for j in range(n) if B[ground_body_idx, j])

    # Joint-joint adjacency. 1 iff some MOVING body contains both joints.
    JJ = np.zeros((n, n), dtype=int)
    for k in range(B.shape[0]):
        if k == ground_body_idx:
            continue
        joints_in_body = [j for j in range(n) if B[k, j]]
        for i in joints_in_body:
            for j in joints_in_body:
                if i != j:
                    JJ[i, j] = 1

    return {
        "JJ":          JJ,
        "PSlice":      PSlice,
        "motor":       motor,
        "fixed_nodes": fixed_nodes,
        "path_node":   path_node,
    }


def filter_rrrr(solutions):
    """Drop non-RRRR solutions from a server response. Preserves order."""
    return [s for s in solutions if is_rrrr(s)]


def convert_response(server_response):
    """Convert an entire server response into a list of Argentina configs."""
    solutions = (server_response["solutions"]
                 if isinstance(server_response, dict)
                 else server_response)

    out = []
    for s in solutions:
        if not is_rrrr(s) or not actuator_is_rotary(s):
            continue
        cfg = bsi_to_argentina(s)
        cfg["error"] = float(s.get("error", float("nan")))
        out.append(cfg)
    return out
