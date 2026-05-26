"""
visualizer.py — Interactive linkage explorer for the Argentina demo.

What the user sees
------------------
- A canvas with the mechanism drawn from a fixed kinematic structure
  (``JJ``, ``fixed_nodes``, ``path_node``, ``motor``) and an editable
  initial joint configuration (``PSlice``).
- The path traced by the path-node joint (live). For non-rotatable
  mechanisms only the reachable arc is shown, and a wrap-around arc
  (e.g. samples [100..T-1] + [0..10]) is rejoined into one curve.
- An optional "expected path" overlaid in red.
- Click-and-drag on any joint to move its initial location. The mechanism
  is re-simulated automatically on release.

Buttons
-------
- Optimize       : fit PSlice so the path-node trace matches the
                   expected path (uses optimizer.fit_to_expected_path).
- Draw Path      : toggle — left-clicks add points to the expected path.
- Load Path      : load expected path from a CSV / .txt file.
- Clear Path     : remove the expected path.
- Reset          : revert PSlice to the initial values this session started
                   with.
- Save Image     : screenshot the canvas as a PNG.
- Save Config    : dump JJ, fixed_nodes, path_node, motor and PSlice to a
                   .txt (JSON-formatted, ready for motiongen_export.py).

Run
---
    python visualizer.py                      # built-in 4-bar example
    python visualizer.py --config cfg.txt     # load a saved config

The kinematic structure (JJ, fixed_nodes, path_node, motor) stays fixed
during a session; only PSlice is editable.
"""

import argparse
import json
import os
import sys
import datetime as dt

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle

import torch

from simulator import sort_mechanism, simulate_batch_no_grad, reachable_arc
from optimizer import fit_to_expected_path


DTYPE = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Default mechanism — the four-bar with moving point from wishlist.docx.
# ---------------------------------------------------------------------------
#
# Joints (original indexing):
#   0  (0, 0)    ground / motor pivot
#   1  (1, 2)    driven crank tip (actuator joint)
#   2  (4, 0)    other ground pivot
#   3  (3, 2)    moving joint connecting to ground 2
#   4  (2, 3)    path node on the coupler body {1, 3, 4}
#
# Edges: 0-1 (crank), 2-3 (rocker), 1-3 (coupler), 1-4 and 3-4 (path triangle).

DEFAULT_JJ = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0],
])
DEFAULT_PSLICE = np.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [4.0, 0.0],
    [3.0, 2.0],
    [2.0, 3.0],
])
DEFAULT_MOTOR = [0, 1]
DEFAULT_FIXED = [0, 2]
DEFAULT_PATH_NODE = 4


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate(JJ, PSlice, motor, fixed_nodes, path_node, n_theta=200):
    """Return (positions (N, T, 2), path-node trace (T, 2))."""
    JJ_s, PSlice_s, _, fn_s, ord_ = sort_mechanism(JJ, PSlice, motor, fixed_nodes)
    N = JJ_s.shape[0]
    n_t = np.zeros([N, 1]); n_t[fn_s] = 1
    JJs = torch.as_tensor(JJ_s[None], dtype=DTYPE)
    PSls = torch.as_tensor(PSlice_s[None], dtype=DTYPE)
    nts = torch.as_tensor(n_t[None], dtype=DTYPE)
    thetas = torch.linspace(0, 2 * np.pi, n_theta + 1, dtype=DTYPE)[:n_theta]
    sol = simulate_batch_no_grad(JJs, PSls, nts, thetas)[0].cpu().numpy()  # (N, T, 2)

    inv = np.empty_like(ord_)
    inv[ord_] = np.arange(N)
    sol_original = sol[inv]
    return sol_original, sol_original[path_node]


# ---------------------------------------------------------------------------
# Config I/O (the .txt format the MotionGen exporter consumes)
# ---------------------------------------------------------------------------

def save_config_txt(path, JJ, PSlice, motor, fixed_nodes, path_node):
    """Write a JSON-formatted .txt config compatible with motiongen_export.py."""
    cfg = {
        "JJ":          np.asarray(JJ, dtype=int).tolist(),
        "PSlice":      np.asarray(PSlice, dtype=float).tolist(),
        "motor":       list(map(int, motor)),
        "fixed_nodes": list(map(int, fixed_nodes)),
        "path_node":   int(path_node),
    }
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Linkage configuration — JSON object.\n")
        f.write("# Keys: JJ, PSlice, motor, fixed_nodes, path_node.\n")
        json.dump(cfg, f, indent=2)
        f.write("\n")


def load_config_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f.readlines() if not ln.lstrip().startswith('#')]
    cfg = json.loads("".join(lines))
    return (
        np.asarray(cfg["JJ"], dtype=int),
        np.asarray(cfg["PSlice"], dtype=float),
        list(cfg["motor"]),
        list(cfg["fixed_nodes"]),
        int(cfg["path_node"]),
    )


def load_path_file(path):
    """Load (M, 2) expected-path points from .csv/.txt (comma or whitespace)."""
    arr = np.loadtxt(path, delimiter=None if path.endswith('.txt') else ',', ndmin=2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected a (M, 2) array of points; got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# The interactive app
# ---------------------------------------------------------------------------

class LinkageVisualizer:

    JOINT_RADIUS = 0.12         # data units — generous click target
    SAVE_DIR = HERE              # all outputs land in Argentina/

    def __init__(self, JJ, PSlice, motor, fixed_nodes, path_node):
        self.JJ = np.array(JJ, dtype=int)
        self.PSlice = np.array(PSlice, dtype=float)
        self.PSlice_initial = self.PSlice.copy()
        self.motor = list(map(int, motor))
        self.fixed_nodes = list(map(int, fixed_nodes))
        self.path_node = int(path_node)

        self.expected_path = []        # list of [x, y]
        self.draw_mode = False         # toggled by the "Draw Path" button

        # Drag state
        self._drag_joint = None
        self._cid_press = None
        self._cid_release = None
        self._cid_motion = None

        # ------------------------------------------------------------------
        # Figure layout
        # ------------------------------------------------------------------
        self.fig = plt.figure(figsize=(12, 7.5))
        self.fig.canvas.manager.set_window_title("Linkage visualizer — Argentina demo")
        self.ax = self.fig.add_axes([0.06, 0.10, 0.68, 0.85])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(
            f"Mechanism (N={self.JJ.shape[0]}, "
            f"path_node={self.path_node}, motor={self.motor}, fixed={self.fixed_nodes})"
        )
        self.status_text = self.fig.text(0.06, 0.02, "", fontsize=10, color='#444444')

        # Buttons live on a right-side strip.
        self._build_buttons()

        # First draw
        self._artists_links = []
        self._artists_joints = []
        self._artists_labels = []
        self.path_line, = self.ax.plot([], [], color='#1f77b4', lw=2.0,
                                       label='path-node trace')
        self.expected_line, = self.ax.plot([], [], color='red', lw=1.6,
                                            ls='--', marker='o', ms=3.5,
                                            label='expected path')
        self.ax.legend(loc='upper right', fontsize=9)
        self._redraw_all()

        # Mouse handlers
        self._cid_press = self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self._cid_release = self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    def _build_buttons(self):
        x = 0.78
        w = 0.18
        h = 0.06
        gap = 0.015
        y = 0.88
        specs = [
            ("Optimize",       self._on_optimize,      '#cce5ff'),
            ("Draw Path",      self._on_toggle_draw,   '#fff3cd'),
            ("Load Path",      self._on_load_path,     '#e2e3e5'),
            ("Clear Path",     self._on_clear_path,    '#f8d7da'),
            ("Reset PSlice",   self._on_reset,         '#e2e3e5'),
            ("Save Image",     self._on_save_image,    '#d4edda'),
            ("Save Config",    self._on_save_config,   '#d4edda'),
        ]
        self.buttons = {}
        for label, cb, color in specs:
            ax_b = self.fig.add_axes([x, y, w, h])
            b = Button(ax_b, label, color=color, hovercolor='#bcbcbc')
            b.on_clicked(cb)
            self.buttons[label] = b
            y -= (h + gap)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _redraw_all(self):
        for art in self._artists_links + self._artists_joints + self._artists_labels:
            try:
                art.remove()
            except Exception:
                pass
        self._artists_links.clear()
        self._artists_joints.clear()
        self._artists_labels.clear()

        # links
        N = self.JJ.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                if self.JJ[i, j]:
                    line, = self.ax.plot(
                        [self.PSlice[i, 0], self.PSlice[j, 0]],
                        [self.PSlice[i, 1], self.PSlice[j, 1]],
                        '-', color='black', lw=2.2, alpha=0.55, zorder=2,
                    )
                    self._artists_links.append(line)
        # ground bar between the two fixed pivots
        if len(self.fixed_nodes) >= 2:
            for i in range(len(self.fixed_nodes)):
                for j in range(i + 1, len(self.fixed_nodes)):
                    a, b = self.fixed_nodes[i], self.fixed_nodes[j]
                    line, = self.ax.plot(
                        [self.PSlice[a, 0], self.PSlice[b, 0]],
                        [self.PSlice[a, 1], self.PSlice[b, 1]],
                        ls=(0, (4, 3)), color='#888888', lw=1.5, alpha=0.6, zorder=1,
                    )
                    self._artists_links.append(line)

        # joints — colored by role
        for k in range(N):
            if k in self.fixed_nodes:
                color, marker, size = 'black', 's', 140
            elif k == self.path_node:
                color, marker, size = '#d62728', 'o', 140
            elif k == self.motor[1]:
                color, marker, size = '#ff7f0e', 'o', 120
            else:
                color, marker, size = '#1f77b4', 'o', 100
            sc = self.ax.scatter(
                self.PSlice[k, 0], self.PSlice[k, 1],
                s=size, marker=marker, color=color,
                edgecolor='black', linewidths=0.8, zorder=5,
            )
            self._artists_joints.append(sc)
            lbl = self.ax.annotate(
                str(k), (self.PSlice[k, 0], self.PSlice[k, 1]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=11, fontweight='bold', zorder=6,
            )
            self._artists_labels.append(lbl)

        self._update_path()
        self._update_expected()
        self._fit_axes()
        self.fig.canvas.draw_idle()

    def _update_path(self):
        """Render the path-node trace as the largest reachable arc.

        For fully-rotatable mechanisms this is the full closed coupler curve
        (samples 0..T-1). For non-rotatable mechanisms reachable_arc()
        returns just the largest contiguous block of finite samples, with
        wrap-around joining. The status line reports the mode and range.
        """
        try:
            _, trace = simulate(
                self.JJ, self.PSlice,
                self.motor, self.fixed_nodes, self.path_node,
            )
            arc, info = reachable_arc(trace)
            T = trace.shape[0]

            if info['count'] == 0:
                self.path_line.set_data([], [])
                self._status(
                    "unreachable at every theta — adjust PSlice "
                    "(check link lengths / assembly mode)."
                )
                return

            # Only close the loop visually if it really IS a closed loop
            # (i.e. every theta sample is finite). Wrap-around and open
            # arcs end at swing limits and must NOT be auto-closed.
            if info['mode'] == 'full':
                xs = np.concatenate([arc[:, 0], arc[:1, 0]])
                ys = np.concatenate([arc[:, 1], arc[:1, 1]])
            else:
                xs, ys = arc[:, 0], arc[:, 1]
            self.path_line.set_data(xs, ys)

            pct = 100.0 * info['fraction']
            if info['mode'] == 'full':
                self._status(f"ok — full rotation, {T}/{T} samples (100%).")
            elif info['mode'] == 'wrap':
                # Wrap-around: [start..T-1] joined with [0..end]
                self._status(
                    f"non-rotatable — wrap arc indices "
                    f"[{info['start']}..{T - 1}] U [0..{info['end']}], "
                    f"{info['count']}/{T} samples ({pct:.0f}%)."
                )
            else:  # 'arc'
                self._status(
                    f"non-rotatable — open arc indices "
                    f"[{info['start']}..{info['end']}], "
                    f"{info['count']}/{T} samples ({pct:.0f}%)."
                )
        except Exception as e:
            self.path_line.set_data([], [])
            self._status(f"simulation error: {e}")

    def _update_expected(self):
        if not self.expected_path:
            self.expected_line.set_data([], [])
            return
        ep = np.asarray(self.expected_path)
        self.expected_line.set_data(ep[:, 0], ep[:, 1])

    def _fit_axes(self):
        xs = list(self.PSlice[:, 0])
        ys = list(self.PSlice[:, 1])
        if len(self.path_line.get_xdata()):
            xs += list(self.path_line.get_xdata())
            ys += list(self.path_line.get_ydata())
        if self.expected_path:
            ep = np.asarray(self.expected_path)
            xs += list(ep[:, 0]); ys += list(ep[:, 1])
        # Guard against any non-finite values sneaking in.
        xs = [v for v in xs if np.isfinite(v)]
        ys = [v for v in ys if np.isfinite(v)]
        if not xs or not ys:
            return
        x_lo, x_hi = min(xs), max(xs)
        y_lo, y_hi = min(ys), max(ys)
        pad = max(0.5, 0.15 * max(x_hi - x_lo, y_hi - y_lo))
        self.ax.set_xlim(x_lo - pad, x_hi + pad)
        self.ax.set_ylim(y_lo - pad, y_hi + pad)

    def _status(self, msg):
        self.status_text.set_text(msg)

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------
    def _hit_joint(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        d2 = (self.PSlice[:, 0] - event.xdata) ** 2 + (self.PSlice[:, 1] - event.ydata) ** 2
        k = int(np.argmin(d2))
        return k if d2[k] <= self.JOINT_RADIUS ** 2 else None

    def _on_press(self, event):
        if event.inaxes is not self.ax:
            return
        if event.button != 1:
            return
        if self.draw_mode:
            self.expected_path.append([float(event.xdata), float(event.ydata)])
            self._update_expected()
            self._fit_axes()
            self.fig.canvas.draw_idle()
            self._status(f"expected path: {len(self.expected_path)} pts")
            return
        k = self._hit_joint(event)
        if k is not None:
            self._drag_joint = k
            self._status(f"dragging joint {k}")

    def _on_motion(self, event):
        if self._drag_joint is None or event.inaxes is not self.ax or event.xdata is None:
            return
        self.PSlice[self._drag_joint] = [float(event.xdata), float(event.ydata)]
        self._redraw_all()

    def _on_release(self, event):
        if self._drag_joint is not None:
            self._drag_joint = None

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------
    def _on_toggle_draw(self, event):
        self.draw_mode = not self.draw_mode
        label = "Draw Path: ON" if self.draw_mode else "Draw Path"
        self.buttons["Draw Path"].label.set_text(label)
        self._status("draw mode ON: left-click to add points" if self.draw_mode
                     else "draw mode off")

    def _on_clear_path(self, event):
        self.expected_path = []
        self._update_expected()
        self.fig.canvas.draw_idle()
        self._status("expected path cleared")

    def _on_load_path(self, event):
        path = self._prompt_path(default=os.path.join(self.SAVE_DIR, "expected_path.csv"),
                                 must_exist=True, label="Load expected path from")
        if not path:
            return
        try:
            arr = load_path_file(path)
            self.expected_path = arr.tolist()
            self._update_expected()
            self._fit_axes()
            self.fig.canvas.draw_idle()
            self._status(f"loaded {len(arr)} expected-path points from {os.path.basename(path)}")
        except Exception as e:
            self._status(f"load failed: {e}")

    def _on_reset(self, event):
        self.PSlice = self.PSlice_initial.copy()
        self._redraw_all()
        self._status("PSlice reset to session initial values")

    def _on_save_image(self, event):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self.SAVE_DIR, f"linkage_{ts}.png")
        self.fig.savefig(out, dpi=150, bbox_inches='tight')
        self._status(f"saved image: {out}")

    def _on_save_config(self, event):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self.SAVE_DIR, f"config_{ts}.txt")
        save_config_txt(out, self.JJ, self.PSlice, self.motor,
                        self.fixed_nodes, self.path_node)
        self._status(f"saved config: {out}")

    def _on_optimize(self, event):
        if len(self.expected_path) < 3:
            self._status("optimize: need >= 3 expected-path points (draw or load some first)")
            return
        self._status("optimizing… (this may take a few seconds)")
        self.fig.canvas.draw_idle()
        plt.pause(0.01)
        try:
            res = fit_to_expected_path(
                self.JJ, self.PSlice, np.asarray(self.expected_path),
                motor=self.motor, fixed_nodes=self.fixed_nodes,
                path_node=self.path_node,
                method='lbfgs', metric='soft_chamfer',
                n_outer=80, lr=0.3, tau_anneal=True,
                verbose=False,
            )
            self.PSlice = res['x_optimized']
            self._redraw_all()
            hist = res['history']
            self._status(f"optimize done — final loss = {hist[-1]:.4e}  "
                         f"(start {hist[0]:.4e}, {len(hist)} outer steps)")
        except Exception as e:
            self._status(f"optimize failed: {e}")

    def _prompt_path(self, default, must_exist, label):
        print(f"\n[{label}] (press Enter to use default)")
        print(f"  default: {default}")
        try:
            p = input("  path > ").strip()
        except EOFError:
            return None
        if not p:
            p = default
        if must_exist and not os.path.exists(p):
            print(f"  ! file not found: {p}")
            return None
        return p

    def run(self):
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Interactive linkage visualizer.")
    p.add_argument('--config', type=str, default=None,
                   help="Path to a saved config .txt (default: built-in 4-bar).")
    args = p.parse_args(argv)

    if args.config:
        JJ, PSlice, motor, fixed_nodes, path_node = load_config_txt(args.config)
    else:
        JJ, PSlice = DEFAULT_JJ, DEFAULT_PSLICE
        motor, fixed_nodes, path_node = DEFAULT_MOTOR, DEFAULT_FIXED, DEFAULT_PATH_NODE

    matplotlib.rcParams['toolbar'] = 'toolmanager' if 'toolmanager' in dir(matplotlib) else 'toolbar2'
    LinkageVisualizer(JJ, PSlice, motor, fixed_nodes, path_node).run()


if __name__ == '__main__':
    main()
