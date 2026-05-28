# Argentina workshop — differentiable four-bar linkage demo

A small, self-contained PyTorch toolkit for the Argentina workshop:
a differentiable simulator for planar four-bar linkages, a curve-matching
optimizer that fine-tunes joint positions via L-BFGS, a freehand-drawing
canvas for hand-drawn target curves, and a glue layer that talks to the
MotionGen image-based path-synthesis backend.

## Quickstart

```
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in numeric order. They build on each other but each
runs independently.

## Notebooks (run in order)

| # | Notebook | What it does |
|---|---|---|
| 1 | `1. draw_demo.ipynb`            | Freehand-draw a curve, watch the four-stage geometric normalization (center → scale → rotate-by-PCA → reflect-by-3rd-moment) lay it into canonical pose. Pure numpy, no simulator. |
| 2 | `2. example_fourbar.ipynb`      | The end-to-end optimizer demo: simulate the wishlist 4-bar, perturb its joints, fit it back to the original trace via L-BFGS, then a metric study (soft-chamfer / hausdorff / soft-DTW) with wall-clock timings. |
| 3 | `3. fourbar_compare.ipynb`      | Applied workflow: you supply two configurations (initial + modified), the notebook simulates both, fine-tunes the modified one toward the initial trace, plots side-by-side with a configurable displacement, and prints MotionGen JSON for both. |
| 4 | `4. drawing_to_mechanism.ipynb` | The merged demo: draw a curve → POST to the MotionGen path-synthesis server → get k candidate four-bars → simulate each on top of the drawing → pick one → differentiable refine → export MotionGen JSON (text on clipboard) + BEFORE/AFTER image (Windows clipboard via CF_DIB, or PNG file). |

Notebooks 1 and 4 use `%matplotlib widget` (`ipympl`). They have a
defensive setup cell that installs ipympl if missing and prompts you to
restart the kernel/runtime — that prompt is load-bearing on Colab,
don't skip it.

## Python modules

| File | Purpose |
|---|---|
| `simulator.py`       | Differentiable dyadic kinematic solver. `simulate_batch`, `simulate_batch_no_grad`, `reachable_arc`, `sort_mechanism`. |
| `optimizer.py`       | Metrics (`ordered_l2`, `chamfer`, `hausdorff`, `soft_chamfer`, `soft_dtw`), composite-loss support, `optimize_lbfgs_nc` (closure-free L-BFGS), `fit_to_expected_path` one-shot wrapper, `normalize_curve` (bbox-normalize, torch, differentiable). |
| `visualizer.py`      | Interactive matplotlib viewer with drag-to-move joints, an Optimize button, save image / save config. Also exports the headless `simulate()` helper. |
| `motiongen_export.py`| `build_motiongen_json` → MotionGen JSON dict. `copy_to_clipboard` for text. Direct Win32 ctypes path that avoids the BOM bug that bit us with `clip.exe` + utf-16. |
| `normalize.py`       | Geometric pose canonicalization (center / scale / PCA-rotate / 3rd-moment reflect). Pure numpy. *Not differentiable* — different job from `optimizer.normalize_curve`. |
| `bsi_converter.py`   | Converts a server-side BSI candidate (`{B, S, I, p, c, error}`) to the Argentina linkage dict (`{JJ, PSlice, motor, fixed_nodes, path_node}`). RRRR + rotary actuator only. |

## CLI tools (standalone)

| File | Purpose |
|---|---|
| `generate_dataset.py` | Mass-produce four-bar samples via `simulate_batch_no_grad`. Saves `dataset.npz` with `configs`, `paths`, `fractions`, `modes`. `python generate_dataset.py --help` for options. |
| `novelty.py`          | Novelty scoring for an existing dataset (link lengths + cross-product features, k-NN distance in standardized feature space). |

Neither is imported by any notebook — they're for offline data work.

## Config / data files

| File | Purpose |
|---|---|
| `example_config.txt` | JSON-with-comments holding the wishlist 4-bar (5 joints, RRRR, rotary actuator at joint 0 driving joint 1, path node 4). Notebooks 2 and 3 use it as the canonical reference. |
| `PAPERS.md`          | Curated references that inform the code (Cuturi & Blondel on soft-DTW, Liu & Nocedal on L-BFGS, Burmester, McCarthy & Soh, geometric-invariant normalization). |
| `requirements.txt`   | Pip dependencies with per-package rationale comments. |

## Running on Google Colab

1. Upload the four notebooks + every `.py` file to the same Colab directory.
2. Open the notebook you want, run the Setup cell first.
3. If the Setup cell prints "ipympl installed — Restart session, then re-run this cell", do exactly that: `Runtime > Restart session`, re-run only the Setup cell. The `raise SystemExit` is intentional — it stops "Run all" from continuing on a stale matplotlib backend.

Notebook #4 (`drawing_to_mechanism`) additionally POSTs to a public
MotionGen synthesis endpoint baked into the Configuration cell. You can
swap it for a different URL if you're running your own backend.

## A note on the two `normalize` functions

There are two functions called `normalize*` in this repo and they do
different things — easy to confuse:

- **`normalize.normalize(curve)`** (this module) is *geometric pose
  canonicalization*: PCA rotation, third-moment reflection. Used at
  the user-input boundary (e.g., the drawn curve before a MotionGen
  KDTree lookup). **Not differentiable** — eigenvector orientations
  flip discontinuously near isotropic covariances, the sign-of-moments
  step is a step function.

- **`optimizer.normalize_curve(curve)`** is a six-line *bounding-box
  rescaling* built on torch primitives. It lives **inside** the L-BFGS
  forward pass and the gradient of the loss has to flow back through
  it to the joint positions. Its job is to make the loss scale-invariant
  so the optimizer doesn't trivially shrink the linkage to zero size.

Don't try to swap one for the other.
