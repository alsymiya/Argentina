# Argentina workshop &mdash; differentiable four-bar linkages

A tiny PyTorch toolkit for the Argentina workshop. A differentiable
kinematic simulator for planar four-bar linkages, a curve-matching
optimizer that fine-tunes joint positions via L-BFGS, a freehand-drawing
canvas for hand-drawn target curves, and a thin client that talks to the
MotionGen image-based path-synthesis backend.

---

## Run the demo &mdash; one click per notebook

Click any badge below. The notebook opens in Google Colab on your own
account; the Setup cell auto-downloads the helper files from this repo.
No install, no Drive, no copy-paste.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alsymiya/Argentina/blob/main/1.%20draw_demo%20%28Colab%29.ipynb)
&nbsp; **1. Draw a curve and watch it get normalized.**
Freehand-draw a curve in the canvas. See it go through the four-stage
geometric normalization (center &rarr; scale &rarr; PCA-rotate &rarr; reflect)
that maps every congruent/scaled/mirrored version of the same intrinsic
shape into the same canonical pose.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alsymiya/Argentina/blob/main/2.%20example_fourbar%20%28Colab%29.ipynb)
&nbsp; **2. Differentiable simulator + optimizer.**
The end-to-end demo: simulate the wishlist 4-bar, perturb its joints,
fit it back to the original trace via L-BFGS, then a metric study
(soft-chamfer / Hausdorff / soft-DTW) with wall-clock timings.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alsymiya/Argentina/blob/main/3.%20fourbar_compare%20%28Colab%29.ipynb)
&nbsp; **3. Compare two four-bars.**
You supply two configurations (initial + modified). The notebook
simulates both, fine-tunes the modified one toward the initial trace,
plots them side-by-side with a configurable displacement, and prints
MotionGen JSON for both.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alsymiya/Argentina/blob/main/4.%20drawing_to_mechanism%20%28Colab%29.ipynb)
&nbsp; **4. Drawing &rarr; MotionGen retrieval &rarr; differentiable refinement.**
The merged demo. Draw a curve &rarr; POST to the MotionGen path-synthesis
server &rarr; get *k* candidate four-bars &rarr; simulate each on top of the
drawing &rarr; pick one &rarr; differentiable refine &rarr; export MotionGen JSON
(text on clipboard) + BEFORE/AFTER image.

That's the whole demo. The Setup cell in each notebook handles the rest.

---

## Run locally (optional)

If you want a persistent copy on your machine instead of Colab:

```
git clone https://github.com/alsymiya/Argentina.git
cd Argentina
pip install -r requirements.txt
jupyter notebook
```

Then open the *non-* `(Colab)` versions of the notebooks &mdash; same content,
but without the GitHub-download Setup cell. They expect the helper `.py`
files to sit in the same folder.

---

## Going deeper

This section is for people extending the code, not workshop attendees.

### Python modules

| File | Role |
|---|---|
| `simulator.py` | Differentiable dyadic kinematic solver. `simulate_batch`, `simulate_batch_no_grad`, `reachable_arc`, `sort_mechanism`. |
| `optimizer.py` | Metrics (`ordered_l2`, `chamfer`, `hausdorff`, `soft_chamfer`, `soft_dtw`), composite-loss support, `optimize_lbfgs_nc` (closure-free L-BFGS), `fit_to_expected_path` wrapper, `normalize_curve` (bbox-normalize, torch, differentiable). |
| `visualizer.py` | Interactive matplotlib viewer with drag-to-move joints + Optimize button + save image / save config. Also exports the headless `simulate()` helper. |
| `motiongen_export.py` | `build_motiongen_json` &rarr; MotionGen JSON dict. `copy_to_clipboard` for text (Win32 ctypes, dodges the BOM bug). |
| `normalize.py` | Geometric pose canonicalization (center / scale / PCA-rotate / 3rd-moment reflect). Pure numpy. *Not differentiable* &mdash; different job from `optimizer.normalize_curve`. |
| `bsi_converter.py` | Converts a server-side BSI candidate (`{B, S, I, p, c, error}`) into the Argentina linkage dict (`{JJ, PSlice, motor, fixed_nodes, path_node}`). RRRR + rotary actuator only. |

### CLI tools (advanced, not used by any notebook)

- `generate_dataset.py` &mdash; mass-produce four-bar samples via batched simulation; saves `dataset.npz` with configs/paths/fractions/modes. `python generate_dataset.py --help`.
- `novelty.py` &mdash; novelty scoring over a dataset (link lengths + cross-product features, k-NN distance in standardised feature space).

### References

[`PAPERS.md`](PAPERS.md) has a short curated list of the papers that inform
the code &mdash; Cuturi & Blondel on soft-DTW, Liu & Nocedal on L-BFGS,
Burmester, McCarthy & Soh, geometric-invariant normalization.

### The two normalisations (easy to confuse)

There are two functions called `normalize*` in this repo and they do
different things:

- **`normalize.normalize(curve)`** is geometric *pose canonicalization*.
  PCA rotation, third-moment reflection. Used at the user-input boundary
  (e.g. the drawn curve before a MotionGen KDTree lookup). **Not
  differentiable** &mdash; eigenvector orientations flip discontinuously near
  isotropic covariances, the sign-of-moments step is a step function.
- **`optimizer.normalize_curve(curve)`** is a six-line *bounding-box
  rescaling* on torch primitives. It lives *inside* the L-BFGS forward
  pass and the gradient of the loss has to flow through it back to the
  joint positions. Its job is to make the loss scale-invariant so the
  optimizer doesn't trivially shrink the linkage to zero size.

Don't try to swap one for the other.

---

## Acknowledgements

The differentiable simulator (`simulator.py`) is based on Amin Heyrani Nobari's
linkage project from MIT 18.337:
[ahnobari/18337-Linakge-Project](https://github.com/ahnobari/18337-Linakge-Project).
Our modifications to that code are mostly in two places:

- **Branch determination.** Handling of non-Grashof / non-rotatable mechanisms &mdash;
  picking and tracking one assembly mode across the theta sweep, returning
  `NaN` for unreachable samples, and the `reachable_arc` helper for joining
  wrap-around arcs into one continuous curve.
- **Variable naming.** Renamed to match MotionGen's own semantics
  (`JJ`, `PSlice`, `motor`, `fixed_nodes`, `path_node`, etc.) so the same
  field names round-trip between the Python simulator and the MotionGen
  JSON / BSI format.

Thank you to Amin for making the original implementation available.
