# Papers worth reading — Argentina demo references

A curated, deliberately short list grouped by the part of the demo each
paper informs.  These are the references I'd want at hand when reading the
Argentina code, or when extending it.  I left out items I'm not personally
confident about; the field is large and your lab will know more recent
work than this list captures.

---

## Loss functions for path / curve matching

**Cuturi & Blondel (2017) — *Soft-DTW: a Differentiable Loss Function for
Time-Series.* ICML.**
The paper that defines `soft_dtw_loss` in `optimizer.py`. Replaces the hard
min in classical DTW with a temperature-controlled log-sum-exp softmin so
the alignment cost is differentiable everywhere. The `gamma` knob plays the
same role as the soft-chamfer `tau`; as `gamma → 0` you recover hard DTW.
Section 4 has the explicit forward/backward recursions used in our O(N·M)
DP implementation.

**Sakoe & Chiba (1978) — *Dynamic Programming Algorithm Optimization for
Spoken Word Recognition.* IEEE Trans. Acoust. Speech.**
The original Dynamic Time Warping reference. Worth reading to internalize
the monotonic-alignment story that motivates soft-DTW in path synthesis:
DTW respects sequence order, so it complements nearest-neighbor chamfer
when the target curve has a consistent parameterization.

**Fan, Su, Guibas (2017) — *A Point Set Generation Network for 3D Object
Reconstruction from a Single Image.* CVPR.**
The paper that popularized the Chamfer distance as a learning loss on point
clouds. The bidirectional formulation (each point in A → nearest in B, and
vice versa, summed) is exactly what `chamfer_loss` and the symmetric
soft-chamfer in `soft_chamfer_loss` use.

**Cuturi (2013) — *Sinkhorn Distances: Lightspeed Computation of Optimal
Transport.* NeurIPS.**
The intellectual ancestor of soft-chamfer. If you ever want to replace the
bidirectional softmin matching with a true optimal-transport coupling
(differentiable, batched, GPU-friendly via Sinkhorn iterations), this is
the starting point.

---

## Optimization

**Liu & Nocedal (1989) — *On the Limited Memory BFGS Method for Large
Scale Optimization.* Mathematical Programming.**
The classical reference for L-BFGS. The two-loop recursion I implemented in
`optimize_lbfgs_nc` comes directly from this paper (their Algorithm 2.1).
Useful to understand the curvature-condition check (`y·s > 0`) and the
scaled-identity initial Hessian `H₀ = γI` choice.

**Nocedal & Wright (2006) — *Numerical Optimization* (2nd ed.). Springer.**
The textbook. Sections 7.2–7.3 (L-BFGS), 10.3 (Levenberg–Marquardt),
3.5 (Armijo / Wolfe line search). Anything you'd want to know about the
optimizers in `optimizer.py` is here.

**Marquardt (1963) — *An algorithm for least-squares estimation of
nonlinear parameters.* SIAM J. Appl. Math.**
Levenberg–Marquardt as classically stated. Pair with Levenberg's 1944 note
for the damping idea. The `_lm_step` helper in `optimizer.py` is the
modern matrix form of their update rule.

**Kingma & Ba (2015) — *Adam: A Method for Stochastic Optimization.*
ICLR.**
Reference for `optimize_adam`. In our setting (small problem, batch
size 1, deterministic) Adam isn't usually the right tool, but it's a
useful baseline because almost everyone in ML reaches for it first.

---

## Kinematic synthesis foundations

**Burmester (1888) — *Lehrbuch der Kinematik.***
The 19th-century foundation of dimensional synthesis: Burmester points,
circle and centerpoint curves, the geometric structure underlying the
"given five precision points, find a 4-bar" problem. Worth a skim even
if your day-to-day work is optimization-based — Burmester theory is the
backstop result you compare against.

**Roberts (1875) — *On Three-Bar Motion in Plane Space.* Proc. Lond.
Math. Soc.**
Roberts' cognate theorem: every coupler curve traced by a four-bar
linkage is traced by exactly three different four-bars. The non-
uniqueness of the design solution is part of what makes path synthesis
hard for optimizers (multiple deep minima of the loss surface).

**McCarthy & Soh (2010) — *Geometric Design of Linkages* (2nd ed.).
Springer.**
The modern reference textbook on kinematic synthesis. Chapters on path
synthesis (4-bar, slider-crank, spherical and spatial cases) and the
loop-equation formulation. The clearest single source if you want to
extend Argentina to spatial mechanisms.

**Norton (2020) — *Design of Machinery.* McGraw-Hill.**
Undergraduate-level but the chapters on Grashof's law, coupler curve
classification, transmission angle, and toggle positions are the most
practical reference I know for sanity-checking what the simulator is
doing on a given config.

---

## Differentiable physics / simulation (the wider context)

**de Avila Belbute-Peres, Smith, Allen, Tenenbaum, Kolter (2018) —
*End-to-End Differentiable Physics for Learning and Control.* NeurIPS.**
One of the early, cleanest demonstrations that running a physics
solver inside an autograd graph and back-propagating through it works.
The Argentina simulator is a tiny instance of the same idea applied to
planar dyadic kinematics.

**Hu et al. (2020) — *DiffTaichi: Differentiable Programming for
Physical Simulation.* ICLR.**
Differentiable physical simulators at scale, with explicit handling of
the gradient-NaN landmines (think `acos` boundaries, contact
discontinuities). Section on "smoothing non-differentiable operations"
is the conceptual sibling of our `safe_acos` clamp.

**Geilinger et al. (2020) — *ADD: Analytically Differentiable Dynamics
for Multi-Body Systems with Frictional Contact.* ACM TOG.**
A different design point — instead of relying on autograd, ADD derives
analytical gradients of a multi-body simulator. Useful comparison
reading if you ever decide the autograd-through-loop approach in
Argentina is too slow for production-scale mechanism design.

---

## Data-driven mechanism design (recent, partial)

I'm deliberately not listing specific recent ML+mechanism-synthesis papers
here because (a) your own lab publishes in this space and you know the
literature better than I do, and (b) I'd rather not name papers I can't
double-check the bibliography of. Two thematic anchors you'd want to keep
in view:

- **Mechanism synthesis as a generative-model problem** — work that
  trains networks to map a desired coupler curve to a candidate
  linkage's `JJ` + link-lengths + initial configuration in one shot,
  using the differentiable simulator only for a final fine-tune. The
  Argentina perturb-and-recover demo in `example_fourbar.ipynb` is a
  miniature of the fine-tune step.

- **Probabilistic / VAE-style mechanism search** — papers that
  represent the design space (topology + dimensions) as a latent
  distribution and sample candidates. Useful complement to gradient
  optimization, which is local; sampling gives you starts that the
  optimizer can then refine.

If you can point me to the specific recent papers your lab considers
canonical here, I'll fold them in with proper citations.
