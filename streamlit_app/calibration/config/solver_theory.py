"""Long-form pedagogical copy for the Theory tab.

Pure data — zero Streamlit dependency. Decoupled from ``constants.py``
(which holds short metadata consumed by the sidebar) so the prose can
grow without bloating runtime imports.

Two-layer pedagogy:

* **Surface layer** (intuition + mechanics + shines/struggles + key
  properties): aimed at an undergraduate / Bachelor-level reader.
  Strong metaphors, plain English, no required prior knowledge of
  optimisation theory.
* **Deep-dive layer** (default-collapsed expander): aimed at an M2
  quant. KaTeX update rule, convergence conditions, references to the
  canonical papers (Marquardt 1963, Storn-Price 1997, Nelder-Mead 1965,
  Byrd-Lu-Nocedal-Zhu 1995, Nocedal-Wright 2006).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverDeepDive:
    """All long-form content for one solver, consumed by the renderer."""

    key: str
    long_name: str
    intuition_md: str
    mechanics_md: str
    deep_dive_md: str
    update_rule_latex: str | None
    shines: tuple[str, ...]
    struggles: tuple[str, ...]
    properties: dict[str, str]
    references: tuple[tuple[str, str], ...]


# ── Concept primer ───────────────────────────────────────────────────────

CONCEPT_INTRO_MD: str = (
    "**Calibration is optimisation.** You hand the calibrator a vector "
    "of parameters $\\theta$ (e.g. for Heston: $v_0, \\kappa, \\sigma^2, "
    "\\alpha, \\rho$) and a **loss function** $L(\\theta)$ that measures "
    "how far the model's prices or returns are from the market. "
    "A *solver* is an algorithm that searches the parameter space for the "
    "$\\theta^\\star$ that minimises $L$.\n\n"
    "Think of $L$ as the elevation of a hilly landscape and $\\theta$ as "
    "the coordinates of a hiker — the solver is the hiker's strategy for "
    "reaching the lowest valley. **The right strategy depends on the "
    "shape of the terrain.** A smooth bowl rewards a careful Newton-like "
    "approach; a wrinkled landscape full of false valleys requires a "
    "scout that explores widely before zooming in. This page explains "
    "the four solvers shipped in this app and when each one shines."
)

VOCAB_EXPANDER_MD: str = (
    "- **Objective / loss / cost function** — the single number "
    "$L(\\theta)$ the solver tries to make as small as possible.\n"
    "- **Iteration** — one round of the solver's update rule. Typical "
    "calibration runs use anywhere from 20 to a few thousand iterations.\n"
    "- **Convergence** — the solver has stopped because either the loss "
    "stopped improving or the parameters stopped moving (within a "
    "tolerance you set).\n"
    "- **Gradient** $\\nabla L$ — the vector of partial derivatives of "
    "$L$ with respect to each parameter; it points in the direction of "
    "steepest ascent. The solver walks against it to descend.\n"
    "- **Hessian** $\\nabla^2 L$ — the matrix of second derivatives. It "
    "encodes the local curvature of the landscape (how bowl-like it is).\n"
    "- **Local vs global minimum** — a local minimum is the bottom of "
    "the nearest valley; a global minimum is the deepest valley anywhere. "
    "Some solvers can only find the former.\n"
    "- **Derivative-free** — a solver that needs *only* function values, "
    "not gradients. Useful when the loss is noisy, non-differentiable, "
    "or the gradient is too expensive to compute."
)

# ── Three axes framework ─────────────────────────────────────────────────

DECISION_AXES_MD: str = (
    "Our four solvers are not random samples of the optimisation zoo — "
    "they cover **three orthogonal design axes**. Once you understand "
    "where a solver sits on these three axes, its strengths and "
    "weaknesses follow mechanically.\n\n"
    "1. **Local vs Global** — *Where does it search?* A local solver "
    "trusts your initial guess and refines it. A global solver scouts "
    "the entire parameter box, paying the cost of many more function "
    "evaluations in exchange for robustness against local minima.\n"
    "2. **Scalar vs Least-squares** — *What shape of loss does it need?* "
    "Most solvers minimise a scalar $L(\\theta)$. Levenberg-Marquardt "
    "exploits the special **least-squares structure** $L = \\sum_i r_i^2$ "
    "by working directly with the residual vector $r$ and its Jacobian "
    "$J = \\partial r / \\partial \\theta$ — it converges much faster "
    "when this structure is available.\n"
    "3. **Gradient-based vs Derivative-free** — *Does it need the slope?* "
    "Gradient-based solvers ask for $\\nabla L$ at each iterate (cheap "
    "in JAX thanks to autodiff). Derivative-free solvers only call $L$ "
    "as a black box — they work even when no gradient is available, but "
    "they pay for it in iterations."
)

# Each row: (solver, axis_1_local_global, axis_2_loss_shape, axis_3_derivative_info)
AXES_TABLE_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("LM-JAX", "Local", "Least-squares (vector residuals)", "Analytical Jacobian (JAX)"),
    ("DE", "Global", "Scalar", "Derivative-free"),
    ("NM", "Local", "Scalar", "Derivative-free"),
    ("L-BFGS-B", "Local", "Scalar", "Gradient (or finite-diff fallback)"),
)

AXES_CAPTION_MD: str = (
    "These three binary axes carve out a $2 \\times 2 \\times 2$ cube "
    "of eight possible designs. Our four solvers occupy four distinct "
    "corners — together they cover the practical use cases a calibration "
    "engine actually meets."
)


# ── Per-solver deep dives ────────────────────────────────────────────────

_LM_JAX = SolverDeepDive(
    key="LM-JAX",
    long_name="Levenberg-Marquardt (Trust-Region-Reflective) · analytical JAX Jacobian",
    intuition_md=(
        "**Newton's method with a leash.** It asks *where is downhill?*, "
        "takes a confident step assuming the loss is locally a perfect "
        "bowl, and if reality disagrees it shortens the leash and tries "
        "a smaller step."
    ),
    mechanics_md=(
        "LM works on **least-squares problems**: the loss is a sum of "
        "squared residuals $\\sum_i r_i(\\theta)^2$, so the calibrator "
        "exposes both the residual vector $r$ and its Jacobian $J$. At "
        "each iteration LM solves a damped linear system that blends "
        "**Gauss-Newton** (aggressive, assumes a quadratic bowl) and "
        "**gradient descent** (conservative, no curvature assumption). "
        "A damping parameter $\\lambda$ adapts on the fly: if the step "
        "succeeded, $\\lambda$ shrinks (more Gauss-Newton, faster); if "
        "it failed, $\\lambda$ grows (more gradient descent, safer). "
        "In our app the Jacobian comes from JAX autodiff — exact, fast, "
        "and free of finite-difference noise."
    ),
    deep_dive_md=(
        "The trust-region-reflective variant we use (SciPy's `method='trf'` "
        "wrapping the MINPACK heritage) replaces the explicit damping "
        "$\\lambda I$ with an adaptive trust-region radius $\\Delta_k$, "
        "and reflects steps that hit box bounds. The local model is "
        "$m_k(\\delta) = \\tfrac{1}{2}\\|r_k + J_k \\delta\\|_2^2$, "
        "minimised inside $\\|\\delta\\| \\le \\Delta_k$. If the "
        "**actual reduction** matches the **predicted reduction**, "
        "$\\Delta_{k+1}$ grows; otherwise it shrinks. Convergence to a "
        "local minimum is **superlinear** (often quadratic) on smooth "
        "well-posed problems — that is why LM-JAX is the production "
        "default whenever the loss is a least-squares surface. The "
        "only thing it can't do is fit a non-residual scalar like a "
        "GARCH log-likelihood."
    ),
    update_rule_latex=(
        r"\left(J^\top J + \lambda I\right)\,\delta = -J^\top r, "
        r"\qquad \theta_{k+1} = \theta_k + \delta"
    ),
    shines=(
        "Smooth IV-surface calibration with analytical residuals",
        "Excellent local accuracy — superlinear convergence near the basin",
        "JAX Jacobian = noise-free derivatives, cache-friendly",
        "Fastest wall-clock time of the four (typical 20-60 iterations)",
    ),
    struggles=(
        "Cannot fit a scalar loss (no GARCH MLE)",
        "Sensitive to initial guess — bad start ⇒ local minimum",
        "Mildly hostile to discontinuous payoffs or noisy residuals",
        "Needs a Jacobian or it falls back to slow finite differences",
    ),
    properties={
        "Local / Global": "Local",
        "Needs gradient": "Yes (Jacobian, exact via JAX)",
        "Needs residuals": "Yes",
        "Typical iterations": "20 – 60",
        "Compute cost vs LM": "1× (baseline)",
    },
    references=(
        ("Gavin — LM nonlinear least squares (Duke, 2024)",
         "https://people.duke.edu/~hpgavin/lm.pdf"),
        ("NEOS Guide — Levenberg-Marquardt Method",
         "https://neos-guide.org/guide/algorithms/lmm/"),
        ("SciPy `least_squares` (method='trf')",
         "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html"),
        ("Marquardt (1963) — SIAM J. Appl. Math. 11",
         "https://doi.org/10.1137/0111030"),
    ),
)


_DE = SolverDeepDive(
    key="DE",
    long_name="Differential Evolution (population-based stochastic global search)",
    intuition_md=(
        "**Natural selection on parameter vectors.** Each generation, a "
        "population of candidate $\\theta$ vectors mutates by stealing "
        "the difference between two random peers; only the fittest "
        "survives, and the swarm slowly converges on the optimum."
    ),
    mechanics_md=(
        "DE maintains a **population** of $N$ candidate parameter "
        "vectors (typically 15–50 for our problems). For each target "
        "$\\mathbf{x}_i$, it picks three other random members "
        "$\\mathbf{x}_{r_1}, \\mathbf{x}_{r_2}, \\mathbf{x}_{r_3}$ and "
        "builds a **mutant** $\\mathbf{v}_i = \\mathbf{x}_{r_1} + F\\,"
        "(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3})$, where $F \\in [0, 2]$ "
        "is the mutation scale. A **crossover** mixes coordinates of "
        "$\\mathbf{v}_i$ and $\\mathbf{x}_i$ with probability $CR$, and "
        "the resulting trial replaces $\\mathbf{x}_i$ only if it has a "
        "lower loss. The whole population marches downhill in parallel, "
        "naturally diversified across the parameter box."
    ),
    deep_dive_md=(
        "Storn & Price (1997) showed DE was surprisingly competitive "
        "with simulated annealing and genetic algorithms despite its "
        "minimal control parameter set ($N$, $F$, $CR$). The mutation "
        "operator is the secret sauce: because the step "
        "$F(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3})$ is *scaled by the "
        "current population spread*, DE automatically slows down as the "
        "swarm tightens — no annealing schedule required. Convergence "
        "is only **probabilistic**: DE has no termination proof, only "
        "empirical tolerance heuristics. SciPy's `differential_evolution` "
        "uses the `best1bin` strategy by default and re-seeds with Latin "
        "Hypercube sampling. In our calibration stack DE is the "
        "**warm-start workhorse**: run it first to localise the global "
        "basin, then hand off to LM-JAX to polish the last 3 digits."
    ),
    update_rule_latex=(
        r"\mathbf{v}_i = \mathbf{x}_{r_1} + F\,(\mathbf{x}_{r_2} - \mathbf{x}_{r_3}), "
        r"\quad \mathbf{u}_i = \text{crossover}(\mathbf{v}_i, \mathbf{x}_i), "
        r"\quad \mathbf{x}_i \leftarrow \mathbf{u}_i\ \text{if}\ L(\mathbf{u}_i) < L(\mathbf{x}_i)"
    ),
    shines=(
        "Robust against multimodal / rugged loss landscapes",
        "Needs no gradient, no residuals — pure black-box loss",
        "Population diversity sidesteps local minima naturally",
        "Trivially parallel across the population (free speedup)",
    ),
    struggles=(
        "Roughly 10× slower than LM-JAX in wall-clock time",
        "Stochastic — two runs with the same seed-free config diverge",
        "No formal convergence guarantee; uses heuristic tolerances",
        "Final polish is coarse — chain with a local solver afterwards",
    ),
    properties={
        "Local / Global": "Global (stochastic)",
        "Needs gradient": "No",
        "Needs residuals": "No",
        "Typical iterations": "500 – 5000 function evaluations",
        "Compute cost vs LM": "≈ 10×",
    },
    references=(
        ("Cornell Optimization Wiki — Differential Evolution",
         "https://optimization.cbe.cornell.edu/index.php?title=Differential_evolution"),
        ("Rodriguez-Mier — A tutorial on DE with Python",
         "https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/"),
        ("SciPy `differential_evolution` docs",
         "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html"),
        ("Storn & Price (1997) — J. Global Optim. 11",
         "https://doi.org/10.1023/A:1008202821328"),
    ),
)


_NM = SolverDeepDive(
    key="NM",
    long_name="Nelder-Mead simplex (geometric, derivative-free)",
    intuition_md=(
        "**A geometric amoeba.** A simplex (a triangle in 2-D, a tetra"
        "hedron in 3-D, …) flips, stretches, and shrinks itself across "
        "the loss landscape, always discarding its worst-performing "
        "corner — oozing downhill toward a local minimum."
    ),
    mechanics_md=(
        "In $p$ dimensions the solver maintains a **simplex** of $p+1$ "
        "vertices. At each iteration it ranks them by their loss and "
        "tries four moves in sequence: **reflect** the worst vertex "
        "through the centroid of the others; if the reflection improves "
        "the best so far, **expand** further in that direction; if it's "
        "worse than the second-worst, **contract** inward; and if "
        "contraction fails too, **shrink** the whole simplex toward the "
        "best vertex. Watching the simplex on a 2-D Rosenbrock looks "
        "exactly like an amoeba navigating a narrow valley."
    ),
    deep_dive_md=(
        "Nelder & Mead (1965) introduced this simplex method long "
        "before autodiff existed, and it remains the canonical example "
        "of a **derivative-free** local optimiser. The geometric moves "
        "are: reflection coefficient $\\alpha = 1$, expansion $\\gamma "
        "= 2$, contraction $\\rho = 0.5$, shrink $\\sigma = 0.5$. "
        "Convergence is **not guaranteed**: McKinnon (1998) constructed "
        "smooth functions where the simplex degenerates and converges "
        "to a non-stationary point, which is why high-dimensional or "
        "ill-conditioned problems can stall. In practice NM is great as "
        "a sanity check on a gradient-based result (does a black-box "
        "method agree?), or when the loss is too noisy / discontinuous "
        "for gradient-based methods to work at all."
    ),
    update_rule_latex=None,
    shines=(
        "Tiny code footprint — no derivatives, no Hessian, no Jacobian",
        "Intuitive geometric behaviour you can literally draw on paper",
        "Robust to mildly noisy or non-smooth loss surfaces",
        "Great pedagogical baseline and sanity-check on harder solvers",
    ),
    struggles=(
        "Slow to converge — many more iterations than gradient methods",
        "Degenerates badly in > 5–10 parameter dimensions",
        "No global guarantee, no rigorous convergence proof",
        "Cannot exploit residual structure (vector loss → scalar collapse)",
    ),
    properties={
        "Local / Global": "Local",
        "Needs gradient": "No",
        "Needs residuals": "No",
        "Typical iterations": "200 – 1000",
        "Compute cost vs LM": "≈ 3–5×",
    },
    references=(
        ("Dowad — Visualizing Nelder-Mead Optimization (interactive)",
         "https://alexdowad.github.io/visualizing-nelder-mead/"),
        ("Scholarpedia — Nelder-Mead algorithm",
         "http://www.scholarpedia.org/article/Nelder-Mead_algorithm"),
        ("Brandewinder — Breaking down the Nelder-Mead algorithm",
         "https://brandewinder.com/2022/03/31/breaking-down-Nelder-Mead/"),
        ("Nelder & Mead (1965) — Computer Journal 7",
         "https://doi.org/10.1093/comjnl/7.4.308"),
    ),
)


_LBFGSB = SolverDeepDive(
    key="L-BFGS-B",
    long_name="Limited-memory BFGS with box bounds (quasi-Newton)",
    intuition_md=(
        "**Newton's method that forgot how to compute Hessians.** It "
        "fakes the curvature matrix from the last handful of gradients "
        "— cheap, scalable, and it respects the walls of a box (lower "
        "and upper bounds on each parameter)."
    ),
    mechanics_md=(
        "BFGS approximates the **inverse Hessian** $H_k$ recursively "
        "from successive gradient differences without ever forming the "
        "true second-derivative matrix. The **limited-memory** variant "
        "stores only the last $m \\approx 10{-}20$ pairs "
        "$(\\mathbf{s}_i, \\mathbf{y}_i) = (\\theta_{i+1} - \\theta_i,\\,"
        "\\nabla L_{i+1} - \\nabla L_i)$ — memory cost stays $O(mp)$ "
        "instead of $O(p^2)$, which matters when $p$ is large. At each "
        "iteration the solver picks a search direction "
        "$\\mathbf{d}_k = -H_k\\,\\nabla L_k$, does a **line search** "
        "for a step size $\\alpha_k$ satisfying the Wolfe conditions, "
        "and takes the step. The **B** suffix adds **box-bound** "
        "handling so a parameter can never escape its physical range."
    ),
    deep_dive_md=(
        "L-BFGS-B (Byrd, Lu, Nocedal & Zhu 1995) extends limited-memory "
        "BFGS to bound-constrained problems via the *gradient projection* "
        "method: at each step, variables hitting a bound are temporarily "
        "fixed (the **active set**), and L-BFGS runs on the remaining "
        "free coordinates. Convergence is **superlinear under standard "
        "Wolfe line search** when the loss is twice continuously "
        "differentiable. The trade-off vs LM-JAX is structural: L-BFGS-B "
        "minimises a *scalar* objective and consumes a *single gradient*, "
        "so it shines on **MLE problems** (e.g. our GARCH negative "
        "log-likelihood) where there is no natural residual vector. "
        "Used as the SciPy `minimize(method='L-BFGS-B')` backend; gradients "
        "come from JAX autodiff or finite differences if not provided."
    ),
    update_rule_latex=(
        r"\theta_{k+1} = \theta_k - \alpha_k\,H_k\,\nabla L(\theta_k), "
        r"\quad H_k \approx \nabla^2 L(\theta_k)^{-1}\ \text{from the last } m\ \text{gradients}"
    ),
    shines=(
        "Workhorse default for scalar / MLE objectives (GARCH family)",
        "Scales gracefully to large parameter dimensions (low memory)",
        "Respects box bounds natively — parameters stay in physical range",
        "Fast in practice when gradients are cheap (JAX autodiff)",
    ),
    struggles=(
        "Still a *local* solver — sensitive to initialisation",
        "Cannot exploit least-squares structure (LM wins on surfaces)",
        "Needs gradient — finite-difference fallback is slow and noisy",
        "Can stall on flat or pathological landscapes (curvature ill-defined)",
    ),
    properties={
        "Local / Global": "Local",
        "Needs gradient": "Yes (autodiff or finite-diff)",
        "Needs residuals": "No (scalar loss)",
        "Typical iterations": "50 – 200",
        "Compute cost vs LM": "≈ 1.5–3× on surfaces; baseline for MLE",
    },
    references=(
        ("aria42 — Numerical Optimization: Understanding L-BFGS",
         "https://aria42.com/blog/2014/12/understanding-lbfgs"),
        ("Towards Data Science — BFGS in a Nutshell",
         "https://towardsdatascience.com/bfgs-in-a-nutshell-an-introduction-to-quasi-newton-methods-21b0e13ee504/"),
        ("SciPy `minimize(method='L-BFGS-B')` docs",
         "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html"),
        ("Byrd, Lu, Nocedal & Zhu (1995) — SIAM J. Sci. Comput. 16",
         "https://doi.org/10.1137/0916069"),
    ),
)


SOLVER_DEEP_DIVES: tuple[SolverDeepDive, ...] = (_LM_JAX, _DE, _NM, _LBFGSB)


# ── Head-to-head comparison matrix ───────────────────────────────────────

# Rows: each tuple is (criterion, LM-JAX, DE, NM, L-BFGS-B).
COMPARISON_MATRIX_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    ("Scope", "Local", "Global (stochastic)", "Local", "Local"),
    ("Loss shape required", "Least-squares (residuals)", "Any scalar", "Any scalar", "Any scalar"),
    ("Derivative information", "Jacobian (exact, JAX)", "None", "None", "Gradient (exact or FD)"),
    ("Wall-clock speed", "🟢 Fastest (~1×)", "🔴 Slow (~10×)", "🟡 Medium (~3–5×)", "🟢 Fast (~1.5–3×)"),
    ("Robustness to local minima", "🔴 Low — needs warm start", "🟢 High — explores the box",
     "🟡 Medium", "🔴 Low — needs warm start"),
    ("Parallelism", "Serial", "🟢 Trivially parallel (population)", "Serial", "Serial"),
    ("Supports box bounds", "✅ (reflective)", "✅ (native)", "Workaround only", "✅ (native)"),
    ("Best for", "IV surfaces", "Multimodal / unknown landscape", "Sanity check, noisy loss", "GARCH MLE, scalar losses"),
)


DECISION_TREE_MD: str = (
    "- **Are you fitting a GARCH-family model** (scalar negative "
    "log-likelihood)?\n"
    "    - ✅ **Yes** → **L-BFGS-B**, the workhorse for MLE. Use NM as "
    "a sanity check when the likelihood surface looks numerically noisy.\n"
    "    - ❌ No → you have an IV-surface problem with residuals.\n"
    "        - **Do you trust your initial guess** "
    "(e.g. previous-day calibration as warm start)?\n"
    "            - ✅ Yes → **LM-JAX**. Production default — fastest "
    "and most accurate when the basin is right.\n"
    "            - ❌ No → run **DE first** to localise the basin, then "
    "**hand off to LM-JAX** to polish. This *DE → LM* warm-start "
    "pattern is how the Bates scenario is calibrated end-to-end.\n"
    "- **Is your loss non-smooth, noisy, or has discontinuous payoffs?** "
    "Add **NM** to the comparison run — it's the only solver here that "
    "makes zero smoothness assumptions about $L$."
)

DECISION_TREE_CAPTION_MD: str = (
    "This is a *heuristic*, not a rule. The whole point of running "
    "several solvers in parallel (the **⚖️ Compare & Restarts** tab) "
    "is that you can confirm any one solver's answer with an "
    "independent method."
)


# ── Further reading (consolidated bibliography) ──────────────────────────

FURTHER_READING_MD: str = (
    "**Textbook foundations**\n"
    "- Nocedal & Wright, *Numerical Optimization* (Springer, 2nd ed. "
    "2006). The reference. Chapter 3 covers line search, chapter 6 "
    "quasi-Newton (BFGS / L-BFGS), chapter 10 nonlinear least-squares "
    "(Gauss-Newton / Levenberg-Marquardt).\n"
    "- Boyd & Vandenberghe, *Convex Optimization* (Cambridge, 2004). "
    "Free PDF. Chapters 9–11 cover unconstrained minimisation, "
    "Newton's method, and interior-point methods.\n\n"
    "**Original papers**\n"
    "- Levenberg, *A method for the solution of certain non-linear "
    "problems in least squares* (Quart. Appl. Math., 1944).\n"
    "- Marquardt, *An algorithm for least-squares estimation of "
    "nonlinear parameters* (SIAM J. Appl. Math., 1963).\n"
    "- Nelder & Mead, *A simplex method for function minimization* "
    "(Computer Journal, 1965).\n"
    "- Storn & Price, *Differential Evolution — a simple and efficient "
    "heuristic for global optimization over continuous spaces* (J. "
    "Global Optim., 1997).\n"
    "- Byrd, Lu, Nocedal & Zhu, *A limited memory algorithm for bound "
    "constrained optimization* (SIAM J. Sci. Comput., 1995).\n\n"
    "**Online tutorials (modern, visual)**\n"
    "- [Gavin (Duke) — Levenberg-Marquardt for nonlinear least squares](https://people.duke.edu/~hpgavin/lm.pdf)\n"
    "- [NEOS Guide — Levenberg-Marquardt Method](https://neos-guide.org/guide/algorithms/lmm/)\n"
    "- [Cornell Optimization Wiki — Differential Evolution](https://optimization.cbe.cornell.edu/index.php?title=Differential_evolution)\n"
    "- [Rodriguez-Mier — DE tutorial in Python](https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/)\n"
    "- [Dowad — Visualizing Nelder-Mead (interactive)](https://alexdowad.github.io/visualizing-nelder-mead/)\n"
    "- [Scholarpedia — Nelder-Mead algorithm](http://www.scholarpedia.org/article/Nelder-Mead_algorithm)\n"
    "- [aria42 — Understanding L-BFGS](https://aria42.com/blog/2014/12/understanding-lbfgs)\n"
    "- [Towards Data Science — BFGS in a Nutshell](https://towardsdatascience.com/bfgs-in-a-nutshell-an-introduction-to-quasi-newton-methods-21b0e13ee504/)\n\n"
    "**SciPy documentation** — the actual implementations behind LM-JAX, "
    "DE, NM and L-BFGS-B all sit one wrapper away from "
    "[`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html). "
    "Worth reading once."
)
