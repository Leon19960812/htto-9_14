# Plan: Replace Trust-Region Globalization with Merit + Line Search (Correa 2004–style)

This document proposes a controlled experiment to replace (or supplement) the current trust‑region globalization with a merit function + Armijo line search, inspired by Correa & Ramírez (2004) for NLSDP/S‑SDP methods. The goal is to improve robustness against negative ρ events and load‑coupling oscillations while keeping implementation risk and code churn manageable.

---

## 1. Motivation & Goals
- Reduce step rejections caused by design‑dependent loads that spoil the ρ ratio.
- Unify acceptance via a single merit decrease test; simplify trust‑region state.
- Provide two operating modes:
  - practical: robust in practice, allows load mixing/freeze heuristics (no strict proof).
  - theory: frozen model during backtracking, aligning with global convergence assumptions.

Success criteria
- Merit φσ monotonically decreases over accepted iterations in both modes.
- Similar or improved convergence rate vs. current TR on our 2–3 reference cases.
- No regressions in geometry feasibility and stiffness SPD checks.

---

## 2. High‑Level Design

### 2.1 Merit Function
- Base form: φσ(θ, A) = C(θ, A) + σ · V(θ, A)
  - C: actual compliance with current shell–truss coupling.
  - V: sum of soft penalties for outer‑loop constraints we may want to measure (default minimal: volume excess max(0, L·A − Vmax); optionally angle spacing, symmetry soft penalties if we decide to “observe” them during line search).
- Optional stability penalty (if needed): κ · max(0, −λmin(Kff(θ, A))) to softly discourage near‑singular designs. Initially keep off; we still retain SPD guard pre‑check.

Parameters
- σ > 0 (exact penalty weight). Start with σ so that σ·V is same order as typical compliance changes; refine per case.
- c ∈ [1e−4, 1e−2] (Armijo slope constant). Default c = 1e−3.
- β ∈ (0,1) backtracking factor. Default β = 0.5.

### 2.2 Descent Direction & Subproblem
- Keep current SDP subproblem (Schur complement LMI) to produce a step (Δθ, ΔA).
- Add a small positive‑definite quadratic term 0.5·dᵀ M d in the subproblem objective (diagonal M) to encourage descent consistency (theoretical mode).
- Maintain existing geometric hard constraints, bounds, symmetry equalities, and per‑variable move caps.
- Remove the TR ball constraint in theory/practical modes; guard with SPD and line search.

### 2.3 Line Search Flow
1) SPD guard (already implemented): backtrack α until Kff(θk+αd, Ak+αd) is Cholesky‑factorable with reasonable condition number.
2) Merit backtracking (Armijo): find largest α ∈ {1, β, β², …} s.t.
   φσ(xk + αd) ≤ φσ(xk) + c · α · (predicted_decrease)  
   where predicted_decrease uses the subproblem model’s decrease (practical surrogate for directional derivative). If unavailable, use simple decrease test φσ(xk+αd) ≤ φσ(xk) − c·α·|ΔĈ|.
3) Accept step at the found α; skip trust‑region radius updates entirely.

Modes
- practical: allow frozen/mixed loads and limited cache reuse inside Armijo to reduce expensive recomputations.
- theory: freeze loads and active constraint set during the entire line search; do not switch symmetry/active‑set or buckling bounds mid‑backtracking.

---

## 3. Code Touch Points (Minimal)

Files of interest
- `Sequential_Convex_Programming/scp_optimizer.py`
  - Replace current “quality backtracking by ρ” with merit backtracking.
  - Bypass trust‑region radius updates (`update_radius`, history logs) when merit mode is on.
  - Add φσ evaluation helper and hooks for practical/theory switches.
- `Sequential_Convex_Programming/algorithm_modules.py`
  - SubproblemSolver: add tiny diagonal PD term to objective (configurable, default on in theory mode).
  - Remove TR ball constraint `||θ−θk||₂ ≤ r_tr` when merit mode is active; keep per‑variable move caps.
- `docs/overview_algorithm.md`
  - Document the alternative globalization path and parameters.

Guard rails
- Keep existing SPD guard and per‑variable move caps.
- Keep geometry feasibility and equality constraints hard inside the subproblem.

---

## 4. Rollout Plan & Switches

Configuration (new options)
- `--globalization=trust|merit-practical|merit-theory` (default: `trust` to avoid surprise).
- `--merit-sigma`, `--merit-armijo-c`, `--merit-beta`, `--merit-max-bt`.
- `--merit-pd-weight` for the quadratic term (theory mode default small > 0; practical can be 0).

Phases
1) Minimal viable (0.1): merit practical mode, φσ = C + σ·max(0, L·A − Vmax); keep SPD guard; disable TR radius updates; remove TR ball.
2) Theory mode (0.2): add PD term, freeze loads and active sets during Armijo, add clean logs for φσ and Armijo residuals.
3) Optional (0.3): λmax/λmin‑based soft penalties if needed for extra stability; only with a separate flag.

Backout
- Single flag switch back to `--globalization=trust` restores current behavior.

---

## 5. Validation & Metrics

Test set
- 2–3 existing reference problems (small/medium), including a case that previously shows negative ρ / oscillatory loads.

What to log
- φσ values per iteration; Armijo trials with α and decision.
- Compliance C, volume excess, and (optional) SPD condition number chosen.
- Number of FEA/loads recomputations per iteration (to estimate cost).

Acceptance
- φσ monotone decrease on accepted steps.
- Comparable or fewer failed steps and stable convergence within similar iterations/time.

---

## 6. Risks & Mitigations

Design‑dependent loads
- Risk: extra recomputations during Armijo.  
  Mitigation: cached evaluations; practical mode allows blending/freeze to reduce recompute count.

Parameter sensitivity (σ, c)
- Risk: overly conservative or insufficient decrease.  
  Mitigation: start with defaults (σ tuned to C scale; c=1e−3; β=0.5) and sweep on two cases.

Model switches during backtracking
- Risk: theory assumptions violated.  
  Mitigation: theory mode freezes loads and active sets within LS loop.

---

## 7. Effort & Timeline

Minimal viable (practical, with switches): 4–8 hours
- Implement φσ + Armijo backtracking and wiring.
- Remove TR ball and bypass radius updates when enabled.
- Basic logs + parameter hooks; quick sanity runs on 1–2 cases.

Theory mode + polishing: 1–2 days
- PD term in subproblem, load/active‑set freeze inside LS, extended logs.
- Parameter tuning on 2–3 cases; short write‑up and figure replicates.

---

## 8. Documentation Updates
- Update `docs/overview_algorithm.md` to include the merit + line search path and parameters.
- Add a short HOWTO with CLI flags and expected log outputs.

---

## 9. Decision Checklist
- Do we adopt `merit-practical` as an experimental option first, while keeping `trust` default?
- Are we targeting theoretical guarantees now (freeze mode), or postpone to a later iteration?
- Which penalties do we want inside V(θ, A) beyond volume excess (if any)?
