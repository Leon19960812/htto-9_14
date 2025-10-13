# Numerical Results — Figure Plan

This note lists proposed figures for the Numerical Results section, with a brief purpose, data source in code, and suggested output names. The goal is to cover SCP vs SDP comparisons, mesh sensitivity, shell–truss loading behavior, and algorithm dynamics, without duplicating earlier method content.

## A. Layout Comparisons (SDP vs SCP)
- Title: Final layouts under hydrostatic loading (SDP vs SCP)
- Purpose: Visual comparison of optimized topologies/geometry; annotate compliance J and effective members.
- Variants: `n_sectors ∈ {8, 12, 16}` (same radius/depth/volume fraction);
- Code hooks: `Sequential_Convex_Programming/visualization.py::_plot_structure`, `cli.py --save-figs`.
- Outputs: `results/sdp_vs_scp/sdp_{N}.pdf`, `results/sdp_vs_scp/scp_{N}.pdf`.

## B. Area Distribution
- Title: Area histograms (SDP vs SCP)
- Purpose: Show material allocation differences and near-threshold sparsity.
- Code hooks: `visualization.py::plot_single_figure(..., figure_type="area_histogram")`, `cli.py` export.
- Outputs: `results/sdp/area_histogram.png`, `results/scp/area_histogram.png`.

## C. Compliance and Trust-Region Dynamics
- Title: Compliance evolution over accepted iterations
- Purpose: Convergence behavior; stall detection wrt 0.1% criterion.
- Code hooks: `visualization.py::plot_compliance_evolution` (called from `cli.py`).
- Output: `results/scp/compliance_evolution.png`.

- Title: Trust-region radius evolution
- Purpose: Step quality control behavior (expand/accept/shrink epochs).
- Code hooks: `visualization.py::plot_trust_region_evolution_only`.
- Output: `results/scp/trust_region_evolution.png`.

## D. Step Quality and Prediction Diagnostics
- Title: Step quality ρ and predicted vs actual compliance
- Purpose: Quality of linearization and load continuation; highlight rejected steps.
- Code hooks: `visualization.py` (ρ annotations around lines ~1200); optional `comparison/plot_step_details.py` for extended per-step panels.
- Output: `results/scp/step_quality_overview.png`.

## E. Load and Shell Response Comparisons (SDP vs SCP)
- Title: Truss nodal load distribution (vectors) — before vs after optimization
- Purpose: How geometry affects mapped loads; angular redistribution.
- Code hooks: `visualization.py::_plot_loads` (SCP); `sdp_truss_optimizer_fixed.py::_plot_loads` (SDP).
- Outputs: `results/sdp/load_distribution.png`, `results/scp/load_distribution.png`.

- Title: Shell displacement fields (snapshot)
- Purpose: Shell response under mapped supports; qualitative deflection change.
- Code hooks: `cli.py` calls `shell_fea.visualize_last_solution(...)` if shell FEA is active; SCP also supports per-iteration snapshots when `--save-shell-iter` is used.
- Outputs: `results/sdp/shell_displacement.png`, `results/scp/shell_displacement.png`; optional sequence `results/scp/shell_displacement_iter/*.png`.

- Title: Shell support reaction statistics (table/plot)
- Purpose: Quantify “better” shell support state under SCP vs SDP.
- Metrics (per case):
  - `max‖λ‖`, `mean‖λ‖`, `std‖λ‖` over supports;
  - angular variance of reaction magnitude (uniformity): `Var{|λ(θ)|}`;
  - global balance check: `‖∑λ + ∑p·n‖` (should be small);
  - integrated pressure at base (reference): `p0 = ρ_w g H` (for context).
- Code hooks: `load_calculator_with_shell.py::shell_fea.solve_with_support_positions(...)` returns reactions; available during load computation.
- Outputs: small CSV/JSON + bar/line plots, e.g., `results/compare_shell_reactions/summary.csv`, `.../lambda_stats.png`.

## F. Mesh Sensitivity (Sectors Study)
- Title: Compliance vs number of sectors (SDP and SCP)
- Purpose: Mesh dependency view; show trend while avoiding over-claiming grid independence.
- Procedure: Sweep `--n-sectors ∈ {8, 12, 16, 24}` with fixed radius/depth/volume fraction; record final compliance, effective members, volume utilization.
- Code hooks: `Sequential_Convex_Programming/cli.py` runner; aggregate logs from `opt._export_iteration_state_logs` and console summaries.
- Outputs: `results/study_mesh/compliance_vs_sectors.png`, `.../active_members_vs_sectors.png`.
- Note: Optionally normalize compliance by a reference (e.g., `C / (ρ_w g H R)`) for scale context; report as supplemental.

## G. Before/After Topology Cleanup Overlay (SCP)
- Title: Optimized vs cleaned design overlay
- Purpose: Visualize effect of node merge/aggregation on final layout readability.
- Code hooks: `visualization.py::plot_single_figure(..., figure_type="cleaned")` and baseline optimized plot.
- Outputs: `results/scp/optimized.png`, `results/scp/after_cleanup.png`, or a single overlay figure.

---

## Suggested Generation Flow (scripts available)

- Single run with exports (SCP):
  - `python -m Sequential_Convex_Programming.cli --n-sectors 12 --depth 50 --radius 5 --volume-fraction 0.2 --max-iterations 30 --save-figs results/scp --save-shell-iter`

- Single fixed-geometry SDP (diagnostic):
  - `python -m Sequential_Convex_Programming.cli --n-sectors 12 --depth 50 --radius 5 --volume-fraction 0.2 --single-subproblem --sdp-fixed-geometry --save-figs results/sdp`

- Mesh sweep (example):
  - `for N in 8 12 16 24: run cli.py with --n-sectors N, collect compliance and figures under results/study_mesh/N/` (use a small wrapper script).

This plan is compatible with existing code paths; only the optional shell reaction metric export (E, last item) may need a tiny helper to dump `λ` statistics when computing loads.

