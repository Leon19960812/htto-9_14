# Shell Pressure Integration Update

This note captures the changes needed to move the shell pressure loading to the work-equivalent, boundary-integration scheme described in `docs/reference/hydrostatic_calculation`.

## 1. Boundary pressure assembly
- Replace the node-area lumping in `Shell2DFEA._compute_pressure_loads` with a loop over boundary edges.
- For each boundary edge (two nodes on the outer hull):
  - Compute the midpoint coordinates and outward unit normal.
  - Evaluate hydrostatic pressure `P = rho_water * g * (depth - y_mid)` (clamped to zero above the free surface as needed).
  - Use a single Gauss point (`w = 1`, `J = L_edge / 2`) to obtain the elemental force `f_edge = P * n * w * J`.
  - Distribute `f_edge` to the two edge nodes by multiplying with the linear shape functions (`N1 = N2 = 0.5`).
- Accumulate the two Cartesian components into `pressure_loads[2*node]` / `[2*node+1]` for every boundary node.
- Optional: precompute and cache the boundary-edge list when the mesh is generated to avoid rebuilding it each call.

## 2. Data structure adjustments
- Extend `ShellMeshData` (or `Shell2DFEA`) to keep `boundary_edges` — e.g. an array of `[node_i, node_j]` pairs determined during `_generate_shell_mesh`.
- Ensure `_compute_boundary_angles` and the support-weight mapping continue to use the existing `boundary_nodes` information (no behavioral change needed).

## 3. Consistency checks & diagnostics
- After the new load assembly, keep the existing total-load printout but recompute via the integrated forces.
- Retain the imbalance warning in `solve_with_support_positions`; with the integration-based loads the imbalance should remain near machine precision.
- Consider adding a debug hook to verify that summing all `pressure_loads` reproduces the analytical hydrostatic resultant (for regression tests).

## 4. Interaction with the truss optimizer
- No change to `LoadCalculatorWithShell`: it still calls `solve_with_support_positions` and then projects the reactions onto the truss nodes.
- Because the shell load is now smoother with geometry changes, trust-region steps in the SCP loop should experience fewer rejections.

## 5. Follow-up tasks (optional)
- If higher accuracy is required, upgrade the edge quadrature to two Gauss points or incorporate curved boundaries.
- Add unit / regression tests that compare the integrated nodal loads against known analytical solutions for simple geometries.

## 6. Follow-up (2025-09-??)
- Updated `LoadCalculatorWithShell` to use the full shell support reactions as truss loads (no radial projection). The shell FEA already resolves the vector reaction; projecting and truncating to radial-only caused large load jumps when the reaction direction varied.
- Debug log (`--shell-debug-log`) now records the raw reaction components, enabling verification that loads are transmitted directly.
- Adjusted load visualization arrows to scale both shaft and head with force magnitude so directions reflect the full 2D reaction components (no apparent asymmetry from wide arrowheads).
- Implemented load-freezing inside the trust-region evaluation: StepQuality now reuses the last accepted shell load vector while assessing candidate steps. Shell FEA is only invoked after a step is accepted, and the resulting load becomes the new frozen state for the next iteration.
