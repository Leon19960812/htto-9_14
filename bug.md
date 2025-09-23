Iteration 37/80
============================================================
Solving joint linearized subproblem...
[SCP-SDP] n_free=82, m=250, lambda_min(K_aff@k)=2.731e-06, lambda_max=1.122e-01
[SCP-SDP] J_f=15, J_o=26, sum||Sym(Kθ_j)||_F(J_o)=1.253e+01, gamma_est(J_o)=4.374e-01, sum||Sym(Kθ_j)||_F(all)=1.717e+01, gamma_est(all)=5.992e-01, TR=3.491e-02
[SCP-GEO] n_vars=41, same_pairs=2, cross_pairs=38, min_same_slack=0.2331638676524054, min_cross_slack=-1.1102230246251565e-15, buffer_slack=(1.438e-01,1.438e-01)
  Gradient cache refreshed
  Line search (SPD guard): alpha_final=1.000 | tried: 1.000 | cond(K_ff)~=4.084e+04
  Step norms: ||theta change||2=3.491e-02, ||A change||2=4.859e-03
  Changes:
    theta change: 3.490645e-02 (max change: 1.058681e-02)
    A change: 4.858968e-03 (max change: 1.764789e-03)
Quality backtracking: α=1.000, ρ=1.068
Trust region: 0.0349 -> 0.0349 (EXPAND)
   Accepted step (success #37)
   Compliance: 3.792287e+03 → 3.748455e+03
   Improvement: 1.16%

============================================================
Iteration 38/80
============================================================
Solving joint linearized subproblem...
[SCP-SDP] n_free=82, m=250, lambda_min(K_aff@k)=3.266e-06, lambda_max=1.334e-01
[SCP-SDP] J_f=15, J_o=26, sum||Sym(Kθ_j)||_F(J_o)=1.732e+01, gamma_est(J_o)=6.045e-01, sum||Sym(Kθ_j)||_F(all)=2.364e+01, gamma_est(all)=8.253e-01, TR=3.491e-02
[SCP-GEO] n_vars=41, same_pairs=2, cross_pairs=38, min_same_slack=0.2363729464344401, min_cross_slack=-6.550315845288424e-14, buffer_slack=(1.477e-01,1.477e-01)
  Gradient cache refreshed
  Line search (SPD guard): alpha_final=1.000 | tried: 1.000 | cond(K_ff)~=5.243e+04
  Step norms: ||theta change||2=3.491e-02, ||A change||2=8.978e-03
  Changes:
    theta change: 3.490643e-02 (max change: 1.100775e-02)
    A change: 8.977766e-03 (max change: 3.867644e-03)
Quality backtracking: α=1.000, ρ=1.098
Trust region: 0.0349 -> 0.0349 (EXPAND)
   Accepted step (success #38)
   Compliance: 3.748455e+03 → 3.702038e+03
   Improvement: 1.24%

============================================================
Iteration 39/80
============================================================
Solving joint linearized subproblem...
[SCP-SDP] n_free=82, m=250, lambda_min(K_aff@k)=3.276e-06, lambda_max=1.718e-01
[SCP-SDP] J_f=15, J_o=26, sum||Sym(Kθ_j)||_F(J_o)=2.802e+01, gamma_est(J_o)=9.781e-01, sum||Sym(Kθ_j)||_F(all)=3.772e+01, gamma_est(all)=1.317e+00, TR=3.491e-02
[SCP-GEO] n_vars=41, same_pairs=2, cross_pairs=38, min_same_slack=0.23914433265425342, min_cross_slack=-2.6867397195928788e-14, buffer_slack=(1.512e-01,1.512e-01)
  Gradient cache refreshed
  Line search (SPD guard): alpha_final=1.000 | tried: 1.000 | cond(K_ff)~=8.522e+04
  Step norms: ||theta change||2=3.491e-02, ||A change||2=3.179e-03
  Changes:
    theta change: 3.490649e-02 (max change: 1.106599e-02)
    A change: 3.179119e-03 (max change: 9.975481e-04)
Quality backtracking: α=1.000, ρ=1.152
Trust region: 0.0349 -> 0.0349 (EXPAND)
   Accepted step (success #39)
   Compliance: 3.702038e+03 → 3.652321e+03
   Improvement: 1.34%
[NodeMerge] 4 candidate group(s) @ 0.100 m: 42<-(43), 31<-(32), 27<-(28), ...
LoadCalculator initialized:
  Mode: SimpleHydrostatic
 reinitialized load calculator with shell FEA.
Cleared linearization cache; next iteration will re-linearize.
Symmetry constraints enabled: 17 mirror pairs; 3 nodes fixed at pi/2.
[Symmetry][repair] added mirror element (18,29) for (24,39)
[Symmetry][repair] added mirror element (27,26) for (15,16)
[Symmetry][repair] added mirror element (16,31) for (26,37)
[Symmetry][repair] added mirror element (12,39) for (2,29)
[Symmetry][repair] added mirror element (4,16) for (10,26)
[Symmetry][repair] added mirror element (27,39) for (15,29)
[Symmetry][repair] added mirror element (18,16) for (24,26)
[Symmetry][repair] added mirror element (26,40) for (16,28)
[Symmetry][repair] added mirror element (30,29) for (38,39)
[Symmetry][repair] added mirror element (4,29) for (10,39)
[Symmetry][repair] added mirror element (1,29) for (13,39)
Area symmetry enabled: 109 mirror member pairs.
   Re-evaluated compliance after merge: 4.766248e+03
   Recomputed cached stiffness matrices; 222 elements total

============================================================
Iteration 40/80
============================================================
Solving joint linearized subproblem...
[SCP-SDP] n_free=74, m=222, lambda_min(K_aff@k)=2.978e-06, lambda_max=1.083e-01
[SCP-SDP] J_f=15, J_o=22, sum||Sym(Kθ_j)||_F(J_o)=1.610e+00, gamma_est(J_o)=5.619e-02, sum||Sym(Kθ_j)||_F(all)=1.921e+01, gamma_est(all)=2.784e-01, TR=3.491e-02
[SCP-GEO] n_vars=37, same_pairs=2, cross_pairs=34, min_same_slack=0.19129931354255789, min_cross_slack=2.4424906541753444e-14, buffer_slack=(1.544e-01,1.544e-01)
  Gradient cache refreshed
  Line search (SPD guard): alpha_final=1.000 | tried: 1.000 | cond(K_ff)~=7.907e+04
  Step norms: ||theta change||2=3.491e-02, ||A change||2=1.918e-02
  Changes:
    theta change: 3.490655e-02 (max change: 1.101605e-02)
    A change: 1.917822e-02 (max change: 9.998870e-03)
Quality backtracking: α=1.000, ρ=1.004
Trust region: 0.0349 -> 0.0349 (EXPAND)
   Accepted step (success #40)
   Compliance: 4.766248e+03 → 3.682641e+03
   Improvement: 22.74%

============================================================
Iteration 41/80
============================================================
Solving joint linearized subproblem...
[SCP-SDP] n_free=74, m=222, lambda_min(K_aff@k)=2.737e-06, lambda_max=2.164e-01
[SCP-SDP] J_f=15, J_o=22, sum||Sym(Kθ_j)||_F(J_o)=1.768e+00, gamma_est(J_o)=6.170e-02, sum||Sym(Kθ_j)||_F(all)=4.170e+01, gamma_est(all)=5.334e-01, TR=3.491e-02
[SCP-GEO] n_vars=37, same_pairs=2, cross_pairs=34, min_same_slack=0.1957767639491745, min_cross_slack=-2.0765611452588928e-12, buffer_slack=(1.604e-01,1.604e-01)
  Gradient cache refreshed
  Line search (SPD guard): alpha_final=1.000 | tried: 1.000 | cond(K_ff)~=1.028e+05
  Step norms: ||theta change||2=3.491e-02, ||A change||2=2.901e-03
  Changes:
    theta change: 3.490658e-02 (max change: 1.101636e-02)
    A change: 2.901086e-03 (max change: 1.286448e-03)
Quality backtracking: α=1.000, ρ=1.029
Trust region: 0.0349 -> 0.0349 (EXPAND)
   Accepted step (success #41)
   Compliance: 3.682641e+03 → 3.645743e+03
   Improvement: 1.00%
[NodeMerge] 2 candidate group(s) @ 0.100 m: 1<-(2), 12<-(13)
LoadCalculator initialized:
  Mode: SimpleHydrostatic
 reinitialized load calculator with shell FEA.
Cleared linearization cache; next iteration will re-linearize.
Symmetry constraints enabled: 16 mirror pairs; 3 nodes fixed at pi/2.
[Symmetry][repair] added mirror element (11,25) for (1,13)
[Symmetry][repair] added mirror element (1,29) for (11,35)
[Symmetry][repair] added mirror element (3,1) for (9,11)
[Symmetry][repair] added mirror element (11,38) for (1,26)
[Symmetry][repair] added mirror element (1,16) for (11,22)
Area symmetry enabled: 102 mirror member pairs.
   Re-evaluated compliance after merge: 4.311666e+03
   Recomputed cached stiffness matrices; 208 elements total

============================================================
Iteration 42/80
============================================================
Solving joint linearized subproblem...
[SCP-SDP] n_free=70, m=208, lambda_min(K_aff@k)=2.560e-06, lambda_max=4.942e-02
[SCP-SDP] J_f=13, J_o=22, sum||Sym(Kθ_j)||_F(J_o)=1.698e+00, gamma_est(J_o)=5.925e-02, sum||Sym(Kθ_j)||_F(all)=3.241e+00, gamma_est(all)=1.131e-01, TR=3.491e-02
[SCP-GEO] n_vars=35, same_pairs=1, cross_pairs=33, min_same_slack=0.24460775704550078, min_cross_slack=0.0, buffer_slack=(1.652e-01,1.652e-01)
⚠️ Area symmetry solve failed; temporarily disabling area equality: Solver 'MOSEK' failed. Try another solver, or solve with verbose=True foor more information.
[SCP-SDP] n_free=70, m=208, lambda_min(K_aff@k)=2.560e-06, lambda_max=4.942e-02
[SCP-SDP] J_f=13, J_o=22, sum||Sym(Kθ_j)||_F(J_o)=1.698e+00, gamma_est(J_o)=5.925e-02, sum||Sym(Kθ_j)||_F(all)=3.241e+00, gamma_est(all)=1.131e-01, TR=3.491e-02
[SCP-GEO] n_vars=35, same_pairs=1, cross_pairs=33, min_same_slack=0.24460775704550078, min_cross_slack=0.0, buffer_slack=(1.652e-01,1.652e-01)
  Gradient cache refreshed
  Line search (SPD guard): alpha_final=1.000 | tried: 1.000 | cond(K_ff)~=1.913e+04
  Step norms: ||theta change||2=3.491e-02, ||A change||2=2.594e-02
  Changes:
    theta change: 3.490657e-02 (max change: 1.057571e-02)
    A change: 2.593647e-02 (max change: 9.998976e-03)
Quality backtracking: α=1.000, ρ=1.002
Trust region: 0.0349 -> 0.0349 (EXPAND)
   Accepted step (success #42)
   Compliance: 4.311666e+03 → 3.123534e+03
   Improvement: 27.56%

============================================================
Iteration 43/80
============================================================
Solving joint linearized subproblem...
[SCP-SDP] n_free=70, m=208, lambda_min(K_aff@k)=2.613e-06, lambda_max=4.998e-02
[SCP-SDP] J_f=13, J_o=22, sum||Sym(Kθ_j)||_F(J_o)=1.984e+00, gamma_est(J_o)=6.924e-02, sum||Sym(Kθ_j)||_F(all)=4.065e+00, gamma_est(all)=1.419e-01, TR=3.491e-02
[SCP-GEO] n_vars=35, same_pairs=1, cross_pairs=33, min_same_slack=0.2478270748847974, min_cross_slack=1.9095836023552692e-14, buffer_slack=(1.657e-01,1.657e-01)
  Gradient cache refreshed
  Line search (SPD guard): alpha_final=1.000 | tried: 1.000 | cond(K_ff)~=2.014e+04
  Step norms: ||theta change||2=3.491e-02, ||A change||2=2.399e-03
  Changes:
    theta change: 3.490658e-02 (max change: 1.043309e-02)
    A change: 2.399294e-03 (max change: 8.985520e-04)
Quality backtracking: α=1.000, ρ=1.027
Trust region: 0.0349 -> 0.0349 (EXPAND)
   Accepted step (success #43)
   Compliance: 3.123534e+03 → 3.086029e+03
   Improvement: 1.20%

============================================================
Iteration 44/80
============================================================
Solving joint linearized subproblem...
[SCP-SDP] n_free=70, m=208, lambda_min(K_aff@k)=2.538e-06, lambda_max=5.110e-02
[SCP-SDP] J_f=13, J_o=22, sum||Sym(Kθ_j)||_F(J_o)=2.132e+00, gamma_est(J_o)=7.441e-02, sum||Sym(Kθ_j)||_F(all)=4.288e+00, gamma_est(all)=1.497e-01, TR=3.491e-02
[SCP-GEO] n_vars=35, same_pairs=1, cross_pairs=33, min_same_slack=0.2509107302908732, min_cross_slack=-5.3512749786932545e-14, buffer_slack=(1.645e-01,1.645e-01)
  Gradient cache refreshed
  Line search (SPD guard): alpha_final=1.000 | tried: 1.000 | cond(K_ff)~=2.110e+04
  Step norms: ||theta change||2=3.491e-02, ||A change||2=2.252e-03
  Changes:
    theta change: 3.490658e-02 (max change: 1.049877e-02)
    A change: 2.252027e-03 (max change: 6.916841e-04)
Quality backtracking: α=1.000, ρ=1.029
Trust region: 0.0349 -> 0.0349 (EXPAND)
   Accepted step (success #44)
   Compliance: 3.086029e+03 → 3.049180e+03
   Improvement: 1.19%

============================================================
Iteration 45/80
============================================================
Solving joint linearized subproblem...
[SCP-SDP] n_free=70, m=208, lambda_min(K_aff@k)=2.487e-06, lambda_max=5.248e-02
[SCP-SDP] J_f=13, J_o=22, sum||Sym(Kθ_j)||_F(J_o)=2.327e+00, gamma_est(J_o)=8.122e-02, sum||Sym(Kθ_j)||_F(all)=4.576e+00, gamma_est(all)=1.597e-01, TR=3.491e-02
[SCP-GEO] n_vars=35, same_pairs=1, cross_pairs=33, min_same_slack=0.25403371022977284, min_cross_slack=-1.84297022087776e-14, buffer_slack=(1.638e-01,1.638e-01)
❌ Iteration 45 failed: MOSEK solve failed for SDP subproblem: Solver 'MOSEK' failed. Try another solver, or solve with verbose=True for more information.

⚠️ Optimization terminated with an error. Partial results will be used for exports.
   Reason: MOSEK solve failed for SDP subproblem: Solver 'MOSEK' failed. Try another solver, or solve with verbose=True for more information.
Current compliance at failure: 3.049180e+03
Failed to save figures: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
Partial figures correspond to the last accepted iterate.