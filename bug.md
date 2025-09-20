迭代 39/50
============================================================
求解联合线性化子问题...
[SCP-SDP] n_free=54, m=147, lambda_min(K_aff@k)=1.102e-05, lambda_max=1.599e-01
[SCP-SDP] J_f=9, J_o=18, sum||Sym(Kθ_j)||_F(J_o)=1.126e+01, gamma_est(J_o)=4.761e-01, sum||Sym(Kθ_j)||_F(all)=1.220e+01, gamma_est(all)=5.574e-01, TR=8.727e-02
[SCP-GEO] n_vars=27, same_pairs=0, cross_pairs=26, min_same_slack=None, min_cross_slack=-7.239986388185571e-12, buffer_slack=(6.275e-09,2.441e-08)
✅ 梯度缓存成功
  Line search (SPD guard): α_final=1.000 | tried: 1.000 | cond(K_ff)≈2.112e+04
  Step norms: ||Δθ||2=8.727e-02, ||ΔA||2=7.909e-03
  变化情况:
    θ变化: 8.726644e-02 (最大变化: 3.117725e-02)
    A变化: 7.908955e-03 (最大变化: 3.411868e-03)
  Quality backtracking: α=1.000, ρ=1.325
Trust region: 0.0873 -> 0.0873 (EXPAND)
✅ 接受步长 (第39次成功)
   柔度: 1.624558e+03 → 1.588543e+03
   改进: 2.22%

============================================================
迭代 40/50
============================================================
求解联合线性化子问题...
[SCP-SDP] n_free=54, m=147, lambda_min(K_aff@k)=1.071e-05, lambda_max=2.262e-01
[SCP-SDP] J_f=9, J_o=18, sum||Sym(Kθ_j)||_F(J_o)=2.744e+01, gamma_est(J_o)=1.201e+00, sum||Sym(Kθ_j)||_F(all)=2.838e+01, gamma_est(all)=1.283e+00, TR=8.727e-02
[SCP-GEO] n_vars=27, same_pairs=0, cross_pairs=26, min_same_slack=None, min_cross_slack=-4.703126776917088e-12, buffer_slack=(2.411e-09,9.963e-09)
✅ 梯度缓存成功
  Line search (SPD guard): α_final=1.000 | tried: 1.000 | cond(K_ff)≈5.244e+04
  Step norms: ||Δθ||2=8.727e-02, ||ΔA||2=6.725e-03
  变化情况:
    θ变化: 8.726640e-02 (最大变化: 3.073333e-02)
    A变化: 6.725042e-03 (最大变化: 4.473740e-03)
  Quality backtracking: α=1.000, ρ=1.436
Trust region: 0.0873 -> 0.0873 (EXPAND)
✅ 接受步长 (第40次成功)
   柔度: 1.588543e+03 → 1.550830e+03
   改进: 2.37%

============================================================
迭代 41/50
============================================================
求解联合线性化子问题...
[SCP-SDP] n_free=54, m=147, lambda_min(K_aff@k)=1.035e-05, lambda_max=5.426e-01
[SCP-SDP] J_f=9, J_o=18, sum||Sym(Kθ_j)||_F(J_o)=1.609e+02, gamma_est(J_o)=6.947e+00, sum||Sym(Kθ_j)||_F(all)=1.620e+02, gamma_est(all)=7.046e+00, TR=8.727e-02
[SCP-GEO] n_vars=27, same_pairs=0, cross_pairs=26, min_same_slack=None, min_cross_slack=-1.787192616120592e-11, buffer_slack=(6.300e-09,2.721e-08)
✅ 梯度缓存成功
  Line search (SPD guard): α_final=1.000 | tried: 1.000 | cond(K_ff)≈2.102e+08
  Step norms: ||Δθ||2=8.727e-02, ||ΔA||2=1.094e-02
  变化情况:
    θ变化: 8.726640e-02 (最大变化: 3.370719e-02)
    A变化: 1.094124e-02 (最大变化: 9.452543e-03)
  Quality backtracking: α=1.000, ρ=-4467.224 | α=0.500, ρ=1.000
  α adjusted by ρ: 1.000 → 0.500
Trust region: 0.0873 -> 0.0873 (EXPAND)
✅ 接受步长 (第41次成功)
   柔度: 1.550830e+03 → 1.534021e+03
   改进: 1.08%
[NodeMerge] 3 candidate group(s) @ 0.025 m: 17<-(18), 10<-(11), 28<-(29)
LoadCalculator initialized:
  Mode: SimpleHydrostatic
   重新初始化载荷计算器（Shell FEA网格已更新）
   清除线性化缓存，强制重线性化
对称约束已停用：半径 2.999980 存在未匹配的节点 (θ=0.040957)。
   合并后重新评估柔度: 1.491550e+08
   重新计算预计算刚度矩阵，共 121 个单元

============================================================
迭代 42/50
============================================================
求解联合线性化子问题...
[SCP-SDP] n_free=48, m=121, lambda_min(K_aff@k)=2.393e-09, lambda_max=5.795e-02
[SCP-SDP] J_f=9, J_o=15, sum||Sym(Kθ_j)||_F(J_o)=2.387e+00, gamma_est(J_o)=6.059e-02, sum||Sym(Kθ_j)||_F(all)=3.535e+00, gamma_est(all)=1.195e-01, TR=8.727e-02
[SCP-GEO] n_vars=24, same_pairs=2, cross_pairs=21, min_same_slack=0.5426423461554943, min_cross_slack=6.705747068735946e-14, buffer_slack=(5.397e-09,2.403e-08)
❌ 迭代 42 失败: MOSEK solve failed for SDP subproblem: Solver 'MOSEK' failed. Try another solver, or solve with verbose=True for more information.

⚠️ Optimization terminated with an error. Partial results will be used for exports.
   Reason: MOSEK solve failed for SDP subproblem: Solver 'MOSEK' failed. Try another solver, or solve with verbose=True for more information.
Current compliance at failure: 1.491550e+08
Failed to save figures: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
Partial figures correspond to the last accepted iterate.