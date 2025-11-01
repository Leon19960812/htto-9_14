# Load Stabilization Playbook for SCP (Shell → Truss)

> Practical methods from Lorentzon & Revstedt (2022) adapted to your **Sequential Convex Programming (SCP)** workflow with **shell-based loads** mapped to a **truss**. Focus: reduce step rejections (ρ<0), smooth design-dependent load updates, and stabilize shell→truss coupling.

---

## 0) Goals & Where They Plug In

* **When:** each time the SCP loop evaluates `actual_compliance(candidate)` and recomputes shell loads.
* **Where:** between `Shell2DFEA` (pressure → shell reactions/tractions) and `Truss` assembly.
* **What:** smooth **load updates** across trust-region steps; suppress high-frequency/load spikes; avoid large prediction–actual gaps that tank ρ.

---

## 1) Methods to Use (Recommended Order)

### 1.1 Continuation (Gradual Load Update) ✅✅✅

Smoothly blend **old** and **new** loads over a few micro-steps before evaluating the candidate design:

[
\mathbf{F}^{(j)} = (1-\alpha_j),\mathbf{F}*{\text{old}} ;+; \alpha_j,\mathbf{F}*{\text{new}},\quad \alpha_j \nearrow 1
]

* Map from paper: continuation/homotopy on traction & structural parameters (Eq. (12)–(14))。
* Effect: mimics “**load freezing → gradual unfreezing**”, drastically reduces load jumps that cause ρ<0.

**Heuristics**

* 3–5 inner blending steps per trust-region evaluation usually enough.
* Use a convex schedule (e.g., (\alpha_j = 1-(1-\alpha_{\max})\gamma^j), (\gamma\in(0.5,0.8))).

---

### 1.2 FIR Low-Pass Filter on Interface Traction ✅✅

Filter the **shell→truss** traction/reaction history before mapping to truss nodes:

[
\tilde{\mathbf t} = \text{FIR}[f_c]{\mathbf t}
]

* Map from paper: **Finite Impulse Response (FIR)** interface filtering (Section 2.6)。
* Effect: removes high-frequency spikes (e.g., higher structural modes / numerical noise) with minor amplitude loss.

**Heuristics**

* Cutoff (f_c) below the first “troublesome” structural mode (paper examples used ~12 Hz for their case)。
* Keep a short window (e.g., 5–9 taps) to avoid lag.

---

### 1.3 SUR / Aitken Δ² on Load Update Weights ✅

Blend loads with **fixed** or **adaptive** weights:

* **SUR (fixed):** (\mathbf{F}_{k+1} = (1-\omega)\mathbf{F}*k + \omega,\mathbf{F}*{\text{shell}}), (0<\omega<1)（Eq. (8)）
* **Aitken Δ² (adaptive):** auto-update (\omega_k) from residuals（Eq. (11)）

**Heuristics**

* Start SUR (\omega=0.6\sim0.8); upgrade to Aitken once residuals are less noisy (possibly after FIR).

---

### 1.4 (Optional) Hybrid Precondition → Fast Relaxation

* Run **a few** continuation steps (Section 1.1) as **precondition**, then switch to a faster relaxation (Aitken / quasi-Newton).
* Map from paper: “pIQN-ILS” concept (preconditioned coupling) showing big stability margin gains。

*Note:* Full IQN-ILS is overkill unless you implement an inner shell–truss iteration; the **idea** of “precondition then accelerate” still applies.

---

## 2) SCP Integration — Drop-in Pseudocode

```python
# Before SCP loop
F_load = F_shell(theta0)  # initial shell→truss load

for k in range(kmax):
    dtheta, dA, C_pred = solve_linearized_subproblem(theta, A)

    candidate = (theta + dtheta, A + dA)

    # --- Recompute shell loads for candidate geometry ---
    F_shell_new = shell_FEA_reactions(candidate)   # pressure→shell
    t_hist.append(F_shell_new.interface_traction)

    # 1) FIR filter interface traction (optional but recommended)
    t_filt = FIR_filter(t_hist, cutoff=fc, window=T)    # Section 1.2
    F_shell_new = map_traction_to_truss(t_filt)         # work-equivalent integration

    # 2) Continuation: gradually blend OLD→NEW loads (inner micro-steps)
    F_tmp = F_load
    for j in range(J):                                   # e.g., J = 3~5
        alpha_j = schedule(j)                            # convex ↑ to 1
        F_blend = (1-alpha_j)*F_tmp + alpha_j*F_shell_new

        # 3) Optional SUR/Aitken on the blend itself (extra damping)
        # F_blend = (1-w)*F_prev + w*F_blend

        # Evaluate compliance with blended load (cheap truss solve)
        C_j = truss_compliance(candidate, F_blend)

        # Early-abort if C_j stabilizes (Δ small) to save solves
        if stable_enough(C_j): 
            break
        F_tmp = F_blend

    C_new = C_j
    # --- trust-region ratio ---
    rho = (C - C_new) / max(C - C_pred, eps)

    if (rho < 0) or (C - C_pred <= 0):
        shrink_trust_region()
        reject_step()
    else:
        accept_step(candidate)
        F_load = F_blend   # commit smoothed load
        maybe_expand_trust_region()
```

**Notes**

* `FIR_filter` operates on interface traction history (`t_hist`).
* `map_traction_to_truss` should use **work-equivalent line integration** to avoid discrete point-load noise.
* `schedule(j)`: e.g., `alpha_j = 1 - (1 - 0.99)*gamma**j` with `gamma=0.6–0.8`, `J=3–5`.
* Early-abort on stable (C_j) prevents unnecessary inner solves.

---

## 3) Practical Defaults & Safeguards

* **FIR cutoff**: start low (e.g., 10–20 Hz equivalent in your units) → tune upward if loads look over-damped。
* **Continuation steps**: J=3–5; more only if ρ still negative.
* **SUR weight**: ω=0.6–0.8; enable **Aitken** after loads become smooth.
* **CFL-style guard** on load change:
  [
  \frac{|\mathbf{F}*{\text{new}}-\mathbf{F}*{\text{old}}|}{|\mathbf{F}_{\text{old}}|} \le \tau \quad(\text{e.g., } \tau=0.2)
  ]
  If violated, shrink trust-region and/or increase continuation steps.
* **Stop criteria** inside continuation: stop when (|C_j-C_{j-1}|/C_{j-1}<10^{-3}).

---

## 4) Why These Help (Paper Mapping)

* **Continuation**: homotopy/blending of traction and structural terms (Eqs. (12)–(14)) stabilizes near instability regions, easy to add to black-box coupling。
* **FIR**: low-pass filter on interface traction to remove destabilizing high-frequency content with minor accuracy penalty (Section 2.6, Table 1 discussion)。
* **SUR/Aitken**: classical and adaptive relaxation to damp zig-zag residuals and reduce divergence risk (Eqs. (8), (11))。
* **Hybrid precondition**: continuation first, then fast relaxer; empirically extends stability limits (Table 2, Figure 6)。

---

## 5) Minimal Code Changes (Checklist)

* [ ] Add **FIR** utility on interface traction history.
* [ ] Replace direct `F_load ← F_shell_new` with **continuation micro-loop**.
* [ ] Add optional **SUR/Aitken** on `F_blend`.
* [ ] Gate trust-region acceptance with **CFL-style load-change cap**.
* [ ] Commit **smoothed** `F_load` only on **accepted** steps.

## 7) 当前实现与剩余问题

* 代码中已经实现 **冻结 + continuation**：在 `StepQualityEvaluator` 里使用 `frozen_load_vector` 作为旧载荷，再按配置的 `alpha` 序列与新载荷进行 3~5 次混合求值；一旦步被接受，`frozen_load_vector` 会更新为最终混合载荷。
* 由于线性子问题仍假定载荷固定，`predicted_compliance` 未考虑反力随几何的变化；当壳体 FEA 重算后，实际柔度可能上升（即使载荷差值很小），导致 `predicted_reduction ≤ 0` 且步被拒绝。
* 目前需要进一步处理 **载荷灵敏度** 或额外约束（例如限制 `‖F_new−F_old‖/‖F_old‖`），否则单靠冻结/continuation 仍会出现“模型预测下降、实际却上升”的情况。

---

## 6) What Not to Do (for Now)

* Full **IQN-ILS** shell–truss inner iterations: powerful, but higher engineering cost unless you move to a true partitioned inner loop. The **precondition ideas** from pIQN-ILS are already captured by **Continuation + FIR**.

---

### References to Source Doc

* SUR & Aitken formulas; coupling residuals & stability: 
* Continuation (homotopy) on traction and structural parameters: 
* FIR interface filtering and accuracy/stability trade-off: 
* Empirical stability limits and hybrid preconditioning (pIQN-ILS): 

> Source: Lorentzon, J., & Revstedt, J. (2022). *On stability and relaxation techniques for partitioned fluid-structure interaction simulations*. **Engineering Reports**, 4(10), e12514. Open access.
