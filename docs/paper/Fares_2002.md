# Summary of Method: Robust Control via Sequential Semidefinite Programming (SSDP)

## 1. Problem Setting

The authors address a class of **nonconvex optimization problems with matrix inequality constraints** that arise in robust control design.

The generic form is:

\[
\begin{aligned}
\min_x \quad & d^T x \\
\text{s.t.} \quad & \mathcal{A}(x) \le 0, \\
& \mathcal{B}(x) = 0,
\end{aligned}
\]

where:

- \(\mathcal{A}(x) \le 0\) denotes **LMI (Linear Matrix Inequality)** constraints;
- \(\mathcal{B}(x) = 0\) represents **nonlinear matrix equality** constraints (often bilinear or rank-type);
- \(x\) is a vector collecting all decision variables.

This formulation covers many robust control problems such as reduced-order \(H_\infty\) synthesis, static output feedback, and gain-scheduling design.

---

## 2. Key Transformation: Handling the Rank Constraint

A reduced-order \(H_\infty\) design involves a **rank constraint**:
\[
\mathrm{rank}(Q(x)) \le r.
\]

This is replaced by introducing a **slack matrix variable** \(W\) such that:
\[
Q(x) = W^T W.
\]

The nonconvex rank constraint becomes a **quadratic matrix equality**:
\[
\mathcal{B}(x, W) = Q(x) - W^T W = 0.
\]

This allows reformulating the entire optimization problem in a smoother, differentiable form.

---

## 3. Reformulated Optimization Model

Let the design vector include both original and slack variables:
\[
\tilde{x} = (x, W).
\]

Then the problem takes the form:

\[
\begin{aligned}
\min_{\tilde{x}} & \quad d^T x \\
\text{s.t.} & \quad \mathcal{A}(x) \le 0, \\
& \quad \mathcal{B}(\tilde{x}) = Q(x) - W^T W = 0.
\end{aligned}
\]

This formulation is called problem (D).

---

## 4. Augmented Penalty Version

To stabilize the numerical behavior, they introduce a **penalty-augmented objective**:

\[
f_c(x) = \gamma + \frac{c}{2} \|\Phi_u \tilde{\Phi}_u - I\|^2,
\]

where \(c > 0\) is the penalty parameter.

The augmented problem is:

\[
\begin{aligned}
\min_x & \quad f_c(x) \\
\text{s.t.} & \quad \mathcal{A}(x) \le 0, \\
& \quad \mathcal{B}(x) = \Phi_u \tilde{\Phi}_u - I = 0.
\end{aligned}
\]

This penalizes violation of the nonlinear constraint while keeping LMI constraints explicit.

---

## 5. Sequential Semidefinite Programming (SSDP) Algorithm

The SSDP method generalizes Sequential Quadratic Programming (SQP) to the **semidefinite programming** setting.

### Step 1. Linearize Nonlinear Constraints

At iteration \(k\), linearize the nonlinear equality \(\mathcal{B}(x) = 0\) around the current iterate \(x_k\):

\[
\mathcal{B}(x_k + \Delta x) \approx \mathcal{B}(x_k) + \nabla \mathcal{B}(x_k) \, \Delta x = 0.
\]

### Step 2. Quadratic Approximation of the Lagrangian

Define the **Lagrangian** of the augmented problem:
\[
L_c(x; \Lambda, \lambda) = f_c(x) + \text{trace}(\Lambda \mathcal{A}(x)) + \lambda^T \text{vec}(\mathcal{B}(x)),
\]
where \(\Lambda\) and \(\lambda\) are Lagrange multipliers.

Approximate \(L_c\) by its **second-order Taylor expansion**:
\[
L_c(x + \Delta x) \approx L_c(x) + \nabla L_c(x)^T \Delta x + \tfrac{1}{2} \Delta x^T \nabla^2 L_c(x) \Delta x.
\]

### Step 3. Tangent Subproblem (T)

The resulting **tangent subproblem** is:

\[
\begin{aligned}
\min_{\Delta x} \quad &
\nabla f_c(x)^T \Delta x + \frac{1}{2} \Delta x^T \nabla^2 L_c(x) \Delta x \\
\text{s.t.} \quad &
\mathcal{A}(x + \Delta x) \le 0, \\
& \mathcal{B}(x) + \nabla \mathcal{B}(x) \Delta x = 0.
\end{aligned}
\]

This subproblem is convex if the reduced Hessian is made positive semidefinite.

### Step 4. Convexification

If \(\nabla^2 L_c(x)\) is indefinite, perform **convexification** using modified Cholesky or Gauss–Newton techniques to ensure convexity of the subproblem.

### Step 5. Semidefinite Subproblem

After convexification, (T) becomes a **Semidefinite Program (SDP)**, which can be efficiently solved by standard SDP solvers.

### Step 6. Line Search / Trust Region

Use line search or trust-region mechanisms to update the iterate:
\[
x_{k+1} = x_k + \alpha_k \Delta x_k,
\]
with step size \(\alpha_k\) chosen to ensure sufficient decrease and maintain feasibility.

---

## 6. Convergence Properties

Under mild regularity assumptions (full rank Jacobians, second-order sufficiency), SSDP exhibits:

- **Global convergence** (if using line search or trust region);
- **Superlinear or quadratic local convergence**, similar to SQP;
- Theoretical justification for positive-definite reduced Hessian.

---

## 7. Summary of Algorithm

**SSDP Algorithm:**

1. Initialize \(x_0\) satisfying \(\mathcal{A}(x_0) \le 0\) and full-rank \(\Phi_u, \tilde{\Phi}_u\).
2. Set Lagrange multipliers \(\Lambda_0 \ge 0, \lambda_0\).
3. Repeat until convergence:
   - Linearize \(\mathcal{B}(x)\) and approximate \(L_c(x)\).
   - Solve convex SDP subproblem (T).
   - Update multipliers and iterate with line search/trust region.
4. Stop when both constraint violation and gradient norm are below tolerance.

---

## 8. Relation to Other Methods

| Method | Comparison |
|--------|-------------|
| **Augmented Lagrangian** | Slower (linear convergence), may require large penalty parameters. |
| **Concave Programming** | Computationally heavier and less stable. |
| **SSDP** | Second-order method with superlinear convergence; preserves LMI structure. |

---

## 9. Applications Demonstrated

- Robust gain-scheduling control of uncertain systems;
- Reduced-order \(H_\infty\) synthesis;
- Static output feedback;
- Robust autopilot design (missile example).

---

## 10. Core Idea in One Sentence

> SSDP extends Sequential Quadratic Programming (SQP) to problems with LMI constraints by iteratively solving **convex semidefinite subproblems** derived from the **second-order Taylor expansion** of the Lagrangian and **linearization of nonlinear matrix equalities**, ensuring robust and theoretically sound convergence.
