\subsection*{Regularity and Scope}

We solve the joint topology--geometry problem with a trust-region sequential convex programming (SCP) scheme. At each iteration, we form a convex semidefinite subproblem by linearizing the truss stiffness $\mathbf{K}(\mathbf{A},\boldsymbol{\theta})$ and the design-dependent loads $\mathbf{f}(\boldsymbol{\theta})$ (via the Schur complement) and solve it with MOSEK.

When $\mathbf{K}$ and $\mathbf{f}$ vary smoothly with $\boldsymbol{\theta}$ and the trust-region together with engineering constraints admits a strictly feasible point, the linearized SDP subproblem is well posed and yields stable primal--dual solutions. This supports reliable step acceptance and trust-region updates in our implementation. These are \emph{local} properties and do not imply a unique global optimum for the original nonconvex problem. In practice, the method converges to a regular local design that may depend on initialization; therefore, we report initial designs, tolerances, and convergence diagnostics alongside the results.

We intentionally avoid heavy formalism here and do not claim global uniqueness. The emphasis is on a practically robust SCP tailored to design-dependent hydrostatic loading, where load Jacobians are handled consistently within the same linearization framework.
