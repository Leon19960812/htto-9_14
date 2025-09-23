\documentclass[preprint,12pt]{elsarticle}

% Required packages
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{url}
\usepackage{float}
\usepackage{subcaption}
\usepackage{booktabs}

% Journal specific packages (automatically loaded by elsarticle)
% natbib, geometry, etc. are handled by elsarticle

\journal{Structures}

\begin{document}

\begin{frontmatter}

\title{Sequential Convex Programming for Truss Topology and Geometry Optimization under Design-Dependent Hydrostatic Loading}

\author[inst1]{Yuan Liang\corref{cor1}}
\ead{your.email@university.edu}

\cortext[cor1]{Corresponding author}

\affiliation[inst1]{organization={Your University},
                   addressline={Department of Civil Engineering},
                   city={Your City},
                   postcode={12345},
                   country={Your Country}}

\begin{abstract}

\end{abstract}

\begin{keyword}
Truss structures \sep Topology optimization \sep Sequential convex programming \sep Underwater structures \sep Design-dependent loading \sep Semidefinite programming \sep Structural optimization \sep Hydrostatic pressure
\end{keyword}

\end{frontmatter}

\section{Introduction}

Recent decades have seen topology optimization evolve into a cornerstone methodology across multiple engineering disciplines including architecture\cite{li2025interactive,isaac2024automated}, civil engineering\cite{moghaddam2025improvement}, advanced manufacturing\cite{li2024strength}, aerospace, and biomedical design. The field of topology optimization encompasses two fundamentally different 
methodologies: continuum and discrete. In architecture and civil engineering, discrete structures, namely truss, are widely used in practical projects due to its superior stiffness-to-weight ratio and high modularity\cite{lai2025new}. The truss optimization problem was first formulated by Michell\cite{michell1904lviii}, who established theoretical limits for minimum material consumption and derived optimal structural configurations. However, Michell's solutions typically involved an infinite number of members, which precluded practical implementation. To enable computational optimization for engineering applications, Dorn et al.\cite{dorn1964automatic} introduced the ground structure approach, wherein a finite set of potential members is predefined and optimization determines which members to retain and their corresponding cross-sectional areas. Ben-Tal and Nemirovski \cite{ben1997robust} later  reformulated this as a semidefinite programming (SDP) problem, establishing a convex optimization framework that transforms the compliance minimization problem into a tractable form with guaranteed global optimality.

While these pioneering researches have established truss topology optimization as a mature field, they fundamentally constrain the design space by fixed nodal positions a priori. Joint topology-geometry optimization addresses this constraint by optimizing node position and cross-section area simultaneously\cite{pedersen1972optimal,svanberg1981optimization,deb2001design} or alternatively\cite{kovcvara1996mathematics,ringertz1985topology}, Introducing the node position as a variable leads to a highly nonlinear problem; nevertheless, this approach suffices to work with a sparse ground structure, and results in a simpler structure\cite{achtziger2007simultaneous}. Thereby achieving enhanced structural efficiency and reduced material consumption while generating more practical designs that require less post-processing for real-world implementation\cite{weldeyesus2020truss}. 

The first work of joint topology-geometry optimization is presented by Pederson in 1970\cite{pedersen1970minimum}, he also gives the proof of an optimal design can always be found among the set of statically determinate structures. Peterson then extended his work to multiload cases\cite{pedersen1972optimal} and 3D trusses using sequential linear programming approaches\cite{pedersen1973optimal}. A major theoretical breakthrough came with Ben-Tal et al.\cite{ben1997robust,ben1993two}, who developed a rigorous mathematical framework for simultaneous topology-geometry optimization. Their approach combines semidefinite programming for the topology subproblem with nonsmooth optimization techniques to handle the inherent non-differentiability arising from geometric parameter variations. In computational programming field, Ben-Tal's approch is referred to as "implicit programming"\cite{outrata2013nonsmooth}. Although Achtziger\cite{achtziger2007simultaneous} point out that  most standard solvers of Nonlinear Optimization fail to directly solve the nonlinear joint topology-geometry optimization, Weldeyesus et al.\cite{weldeyesus2020truss} successfully addressed this problem using a standard primal-dual interior point implementation\cite{weldeyesus2018specialized}. 

However, these advances share a fundamental limitation: they assume that external loading conditions remain independent of the structural configuration and that the positions of supported and loaded joints are always fixed. This assumption, while simplifying the mathematical formulation, may not hold in many engineering applications where the applied forces are intrinsically coupled to the geometric design variables, creating design-dependent loading conditions. Design dependent loading can be classified into transmissible and pressure loading, self-weight and thermal loading\cite{deaton2014survey}. This paper concentrates on truss topology and geometry optimization under design dependent hydrostatic loading. 

The main approach employed in this paper is Sequential Convex Programming (SCP). To the best of the authors' knowledge, the SCP framework for structural optimization was first formalized by Zillober \cite{zillober1993globally}, who incorporated a line search procedure into the method of moving asymptotes to ensure global convergence. SCP has proven to be highly effective in structural optimization applications \cite{ni2005sequential}, particularly for problems involving displacement-dependent constraints that can be well-approximated through convex formulations. The fundamental principle of SCP lies in solving nonconvex optimization problems by iteratively constructing and solving a sequence of convex subproblems \cite{wang2025adaptive}. In our implementation, each subproblem is formulated as a semidefinite programming (SDP) problem that can be efficiently solved using modern optimization solvers such as MOSEK. The details are explained in chapter 3. 

The remainder of the paper is organized as follows. Section 2 presents the problem formulation for joint topology-geometry optimization of underwater truss structures, including the mathematical modeling of design-dependent hydrostatic loading and the inherent non-convexities in the coupled system. Section 3 develops the Sequential Convex Programming framework, detailing the linearization techniques, trust-region management, and the semidefinite programming formulation of each subproblem. Section 4 demonstrates the effectiveness of the proposed approach through numerical examples of underwater circular truss structures, comparing results with traditional fixed-loading methods and analyzing the impact of design-dependent loading on optimal configurations. Section 5 concludes the paper with a summary of key findings and directions for future research.


\section{Problem Formulation}

In this section, we present the problem formulation for joint topology-geometry optimization under design-dependent loading. We first review the evolution of truss optimization formulations to establish the mathematical foundation and highlight the unique challenges addressed in this work.

 Ben-Tal and Nemirovski\cite{ben1997robust} formulate the truss topology problem as a semidefinite program using the Schur complement lemma:

\begin{align}
\min_{\mathbf{A}, t} \quad & t \label{eq:sdp_objective} \\
\text{subject to} \quad & \begin{bmatrix} t & \mathbf{f}^T \\ \mathbf{f} & \mathbf{K}(\mathbf{A}) \end{bmatrix} \succeq \mathbf{0} \label{eq:schur_complement} \\
& \sum_{i=1}^{m} A_i \bar{L}_i \leq V_{\max} \label{eq:sdp_volume} \\
& A_{\min} \leq A_i \leq A_{\max}, \quad i = 1, \ldots, m \label{eq:sdp_bounds}
\end{align}

The Schur complement constraint \eqref{eq:schur_complement} ensures that $t \geq \mathbf{f}^T \mathbf{K}(\mathbf{A})^{-1} \mathbf{f}$, effectively minimizing the structural compliance while avoiding the explicit inversion of the stiffness matrix. This reformulation transforms the originally non-convex problem into a convex semidefinite program with guaranteed global optimality.

Weldeyesus et al. \cite{weldeyesus2020truss} extended this framework to simultaneously optimize nodal positions and cross section area:

\begin{align}
\min_{\mathbf{A}, \mathbf{v}} \quad & \mathbf{f}^T \mathbf{u} \label{eq:stability_objective} \\
\text{subject to} \quad & \begin{bmatrix} \zeta & \mathbf{f}^T \\ \mathbf{f} & \mathbf{K}(\mathbf{A}, \mathbf{v}) \end{bmatrix} \succeq \mathbf{0} \label{eq:weldeyesus_schur} \\
& \mathbf{K}(\mathbf{A}, \mathbf{v}) + \tau \mathbf{G}(\mathbf{A}, \mathbf{v}, \mathbf{u}) \succeq \mathbf{0} \label{eq:stability_constraint} \\
& \mathbf{v} \in \mathcal{V} \label{eq:geometry_domain} \\
& \sum_{i=1}^{m} A_i L_i(\mathbf{v}) \leq V_{\max} \label{eq:geometry_volume} \\
& A_{\min} \leq A_i \leq A_{\max}, \quad i = 1, \ldots, m \label{eq:geometry_bounds}
\end{align}

where $\mathbf{v}$ represents nodal coordinates, $\mathbf{G}(\mathbf{A}, \mathbf{v}, \mathbf{u})$ is the geometric stiffness matrix, and $\tau$ is a stability parameter. Crucially, this formulation maintains the SDP framework because the load vector $\mathbf{f}$ remains independent of the design variables.

Our work addresses a fundamental limitation of existing approaches: the assumption that loading conditions remain independent of structural configuration. For underwater structures, hydrostatic pressure creates an inherent coupling between geometry and applied forces. Using angular parameterization for circular truss structures, the formulation becomes:

\begin{align}
\min_{\mathbf{A}, \boldsymbol{\theta}} \quad & \mathbf{f}(\boldsymbol{\theta})^T \mathbf{u}(\mathbf{A}, \boldsymbol{\theta}) \label{eq:original_problem} \\
\text{subject to} \quad & \mathbf{K}(\mathbf{A}, \boldsymbol{\theta}) \mathbf{u} = \mathbf{f}(\boldsymbol{\theta}) \label{eq:equilibrium} \\
& \sum_{i=1}^{m} A_i L_i(\boldsymbol{\theta}) \leq V_{\max} \label{eq:volume_constraint} \\
& A_{\min} \leq A_i \leq A_{\max}, \quad i = 1, \ldots, m \label{eq:area_bounds} \\
& \boldsymbol{\theta} \in \Theta \label{eq:geometry_feasible}
\end{align}

Following the Schur complement approach, this can be formally written as:

\begin{align}
\min_{\mathbf{A}, \boldsymbol{\theta}, t} \quad & t \label{eq:design_dependent_sdp} \\
\text{subject to} \quad & \begin{bmatrix} t & \mathbf{f}(\boldsymbol{\theta})^T \\ \mathbf{f}(\boldsymbol{\theta}) & \mathbf{K}(\mathbf{A}, \boldsymbol{\theta}) \end{bmatrix} \succeq \mathbf{0} \label{eq:nonconvex_schur} \\
& \sum_{i=1}^{m} A_i L_i(\boldsymbol{\theta}) \leq V_{\max} \label{eq:design_volume} \\
& A_{\min} \leq A_i \leq A_{\max}, \quad i = 1, \ldots, m \label{eq:design_bounds} \\
& \boldsymbol{\theta} \in \Theta \label{eq:design_geometry}
\end{align}

where:
\begin{itemize}
\item $\mathbf{A} = [A_1, A_2, \ldots, A_m]^T$ are the cross-sectional areas
\item $\boldsymbol{\theta} = [\theta_1, \theta_2, \ldots, \theta_n]^T$ are the nodal angular positions
\item $\mathbf{K}(\mathbf{A}, \boldsymbol{\theta})$ is the global stiffness matrix
\item $\mathbf{f}(\boldsymbol{\theta})$ is the design-dependent load vector
\item $L_i(\boldsymbol{\theta})$ is the length of element $i$
\item $\Theta$ is the feasible set for nodal positions
\end{itemize}

For underwater truss structures, the loading conditions arise from shell-structure interaction where the truss serves as an internal support system for a pressurized outer shell. Consider a circular truss positioned within a cylindrical shell submerged at depth $H$. 

The outer shell experiences hydrostatic pressure distribution:
\begin{equation}
p(y) = \rho_w g (H - y) \label{eq:hydrostatic_pressure}
\end{equation}
where $\rho_w$ is water density, $g$ is gravitational acceleration, and $y$ is the vertical coordinate.

The truss nodes at angular positions $\boldsymbol{\theta} = [\theta_1, \theta_2, ..., \theta_n]$ provide discrete support points to the shell. Through shell finite element analysis, the support reactions transmitted to the truss are computed as:
\begin{equation}
\mathbf{R}(\boldsymbol{\theta}) = \text{ShellFEA}(p(y), \boldsymbol{\theta}) \label{eq:shell_fea}
\end{equation}

The resulting load vector applied to the truss becomes:
\begin{align}
f_{x,i} &= R_{x,i}(\boldsymbol{\theta}) \label{eq:reaction_fx} \\
f_{y,i} &= R_{y,i}(\boldsymbol{\theta}) \label{eq:reaction_fy}
\end{align}

This shows that the load vector $\mathbf{f}(\boldsymbol{\theta})$ explicitly depends on the support configuration through the shell structural response, creating the design-dependent loading condition. The support reactions account for the complex load redistribution within the shell structure and cannot be simplified to direct pressure application.

 This bilinear coupling between cross-sectional areas and nodal positions, combined with the design-dependent loading, creates multiple sources of non-convexity that preclude direct solution using standard SDP solvers. 

The key challenges include:
\begin{enumerate}
\item Bilinear coupling: $\mathbf{K}(\mathbf{A}, \boldsymbol{\theta}) = \sum_{i=1}^{m} A_i \mathbf{K}_i^{\text{geo}}(\boldsymbol{\theta})$
\item Geometric non-linearity: Element lengths $L_i(\boldsymbol{\theta})$ and directions depend non-linearly on $\boldsymbol{\theta}$
\item Load-geometry coupling: $\mathbf{f}(\boldsymbol{\theta})$ varies with structural configuration
\item Non-convex feasible region: The constraint \eqref{eq:nonconvex_schur} defines a non-convex set
\end{enumerate}

Given these complexities, we employ a Sequential Convex Programming approach developed in Section 3, which systematically handles the non-convex couplings through iterative convex approximations while preserving the computational advantages of the SDP framework.

\section{Sequential Convex Programming Approach}

In this section, we present a Sequential Convex Programming framework for joint topology-geometry optimization under design-dependent hydrostatic loading. We adopt the core SCP principle of iteratively solving convex approximations to handle nonconvex problems \cite{ni2005sequential}, but develop a fundamentally different implementation tailored to our specific problem structure. Unlike traditional SCP methods that employ reciprocal approximations and dual solvers, our approach uses Taylor linearization with trust-region control, formulating each subproblem as a semidefinite programming problem to leverage the inherent structure of truss optimization.This design enables direct utilization of robust SDP solvers like MOSEK, provides better numerical stability for handling bilinear couplings, and offers a more straightforward implementation pathway for practical engineering applications.

\subsection{Sensitivity Analysis and Gradient Computation}

The SCP framework requires accurate gradient information to construct the linearized subproblems at each iteration. For joint optimization of cross-sectional areas $\mathbf{A}$ and nodal angular positions $\boldsymbol{\theta}$, we need gradients of the compliance function with respect to both design variable types.

The gradient with respect to cross-sectional areas follows standard sensitivity analysis: $\frac{\partial C}{\partial A_i} = -\mathbf{u}^T \mathbf{K}_i^{\text{geo}}(\boldsymbol{\theta}) \mathbf{u}$, where $\mathbf{K}_i^{\text{geo}}(\boldsymbol{\theta})$ is the geometric stiffness matrix of element $i$.

For nodal positions, the sensitivity analysis is more complex due to the design-dependent loading condition. The total derivative of compliance $C(\mathbf{A}, \boldsymbol{\theta}) = \mathbf{f}(\boldsymbol{\theta})^T \mathbf{u}(\mathbf{A}, \boldsymbol{\theta})$ with respect to angular position $\theta_j$ includes both direct and indirect effects:

\begin{equation}
\frac{\partial C}{\partial \theta_j} = \frac{\partial \mathbf{f}}{\partial \theta_j}^T \mathbf{u} - \mathbf{u}^T \frac{\partial \mathbf{K}}{\partial \theta_j} \mathbf{u} \label{eq:total_gradient}
\end{equation}

The first term represents the direct effect of load redistribution as hydrostatic pressure varies with nodal positions, while the second term captures the indirect effect of geometric changes on structural response. For the hydrostatic loading scenario, the load gradient is $\frac{\partial f_{y,j}}{\partial \theta_j} = -\rho_w g R \cos(\theta_j) A_{\text{trib}}$, and the stiffness gradient accounts for changes in element geometry: $\frac{\partial \mathbf{K}}{\partial \theta_j} = \sum_{i=1}^{m} A_i \frac{\partial \mathbf{K}_i^{\text{geo}}}{\partial \theta_j}$.

These gradients enable the construction of linearized system matrices in equations~(25)-(27), where the coupling between load-geometry dependency and structural response changes is essential for the joint linearization process.

\subsection{Joint Linearization Framework}

At each iteration $k$, we linearize both the stiffness matrix and load vector around the current design point $(\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)})$ and solve the resulting joint convex subproblem.

\textbf{Linearization of System Matrices:}

The stiffness matrix is linearized with respect to both design variables $\mathbf{A}$ and $\boldsymbol{\theta}$, where the approximation is based on first-order Taylor expansion:
\begin{align}
\mathbf{K}_{\text{lin}}(\mathbf{A}, \boldsymbol{\theta}) &= \mathbf{K}(\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)}) + \sum_{i=1}^{m} (A_i - A_i^{(k)}) \mathbf{K}_i^{\text{geo}}(\boldsymbol{\theta}^{(k)}) \nonumber \\
&\quad + \sum_{j=1}^{n} (\theta_j - \theta_j^{(k)}) \sum_{i=1}^{m} A_i^{(k)} \frac{\partial \mathbf{K}_i^{\text{geo}}}{\partial \theta_j}\bigg|_{\boldsymbol{\theta}^{(k)}} \label{eq:joint_stiffness_linearization}
\end{align}

The design-dependent hydrostatic load vector is linearized as:
\begin{align}
\mathbf{f}_{\text{lin}}(\boldsymbol{\theta}) &= \mathbf{f}(\boldsymbol{\theta}^{(k)}) + \sum_{j=1}^{n} \frac{\partial \mathbf{f}}{\partial \theta_j}\bigg|_{\boldsymbol{\theta}^{(k)}} (\theta_j - \theta_j^{(k)}) \label{eq:joint_load_linearization}
\end{align}



where the gradient of the hydrostatic load vector with respect to angular position $\theta_i$ is:
\begin{equation}
\frac{\partial \mathbf{f}}{\partial \theta_i} = \begin{bmatrix} 0 \\ -\rho_w g R \cos(\theta_i) A_{trib} \\ \vdots \end{bmatrix} \label{eq:load_gradient}
\end{equation}

Using the linearized system matrices $\mathbf{K}_{\text{lin}}(\mathbf{A}, \boldsymbol{\theta})$ and $\mathbf{f}_{\text{lin}}(\boldsymbol{\theta})$, the joint subproblem at iteration $k$ becomes:

\begin{align}
\min_{\mathbf{A}, \boldsymbol{\theta}, t} \quad & t \label{eq:joint_objective} \\
\text{subject to} \quad & \begin{bmatrix} t & \mathbf{f}_{\text{lin}}(\boldsymbol{\theta})^T \\ \mathbf{f}_{\text{lin}}(\boldsymbol{\theta}) & \mathbf{K}_{\text{lin}}(\mathbf{A}, \boldsymbol{\theta}) \end{bmatrix} \succeq 0 \label{eq:joint_schur} \\
& \sum_{i=1}^{m} A_i L_i(\boldsymbol{\theta}^{(k)}) \leq V_{\max} \label{eq:joint_volume} \\
& A_{\min} \leq A_i \leq A_{\max}, \quad i = 1, \ldots, m \label{eq:joint_area_bounds} \\
& \boldsymbol{\theta} \in \Theta \label{eq:joint_geometry_constraints} \\
& \|\boldsymbol{\theta} - \boldsymbol{\theta}^{(k)}\|_2 \leq \Delta^{(k)} \label{eq:joint_trust_region}
\end{align}


\subsection{Geometry Constraints and Trust Region Management}

For circular truss structures, we parameterize nodal positions using angular coordinates. The geometry constraint set $\Theta$ includes:

\begin{align}
\theta_{\min} &\leq \theta_1 < \theta_2 < \cdots < \theta_n \leq \theta_{\max} \label{eq:monotonicity} \\
\theta_{i+1} - \theta_i &\geq \delta_{\min}, \quad i = 1, \ldots, n-1 \label{eq:min_spacing} \\
\theta_i + \theta_{n+1-i} &= \pi, \quad i = 1, \ldots, \lfloor n/2 \rfloor \label{eq:symmetry}
\end{align}

Trust region (TR) method is a powerful gradient-based algorithm which introduces a criteria to judge the quality of each step \cite{hu2022adaptive}. The trust region algorithm limits the step size within a radius $\Delta^{(k)}$ around the current design point:
\begin{equation}
\|\boldsymbol{\theta} - \boldsymbol{\theta}^{(k)}\|_2 \leq \Delta^{(k)} \label{eq:trust_region_constraint}
\end{equation}

The trust region radius $\Delta^{(k)}$ is updated based on the agreement between the actual objective function reduction and the predicted reduction from the linearized model. This agreement is quantified by the \textit{step quality ratio}:
\begin{equation}
\rho^{(k)} = \frac{C(\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)}) - C(\mathbf{A}^{(k+1)}, \boldsymbol{\theta}^{(k+1)})}{m^{(k)}(\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)}) - m^{(k)}(\mathbf{A}^{(k+1)}, \boldsymbol{\theta}^{(k+1)})} \label{eq:step_quality_ratio}
\end{equation}
where $C(\mathbf{A}, \boldsymbol{\theta}) = \mathbf{f}(\boldsymbol{\theta})^T \mathbf{u}(\mathbf{A}, \boldsymbol{\theta})$ is the actual compliance evaluated using the original nonlinear system, and $m^{(k)}$ represents the compliance predicted by the linearized model at iteration $k$. The step quality ratio $\rho^{(k)}$ measures how well the linearized approximation predicts the actual behavior: $\rho^{(k)} \approx 1$ indicates excellent agreement, while $\rho^{(k)} \ll 1$ suggests poor linearization accuracy.

Based on the step quality ratio, the trust region radius is updated according to:
\begin{equation}
\Delta^{(k+1)} = \begin{cases}
\gamma_{\text{expand}} \Delta^{(k)} & \text{if } \rho^{(k)} > \eta_2 \\
\Delta^{(k)} & \text{if } \eta_1 \leq \rho^{(k)} \leq \eta_2 \\
\gamma_{\text{shrink}} \Delta^{(k)} & \text{if } \rho^{(k)} < \eta_1
\end{cases} \label{eq:trust_region_update}
\end{equation}
When $\rho^{(k)} > \eta_2$, the linearization proves highly accurate, justifying an expanded trust region for the next iteration. Conversely, when $\rho^{(k)} < \eta_1$, the linearization quality is insufficient, necessitating both trust region reduction and step rejection. For intermediate values of $\rho^{(k)}$, the current trust region size is maintained while accepting the computed step.

Trust region parameters are chosen following established guidelines in the literature. The shrinking and expanding factors ($\gamma_{\text{shrink}} = 0.5$, $\gamma_{\text{expand}} = 2.0$) align with typical values suggested in \cite{yuan2000review}, while the threshold parameters ($\eta_1 = 0.01$, $\eta_2 = 0.75$) are set within the commonly used ranges for practical engineering applications \cite{yuan2015recent}.

\subsection{Algorithm}

An overview of SCP algorithm is summarized in algorithm 1. The algorithm integrates the linearization framework developed in Section 3.2 with the trust region management strategy outlined in Section 3.3. At each iteration, the algorithm performs three main operations: joint linearization of both stiffness matrix and load vector around the current design point, solution of the SDP subproblem, and adaptive trust region radius adjustment based on step quality assessment. The algorithm simultaneously optimizes cross-sectional areas $\mathbf{A}$ and nodal positions $\boldsymbol{\theta}$, with the trust region constraint $\|\boldsymbol{\theta} - \boldsymbol{\theta}^{(k)}\|_2 \leq \Delta^{(k)}$ ensuring the validity of linearization approximations. The global convergence of the proposed SCP algorithm is guaranteed by the well-established trust region theory \cite{yuan2015recent}. The algorithm satisfies the standard convergence conditions: sufficient descent property through the linearized subproblems, bounded model errors within the trust region, and adaptive step control via the trust region radius management. Under standard assumptions on problem regularity, the sequence of iterates converges to a stationary point of the original nonconvex problem. The convergence rate depends on the linearization quality and typically requires 20-40 iterations for problems of moderate size, as demonstrated in the numerical examples.

\begin{algorithm}
\caption{Joint Sequential Convex Programming for Truss Optimization}
\begin{algorithmic}[1]
\STATE Initialize $\boldsymbol{\theta}^{(0)}$, $\mathbf{A}^{(0)}$, $\Delta^{(0)}$
\STATE Set iteration counter $k \leftarrow 0$
\WHILE{stopping criteria is not satisfied}
    \STATE \textbf{Joint Linearization:}
    \STATE Compute gradients $\frac{\partial \mathbf{K}}{\partial \theta_j}|_{(\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)})}$ and $\frac{\partial \mathbf{f}}{\partial \theta_j}|_{\boldsymbol{\theta}^{(k)}}$ using equation~\eqref{eq:load_gradient}
    \STATE Construct linearized system matrices $\mathbf{K}_{\text{lin}}(\mathbf{A}, \boldsymbol{\theta})$ and $\mathbf{f}_{\text{lin}}(\boldsymbol{\theta})$ using equations~(25) and~(26)
    \STATE \textbf{Joint SDP Solution:}
    \STATE Solve joint SDP subproblem \eqref{eq:joint_objective}--\eqref{eq:joint_trust_region} for $(\mathbf{A}^{(k+1)}, \boldsymbol{\theta}^{(k+1)})$
    \STATE \textbf{Trust Region Update:}
    \STATE Evaluate step quality $\rho^{(k)}$ using equation~(38)
    \IF{$\rho^{(k)} < \eta_1$}
        \STATE Reject step: $(\mathbf{A}^{(k+1)}, \boldsymbol{\theta}^{(k+1)}) = (\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)})$
        \STATE Reduce trust region using equation~(39)
    \ELSE
        \STATE Accept step and update trust region using equation~(39)
    \ENDIF
    \STATE Check convergence criteria
    \STATE $k \leftarrow k + 1$
\ENDWHILE
\STATE \textbf{return} Optimal design $(\mathbf{A}^*, \boldsymbol{\theta}^*)$
\end{algorithmic}
\end{algorithm}


\section{Numerical Results}

The Sequential Convex Programming framework has been implemented in Python with gradient computations and trust region management handled through custom algorithms. The semidefinite programming subproblems within each SCP iteration are formulated using the CVXPY optimization modeling language and solved by MOSEK. All numerical experiments have been performed on a PC equipped with an Intel(R) Core(TM) i7-9750H CPU running at 2.60 GHz with 32 GB RAM.

For the numerical parameters, we use normalized input data following standard practice in topology optimization literature, where all physical quantities are made dimensionless by appropriate scaling factors. The material properties are normalized with Young's modulus $E = 1$ and density $\rho = 1$. For the hydrostatic loading scenarios, we use water density $\rho_w = 1000$ kg/m$^3$ and gravitational acceleration $g = 9.81$ m/s$^2$. The structural depth $H$ is specified for each example case.

Gradient computations for the geometric design variables utilize finite difference approximation with step size $h = 1 \times 10^{-6}$. The convergence criteria include both relative change in objective function (tolerance $1 \times 10^{-4}$) and constraint satisfaction (tolerance $1 \times 10^{-6}$). In the optimal design visualizations, only structural members with cross-sectional area $A_i \geq 0.001A_{\max}$ are displayed, where $A_{\max}$ represents the maximum cross-sectional area in the current design. Spherical markers indicate the optimized joint positions.

\subsection{Fixed Geometry (SDP) vs Moving Nodes (SCP)}

In this section, we present a semicircular truss dome under hydrostatic loading  case to demonstrate the advantages of geometry optimization in dealing with  design-dependent pressure conditions. This is done by comparing solutions obtained with fixed geometry (SDP) and movable nodes (SCP). Moreover, we 
demonstrate the performance improvements in structural compliance achieved  through the proposed SCP framework.

To comprehensively evaluate the performance of SCP and SDP, we conduct comparisons across different discretization levels using 16, 12, and 8 sectors. This parametric study reveals how the benefits of geometry optimization vary with mesh density and helps identify scenarios where joint topology-geometry optimization provides the most significant improvements.

The connectivity level follows the three-level ground structure as Cai\cite{CAI2025109205} recommends in truss optimization problems. The baseline configuration (16 sectors) creates a ground structure with two layers of trusses, 51 nodes and 262 members in total, as shown in Fig.\ref{fig:ground_structure_complete}. Structures with fewer sectors maintain the same connectivity pattern but with correspondingly fewer nodes and members. The volume fraction is set to 0.2 in all three cases.  


% 图片代码
\begin{figure}[htbp]
    \centering
    % 第一行
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ground structure/level1.pdf}
        \caption{Level 1 connections}
        \label{fig:level1}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ground structure/level2.pdf}
        \caption{Level 2 connections}
        \label{fig:level2}
    \end{subfigure}
    
    \vspace{0.2cm}  % 行间距
    
    % 第二行
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ground structure/level3.pdf}
        \caption{Level 3 connections}
        \label{fig:level3}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ground structure/level123.pdf}
        \caption{Complete ground structure (Levels 1-3)}
        \label{fig:level123}
    \end{subfigure}
    
    \caption{Ground structure connectivity : (a) Level 1 - adjacent node connections, (b) Level 2 - connections spanning two nodes, (c) Level 3 - connections spanning three nodes, (d) Complete ground structure with all connection levels}
    \label{fig:ground_structure_complete}
\end{figure}



\begin{figure}[htbp]
    \centering
    \makebox[0.49\textwidth]{\fontsize{10}{12}\selectfont\textbf{SDP}}\hfill
    \makebox[0.49\textwidth]{\fontsize{10}{12}\selectfont\textbf{SCP}}\\[0.5em]
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{sdp vs scp/sdp_16.pdf}
        \caption{Compliance=585.6J  Effective members:126}
        \label{fig:sdp_results_16}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{sdp vs scp/scp_16.pdf}
        \caption{Compliance=479.7J Effective members:106}
        \label{fig:scp_results_16}
    \end{subfigure}  

    \vspace{0.2cm}  % 行间距

    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{sdp vs scp/sdp_12.pdf}
        \caption{Compliance=647.8J Effective members:95}
        \label{fig:sdp_results_12}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{sdp vs scp/scp_12.pdf}
        \caption{Compliance=503.4J Effective members:80}
        \label{fig:scp_results_12}
    \end{subfigure} 

    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{sdp vs scp/sdp_8.pdf}
        \caption{Compliance=811.5J Effective members:96}
        \label{fig:sdp_results_8}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{sdp vs scp/scp_8.pdf}
        \caption{Compliance=552.0J Effective members:46}
        \label{fig:scp_results_8}
    \end{subfigure} 

    \caption{SDP vs SCP}
    \label{fig:sdp_scp_comparison}
\end{figure}


It is obvious from Fig\ref{fig:sdp_scp_comparison} that across all discretizations, the SCP consistently achieve lower compliance than the fixed-geometry SDP baselines. Measured at the same volume and visualization filter ($A_i \ge 0.001A_{\max}$), the relative improvements of SCP over SDP increase as the sector number decreases (i.e., as the angular discretization becomes coarser), The computational statistics for all cases are presented in Table\ref{tab:compliance_comparison}.

\begin{table}[htbp]
\centering
\caption{Compliance comparison between SDP and SCP}
\label{tab:compliance_comparison}
\begin{tabular}{cccc}
\toprule
Sectors & SDP Compliance (J) & SCP Compliance (J) & Improvement (\%) \\
\midrule
16 & 585.6 & 479.7 & 18.1 \\
12 & 647.8 & 503.4 & 22.3 \\
8  & 811.5 & 552.0 & 32.0 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    % 列标题
    \makebox[0.49\textwidth]{\large\textbf{SDP}}\hfill
    \makebox[0.49\textwidth]{\large\textbf{SCP}}\\[0.5em]
    
    % 第一行：16扇形
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{area distribution/sdp_16_area.pdf}
        \caption{SDP - 16 sectors}
        \label{fig:sdp_16_area}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{area distribution/scp_16_area.pdf}
        \caption{SCP - 16 sectors}
        \label{fig:scp_16_area}
    \end{subfigure}
    
    \vspace{0.3cm}
    
    % 第二行：12扇形
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{area distribution/sdp_12_area.pdf}
        \caption{SDP - 12 sectors}
        \label{fig:sdp_12_area}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{area distribution/scp_12_area.pdf}
        \caption{SCP - 12 sectors}
        \label{fig:scp_12_area}
    \end{subfigure}
    
    \vspace{0.3cm}
    
    % 第三行：8扇形
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{area distribution/sdp_8_area.pdf}
        \caption{SDP - 8 sectors}
        \label{fig:sdp_8_area}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{area distribution/scp_8_area.pdf}
        \caption{SCP - 8 sectors}
        \label{fig:scp_8_area}
    \end{subfigure}
    
    \caption{Cross-sectional area distribution comparison between SDP and SCP across different sector discretizations}
    \label{fig:area_distribution_comparison}
\end{figure}

Fig.\ref{fig:area_distribution_comparison} examines the cross-sectional area distributions. SDP exhibits a characteristic bimodal pattern with members concentrated near both the removal threshold (0-1000 mm²) and maximum area limit (10,000 mm²). This pattern intensifies as discretization becomes coarser, with the 8-sector case showing 18 near-threshold members.

SCP demonstrates a more balanced distribution across intermediate areas (2000-6000 mm²), with only 4 near-threshold members in the 8-sector case. This selective material allocation explains how SCP achieves superior performance with fewer active members (46 vs 96 effective members for 8 sectors).

\subsection{Parameter Sensitivity Analysis}


\subsection{Algorithm Performance Analysis}


\clearpage
\section*{Acknowledgments}


\bibliographystyle{elsarticle-num}
\bibliography{references}

\end{document}