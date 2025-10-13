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

Recent decades have seen topology optimization evolve into a cornerstone methodology across multiple engineering disciplines including architecture\cite{li2025interactive,isaac2024automated}, civil engineering\cite{moghaddam2025improvement}, advanced manufacturing\cite{li2024strength}, aerospace\cite{DAGKOLU2021238}, and biomedical design\cite{VILARDELL2019138330}. The field of topology optimization encompasses two fundamentally different methodologies: continuum and discrete. In architecture and civil engineering, discrete structures, namely truss, are widely used in practical projects due to its superior stiffness-to-weight ratio and high modularity\cite{lai2025new}. The truss optimization problem was first formulated by Michell\cite{michell1904lviii}, who established theoretical limits for minimum material consumption and derived optimal structural configurations. However, Michell's solutions typically involved an infinite number of members, which precluded practical implementation. To enable computational optimization for engineering applications, Dorn et al.\cite{dorn1964automatic} introduced the ground structure approach, wherein a finite set of potential members is predefined and optimization determines which members to retain and their corresponding cross-sectional areas. Ben-Tal and Bendsøe \cite{ben1993new} recast the compliance minimization problem into a nonsmooth convex optimization framework, showing that the truss topology design problem can be expressed as the maximum of convex quadratic forms. This laid the theoretical foundation for the later semidefinite programming (SDP) formulation by Ben-Tal and Nemirovski\cite{ben1997robust}. The SDP formulation provides the basis for the subproblem adopted in this paper, as detailed in section 2.  

While these pioneering researches have established truss topology optimization as a mature field, they fundamentally constrain the design space by fixed nodal positions a priori. Joint topology-geometry optimization addresses this constraint by optimizing node position and cross-section area simultaneously\cite{pedersen1972optimal,svanberg1981optimization,deb2001design} or alternatively\cite{kovcvara1996mathematics,ringertz1985topology}, Introducing the node position as a variable leads to a highly nonlinear problem; nevertheless, this approach suffices to work with a sparse ground structure, and results in a simpler structure\cite{achtziger2007simultaneous}. Thereby achieving enhanced structural efficiency and reduced material consumption while generating more practical designs that require less post-processing for real-world implementation\cite{weldeyesus2020truss}. 

The first work of joint topology-geometry optimization is presented by Pederson in 1970\cite{pedersen1970minimum}, he also gives the proof of an optimal design can always be found among the set of statically determinate structures. Peterson then extended his work to multiload cases\cite{pedersen1972optimal} and 3D trusses using sequential linear programming approaches\cite{pedersen1973optimal}. A major theoretical breakthrough came with Ben-Tal et al.\cite{ben1997robust,ben1993two}, who developed a rigorous mathematical framework for simultaneous topology-geometry optimization. Their approach combines semidefinite programming for the topology subproblem with nonsmooth optimization techniques to handle the inherent non-differentiability arising from geometric parameter variations. In computational programming field, Ben-Tal's approch is referred to as "implicit programming"\cite{outrata2013nonsmooth}. We refer the readers to Achtziger's article\cite{achtziger2007simultaneous} for a explicit review on the implicit programming and its improved approach. Apart from this, different methodologies have been developed to address the nonconvexity in truss topology-geometry optimization, including the method of moving asymptotes(MMA)\cite{svanberg1987method}, and its global convergent version\cite{zillober1993globally}, the specialized primal dual interior point method\cite{weldeyesus2018specialized} and the extended version considering global stability constraints.  

In this paper, we address the topology-geometry problem using the Sequential Convex Programming(SCP) approach. Such problem has been extensively studied but with a fundamental limitation: they assume that external loading conditions remain independent of the structural configuration and that the positions of loaded joints are always fixed. This assumption, while simplifying the mathematical formulation, may not hold in many engineering applications where the applied forces are intrinsically coupled to the geometric design variables, creating design-dependent loading conditions\cite{lee2012structural}. Design dependent loading conditions can be classified into volumetric and surface loads. The first includes thermal expansion and self-weight loads, the second acts on the boundaries including distributed pressure loads, heat transfer, and fluid-structure interaction\cite{deaton2014survey,picelli2019topology}. Our study concentrates on topology-geometry optimization of a semicircular truss structure under design-dependent hydrostatic loading.

The proposed SCP framework embeds load–geometry coupling into tractable subproblems. By jointly linearizing stiffness and loads within trust regions, the nonconvex problem is approximated by a sequence of SDP subproblems that can be solved efficiently with interior-point methods. To the best of the authors’ knowledge, this is the first attempt to formulate and solve such a problem within an SCP setting.

The concept of Sequential Convex Programming (SCP) is not new and can be traced back to the seminal works of Svanberg and Zillober \cite{svanberg1987method,zillober1993globally}, where each iteration generates and solves a strictly convex approximating subproblem. Building upon this idea, Fares extended the framework by employing semidefinite programming (SDP) as the subproblem to address nonlinear semidefinite programming problems, which he referred to as S-SDP\cite{fares2002robust}. Correa further develops this idea to a global algorithm\cite{correa2004global}. Subsequent studies have demonstrated the effectiveness of SCP in a wide range of structural optimization applications \cite{ni2005sequential}. Although different terminologies and linearization schemes have been proposed, the key principle of SCP remains the same—nonconvex problems are solved through a sequence of convex approximations \cite{wang2025adaptive}. In our implementation, each subproblem is formulated as a semidefinite program and solved using the modern optimization solver MOSEK. Despite the strong nonconvexity of the overall problem, the solution procedure remains straightforward and accessible to engineers.

The remainder of the paper is organized as follows. Section 2 presents the problem formulation for joint topology-geometry optimization of underwater truss structures, including the mathematical modeling of design-dependent hydrostatic loading and the inherent non-convexities in the coupled system. Section 3 develops the Sequential Convex Programming framework, detailing the linearization techniques, trust-region management, and the semidefinite programming formulation of each subproblem. Section 4 demonstrates the effectiveness of the proposed approach through numerical examples of underwater semicircular truss structures, comparing results with traditional fixed loading methods and analyzing the impact of design-dependent loading on optimal configurations. Section 5 concludes the paper with a summary of key findings and directions for future research.


\section{Problem Formulation}

This section describes the essential mathematical formulation of the truss structure, the geometrical feasible set, and the hydrostatic loading. We then present the problem formulation for joint topology–geometry optimization under design‑dependent loading, and conclude by identifying the main sources of nonconvexity that motivate the sequential convex programming framework.  

\subsection{Ground Structure}
We adopt Dorn's ground-structure \cite{dorn1964automatic} paradigm with an algorithmically generated candidate set \cite{he2019python} in 2 dimensional space. For the truss structure parameterized by nodal angles $\boldsymbol{\theta}$, let $\mathbf{v}_j(\boldsymbol{\theta}) \in \mathbb{R}^2$ $(j=1,\ldots,n)$ denote the coordinates of the $n$ nodes. Let m be the number of members with the cross-sectional area $A_i, i = {1,\ldots,m}$. for each member with start nodes $\mathbf{v}_i^{(1)}$, and end nodes $\mathbf{v}_i^{(2)}$, the member length is
\begin{equation}
L_i(\boldsymbol{\theta}) = \bigl\|\mathbf{v}_i^{(2)}(\boldsymbol{\theta})-\mathbf{v}_i^{(1)}(\boldsymbol{\theta})\bigr\|.
\label{member_length}
\end{equation}

We construct the candidate member set via an adaptive, geometry-driven procedure akin to He's python script \cite{he2019python} but in a polar coordinate framework tailored to concentric rings. 

Nodes are placed on concentric semicircular rings with prescribed radii and uniformly distributed initial angles, with supports explicitly marked and eliminated from the free degrees of freedom. Candidate members are generated by filtering out any segment that is not entirely contained in the shell's ring domain (polygon coverage), ensuring the structural envelope is respected; connecting each node only to neighbors within a small angular window, $\Delta\theta \le K\,\overline{\Delta\theta}$ (with fixed $K$ and median angular gap $\overline{\Delta\theta}$), and pruning nearly collinear radial families by retaining only connections between adjacent radii while discarding longer members that skip intermediate rings (a gcd-like adjacency rule), thereby removing redundant long members without sacrificing load paths.

The resulting connection set defines the candidate members used throughout optimization. As $\boldsymbol{\theta}$ evolves, member lengths $L_i(\boldsymbol{\theta})$ are refreshed each iteration; the candidate set remains fixed unless an explicit node-merge procedure is invoked detailed in section 3.

\subsection{Geometrically Feasible Set}
To avoid singularity or non-differentiability caused by coincident nodes, we enforce angular spacing and per-node step caps under the polar parameterization. Let $\boldsymbol{\theta} = [\theta_1,\ldots,\theta_n]^\top$ denote the ordered angular positions of the free nodes, and let $r_j$ be the current radial distance of joint $j$ from the center. Define a small radius tolerance $\tau_r > 0$ and a minimal angular spacing $\delta_{\min} > 0$. We impose
\begin{align}
\theta_{j+1} &\ge \theta_j, \quad j=1,\ldots,n-1, \\
\theta_{j+1} - \theta_j &\ge \delta_{\min} \quad \text{whenever } |r_{j+1}-r_j| \le \tau_r.
\label{eq:min_spacing_rule}
\end{align}
Thus, nodes on the same ring are separated by a minimum angular gap of at least $\delta_{\min}$, while nodes on different rings maintain a non‑decreasing angular order.

For element $i$ with length $L_i(\boldsymbol{\theta})$, let the associated vector of direction cosines in element space be $\boldsymbol{\gamma}_i^{e}(\boldsymbol{\theta}) \,{=}\, \tfrac{1}{L_i(\boldsymbol{\theta})}(\mathbf{v}_q{-}\mathbf{v}_p)^{\top} \in \mathbb{R}^{1\times 2}$. Construct the global vector $\boldsymbol{\gamma}_i(\boldsymbol{\theta}) \in \mathbb{R}^{n_{\text{dof}}}$ by embedding $\bigl(-\boldsymbol{\gamma}_i^{e}(\boldsymbol{\theta}),\, \boldsymbol{\gamma}_i^{e}(\boldsymbol{\theta})\bigr)^{\top}$ at the DOFs of its end nodes (zeros elsewhere). The global stiffness is
\begin{equation}
\mathbf{K}(\mathbf{A}, \boldsymbol{\theta})
= \sum_{i=1}^{m} \frac{E\,A_i}{L_i(\boldsymbol{\theta})}\, \boldsymbol{\gamma}_i(\boldsymbol{\theta})\, \boldsymbol{\gamma}_i(\boldsymbol{\theta})^{\top}.
\end{equation}


\subsection{Hydrostatic Loading}

For underwater truss structures, the loading conditions arise from shell-structure interaction where the truss serves as an internal support system for a pressurized outer shell. Consider a circular truss positioned within a thin curved shell segment represented by a 2D semicircular slice (outer boundary a circular arc) submerged at depth $H$. 

The outer shell experiences hydrostatic pressure distribution:
\begin{equation}
p(y) = \rho_w g (H - y) \label{eq:hydrostatic_pressure}
\end{equation}
where $\rho_w$ is water density, $g$ is gravitational acceleration, and $y$ is the vertical coordinate.

The nodes at angular positions $\boldsymbol{\theta} = [\theta_1, \theta_2, ..., \theta_n]$ provide discrete support points to the shell. The discrete support locations are mapped to the shell boundary vertices using a Gaussian softmax over angular distance across all boundary nodes (temperature $\tau$), with $\tau$ proportional to the boundary angular grid spacing and optionally adapted to local span; this yields a smooth dependence on $\boldsymbol{\theta}$. The shell equilibrium and support reactions are defined implicitly by the augmented system.


\begin{equation}
\begin{bmatrix}
\mathbf{K}_\mathrm{s} & \mathbf{C}^{\top} \\
\mathbf{C} & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\mathbf{u}_\mathrm{s} \\
\boldsymbol{\lambda}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{f}_{\mathrm{p}} \\
\mathbf{0}
\end{bmatrix},\quad
\mathbf{C} = \mathbf{C}(\boldsymbol{\theta}),\ \ \mathbf{R}(\boldsymbol{\theta}) \equiv \boldsymbol{\lambda},
\label{eq:shell_fea}
\end{equation}
where $\mathbf{K}_\mathrm{s}$ is the shell stiffness, $\mathbf{C}(\boldsymbol{\theta})$ encodes the softmax-mapped supports on boundary vertices, $\mathbf{u}_\mathrm{s}$ are shell displacements, $\boldsymbol{\lambda}$ are support reactions, and $\mathbf{f}_{\mathrm{p}}$ is the pressure-induced load vector.

A hard nearest-vertex assignment makes $\mathbf{C}(\boldsymbol{\theta})$ piecewise constant and non-differentiable in $\boldsymbol{\theta}$, yielding undefined or zero gradients almost everywhere and breaking the first-order model used by SCP. With a softmax kernel (finite $\tau>0$), the weights and thus $\mathbf{C}(\boldsymbol{\theta})$ vary smoothly with $\boldsymbol{\theta}$, so the reactions $\mathbf{R}(\boldsymbol{\theta})$ and the induced truss loads $\mathbf{f}(\boldsymbol{\theta})$ are differentiable. This enables an analytical Jacobian used by the linearization step. Differentiating the augmented system.

Why softmax mapping. A hard nearest-vertex assignment makes $\mathbf{C}(\boldsymbol{\theta})$ piecewise constant and non-differentiable in $\boldsymbol{\theta}$, yielding undefined or zero gradients almost everywhere and breaking the first-order model used by SCP. With a softmax kernel (finite $\tau>0$), the weights and thus $\mathbf{C}(\boldsymbol{\theta})$ vary smoothly with $\boldsymbol{\theta}$, so the reactions $\mathbf{R}(\boldsymbol{\theta})$ and the induced truss loads $\mathbf{f}(\boldsymbol{\theta})$ are differentiable. Concretely, we use logits $\ell_i=-\tfrac{1}{2}(\Delta\theta_i/\tau)^2$ evaluated against all boundary nodes (no $k$-NN truncation), with $\tau\approx\sigma\,\Delta\theta_{\text{grid}}$ and an optional adaptive scaling by the local angular span to stabilize gradients. This enables an analytical Jacobian used by the linearization step. Differentiating the augmented system



\begin{equation}
\underbrace{\begin{bmatrix}
\mathbf{K}_\mathrm{s} & \mathbf{C}^{\top} \\
\mathbf{C} & \mathbf{0}
\end{bmatrix}}_{\mathcal{S}_\mathrm{s}(\boldsymbol{\theta})}
\begin{bmatrix}
\mathbf{u}_\mathrm{s} \\
\boldsymbol{\lambda}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{f}_{\mathrm{p}} \\
\mathbf{0}
\end{bmatrix},\quad \Rightarrow\quad
\mathcal{S}_\mathrm{s}(\boldsymbol{\theta})
\begin{bmatrix}
\mathbf{u}'_{\mathrm{s},j} \\
\boldsymbol{\lambda}'_j
\end{bmatrix}
=-\,\mathcal{S}'_{\mathrm{s},j}(\boldsymbol{\theta})
\begin{bmatrix}
\mathbf{u}_\mathrm{s} \\
\boldsymbol{\lambda}
\end{bmatrix},\label{eq:aug_sensitivity}
\end{equation}
where $\mathcal{S}'_{\mathrm{s},j}(\boldsymbol{\theta})=\begin{bmatrix} \mathbf{0} & (\partial\mathbf{C}/\partial\theta_j)^{\!\top} \\ \partial\mathbf{C}/\partial\theta_j & \mathbf{0}\end{bmatrix}$ comes solely from $\mathbf{C}(\boldsymbol{\theta})$, gives the reaction sensitivity $\boldsymbol{\lambda}'_j=\partial\mathbf{R}/\partial\theta_j$. Since $\mathbf{f}(\boldsymbol{\theta})=-\mathbf{R}(\boldsymbol{\theta})$, the load Jacobian follows as $\partial\mathbf{f}/\partial\theta_j=-\boldsymbol{\lambda}'_j$. In practice, $\tau$ is chosen small enough to localize the mapping (approaching hard assignment as $\tau\to 0$) while remaining large enough to keep gradients stable for optimization.
The shell pressure $p(y)$ is applied as edge line loads along the outer boundary of the shell mesh: for each boundary edge $e$ with midpoint vertical coordinate $y_m$ and length $\ell_e$, the line load magnitude is integrated by the midpoint rule as $q_e = p(y_m)\, t\, \ell_e$ (with shell thickness $t$) acting in the inward radial normal direction; these contributions assemble $\mathbf{f}_{\mathrm{p}}$ on boundary vertices.

The resulting load vector applied to the truss is the negative of the shell reactions,
\begin{equation}
\mathbf{f}(\boldsymbol{\theta}) = -\,\mathbf{R}(\boldsymbol{\theta}) \label{eq:truss_loads}
\end{equation}
so that each node receives the equal-and-opposite forces developed by the shell at the corresponding support.

This shows that the load vector $\mathbf{f}(\boldsymbol{\theta})$ explicitly depends on the support configuration through the shell structural response, creating the design-dependent loading condition. The support reactions account for the complex load redistribution within the shell structure and cannot be simplified to direct pressure application.




\subsection{Full problem}
The full problem formulation for joint topology–geometry optimization under hydrostatic design-dependent loading can now be written as:
\begin{align}
\min_{\mathbf{A}, \boldsymbol{\theta}} \quad & \mathbf{f}(\boldsymbol{\theta})^T \mathbf{u}(\mathbf{A}, \boldsymbol{\theta}) \label{eq:original_problem} \\
\text{subject to} \quad & \mathbf{K}(\mathbf{A}, \boldsymbol{\theta}) \mathbf{u} = \mathbf{f}(\boldsymbol{\theta}) \label{eq:equilibrium} \\
& \sum_{i=1}^{m} A_i L_i(\boldsymbol{\theta}) \leq V_{\max} \label{eq:volume_constraint} \\
& A_{\min} \leq A_i \leq A_{\max}, \quad i = 1, \ldots, m \label{eq:area_bounds} \\
& \boldsymbol{\theta} \in \Theta \label{eq:geometry_feasible}
\end{align}

We assemble the truss stiffness as a geometry-dependent linear combination of per-element unit stiffness contributions:
\begin{equation}
\mathbf{K}(\mathbf{A}, \boldsymbol{\theta}) = \sum_{i=1}^{m} A_i\, \widehat{\mathbf{K}}_i(\boldsymbol{\theta}),
\label{K_decomposition}
\end{equation}
where $\widehat{\mathbf{K}}_i(\boldsymbol{\theta})$ denotes the global matrix contribution of element $i$ computed with unit area (a unit-stiffness depending on element orientation and length induced by $\boldsymbol{\theta}$). This clarifies that areas enter linearly while geometry affects the element directions and lengths.

Following the Schur complement approach, this can be formally written as:

\begin{align}
\min_{\mathbf{A}, \boldsymbol{\theta}, t} \quad & t \label{eq:design_dependent_sdp} \\
\text{subject to} \quad & \begin{bmatrix} t & \mathbf{f}(\boldsymbol{\theta})^T \\ \mathbf{f}(\boldsymbol{\theta}) & \mathbf{K}(\mathbf{A}, \boldsymbol{\theta}) \end{bmatrix} \succeq \mathbf{0} \label{eq:nonconvex_schur} \\
& \sum_{i=1}^{m} A_i L_i(\boldsymbol{\theta}) \leq V_{\max} \label{eq:design_volume} \\
& A_{\min} \leq A_i \leq A_{\max}, \quad i = 1, \ldots, m \label{eq:design_bounds} \\
& \boldsymbol{\theta} \in \Theta \label{eq:design_geometry}
\end{align}


This multiplicative coupling between cross-sectional areas and nodal positions, together with the design-dependent loading, yields several sources of nonconvexity:

\begin{enumerate}
\item Area–geometry multiplicative coupling: $\mathbf{K}(\mathbf{A},\boldsymbol{\theta})=\sum_{i=1}^m A_i\,\widehat{\mathbf{K}}_i(\boldsymbol{\theta})$. For fixed $\boldsymbol{\theta}$ it is affine in $\mathbf{A}$, whereas $\widehat{\mathbf{K}}_i(\boldsymbol{\theta})$ is nonlinear in $\boldsymbol{\theta}$; jointly this induces a nonconvex dependence on $(\mathbf{A},\boldsymbol{\theta})$.
\item Geometry dependence (nonlinear in $\boldsymbol{\theta}$): element lengths $L_i(\boldsymbol{\theta})$ and direction cosines enter $\widehat{\mathbf{K}}_i(\boldsymbol{\theta})$ nonlinearly, which makes the state $\mathbf{u}(\mathbf{A},\boldsymbol{\theta})$ and the compliance nonconvex in $\boldsymbol{\theta}$ even for fixed $\mathbf{A}$.
\item Design-dependent loading: $\mathbf{f}(\boldsymbol{\theta})$ varies with the support configuration through the shell response, introducing additional nonaffine, nonconvex dependence on $\boldsymbol{\theta}$ and complicating sensitivities.
\item Nonaffine Schur complement constraint: the LMI $\begin{bmatrix} t & \mathbf{f}(\boldsymbol{\theta})^T \\ \mathbf{f}(\boldsymbol{\theta}) & \mathbf{K}(\mathbf{A},\boldsymbol{\theta}) \end{bmatrix}\succeq \mathbf{0}$ is convex only when the matrix depends affinely on the decision variables; here the nonaffine dependence through $\mathbf{K}(\mathbf{A},\boldsymbol{\theta})$ and $\mathbf{f}(\boldsymbol{\theta})$ renders the feasible set nonconvex.
\end{enumerate}

Given these complexities, we employ a Sequential Convex Programming approach developed in Section 3.


\section{Sequential Convex Programming Framework}

In this section, we present an overview of the Sequential Convex Programming framework for joint topology-geometry optimization under design-dependent hydrostatic loading. We adopt the core SCP principle of iteratively solving convex approximations to handle nonconvex problems \cite{ni2005sequential}, but develop a fundamentally different implementation tailored to our specific problem. our approach uses Taylor linearization with trust-region control, formulating each subproblem as a semidefinite programming problem to leverage the inherent structure of truss optimization.This design enables direct utilization of robust SDP solver MOSEK, provides better numerical stability for handling bilinear couplings, and offers a more straightforward implementation pathway for practical engineering applications.

\subsection{Sensitivity Analysis and Gradient Computation}

The SCP framework requires gradient information to construct linearized subproblems. For joint optimization of cross-sectional areas $\mathbf{A}$ and nodal angular positions $\boldsymbol{\theta}$, we use the decomposition as Eq.~\eqref{K_decomposition}.

In the design-dependent loading setting considered here, the load vector $\mathbf{f}(\boldsymbol{\theta})$ depends on geometry but not on areas. Therefore, the area sensitivity retains the standard form:
\begin{equation}
\frac{\partial C}{\partial A_i} = -\,\mathbf{u}^T \, \widehat{\mathbf{K}}_i(\boldsymbol{\theta}) \, \mathbf{u} .
\label{eq:area_grad}
\end{equation}

When $\mathbf{f}$ depends on geometry (design-dependent loading), the total derivative with respect to $\theta_j$ includes both load and stiffness terms:
\begin{equation}
\frac{\partial C}{\partial \theta_j} = 2\,\Big(\frac{\partial \mathbf{f}}{\partial \theta_j}\Big)^{\!T} \mathbf{u}\; -\; \mathbf{u}^T\, \frac{\partial \mathbf{K}}{\partial \theta_j} \, \mathbf{u},
\quad \text{with} \quad
\frac{\partial \mathbf{K}}{\partial \theta_j} = \sum_{i=1}^{m} A_i\, \frac{\partial \, \widehat{\mathbf{K}}_i}{\partial \theta_j} .
\label{eq:total_gradient}
\end{equation}
For the shell–truss coupling used here, $\mathbf{f}(\boldsymbol{\theta}) = -\mathbf{R}(\boldsymbol{\theta})$ where $\mathbf{R}$ are shell support reactions. Differentiating the augmented shell system (Eq.~\eqref{eq:aug_sensitivity}) gives $\boldsymbol{\lambda}'_j = \partial \mathbf{R}/\partial \theta_j$, hence
\begin{equation}
\frac{\partial \mathbf{f}}{\partial \theta_j} = -\,\boldsymbol{\lambda}'_j \quad .
\label{load_gradient}
\end{equation}

\subsection{Joint Linearization Framework}

At each iteration $k$, we linearize both the stiffness matrix and load vector around the current design point $(\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)})$ and solve the resulting convex subproblem.

\textbf{Linearization of System Matrices:}

The stiffness matrix is linearized with respect to both $\mathbf{A}$ and $\boldsymbol{\theta}$ using a first-order expansion consistent with the above definition:
\begin{align}
\mathbf{K}_{\text{lin}}(\mathbf{A}, \boldsymbol{\theta}) &= \mathbf{K}(\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)}) + \sum_{i=1}^{m} (A_i - A_i^{(k)}) \, \widehat{\mathbf{K}}_i(\boldsymbol{\theta}^{(k)}) \nonumber \\
&\quad + \sum_{j=1}^{n} (\theta_j - \theta_j^{(k)}) \sum_{i=1}^{m} A_i^{(k)} \, \frac{\partial \, \widehat{\mathbf{K}}_i}{\partial \theta_j}\bigg|_{\boldsymbol{\theta}^{(k)}} \label{eq:joint_stiffness_linearization}
\end{align}

The design-dependent load vector is linearized as:
\begin{align}
\mathbf{f}_{\text{lin}}(\boldsymbol{\theta}) &= \mathbf{f}(\boldsymbol{\theta}^{(k)}) + \sum_{j=1}^{n} \Big(\frac{\partial \mathbf{f}}{\partial \theta_j}\Big)\Big|_{\boldsymbol{\theta}^{(k)}} (\theta_j - \theta_j^{(k)}) \label{eq:joint_load_linearization}
\end{align}

Here $\partial \mathbf{f}/\partial \theta_j$ is obtained from the shell–support sensitivity via $\partial\mathbf{f}/\partial\theta_j = -\boldsymbol{\lambda}'_j$.

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

At iteration $k$, each angular variable is limited by a per-node step cap $\kappa_j>0$ computed from the current geometry:
\begin{equation}
|\theta_j - \theta_j^{(k)}| \le \kappa_j, \quad j=1,\ldots,n.
\label{eq:per_node_caps}
\end{equation}
The constraints \eqref{eq:monotonicity}–\eqref{eq:symmetry} and \eqref{eq:per_node_caps} prevent nodes collisions and keep all member lengths strictly positive, thereby avoiding singular or non-differentiable configurations while remaining consistent with our implementation.

A concrete choice for the per-node caps is
\begin{equation}
\kappa_j 
= \max\bigl(\varepsilon_{\text{cap}},\; \min\bigl(\kappa_{\max},\; k\, \tfrac{L_{\min}(j)}{r_j}\bigr)\bigr),
\label{eq:cap_definition}
\end{equation}
where $L_{\min}(j)$ is the shortest length among members incident to joint $j$ at the current design $(\mathbf{A}^{(k)},\boldsymbol{\theta}^{(k)})$, $r_j$ is the current radial distance of joint $j$, $k=\tfrac{1}{2}$, $\kappa_{\max}$ is a global angular cap, and $\varepsilon_{\text{cap}}$ is a small positive floor. This scaling converts an admissible linear displacement (half of the shortest incident member length) into an angular bound via $\Delta s \approx r_j\, \Delta\theta$.

Trust region (TR) method is a gradient-based algorithm which introduces a criteria to judge the quality of each step \cite{hu2022adaptive}. The trust region algorithm limits the step size within a radius $\Delta^{(k)}$ around the current design point:
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


\subsection{Node Merging Strategy}

To maintain numerical stability and produce cleaner, manufacturable topologies during joint topology--geometry optimization, we apply a node merging strategy in SCP iterations similar with the former studies on truss geometry optimization\cite{weldeyesus2020truss,he2015rationalization}.

When two nodes become nearly coincident, or when a member becomes extremely short, the global stiffness matrix can become ill-conditioned and the linearization inaccurate. Merging such nodes (i) removes nearly zero-length members and redundant joints, (ii) prevents singular/ill-conditioned configurations, and (iii) reduces graph complexity while preserving structural intent.

We build proximity clusters among free nodes using a user-configurable tolerance $\varepsilon_{\text{merge}}$ and/or a minimum admissible member length $L_{\min}^{\text{merge}}$:
\begin{equation}
\label{eq:merge_criterion}
\lVert \mathbf{x}_i-\mathbf{x}_j \rVert \;\le\; \varepsilon_{\text{merge}}
\quad \text{or} \quad
\exists\, e=(i,j):\; L_e < L_{\min}^{\text{merge}}.
\end{equation}
Proximity is evaluated on the current geometry $\mathbf{x}(\boldsymbol{\theta})$. The tolerances are chosen small relative to the structure scale (e.g., a fraction of the local radius or typical member length). Transitive merges are handled via clustering (e.g., a union--find data structure).

For each cluster $\mathcal{C}$, we create a representative node $\hat i$:
\begin{itemize}
  \item if $\mathcal{C}$ contains any support node, reuse that node's coordinates as the representative;
  \item otherwise, place the representative at the arithmetic mean of the clustered coordinates.
\end{itemize}
This preserves boundary conditions and load mapping.

All members incident to nodes in $\mathcal{C}$ are rewired to the representative node. Edges that become self-loops are removed. Parallel members that end with the same pair of representatives are aggregated by summing their cross-sectional areas, with optional capping by $A_{\max}$ to preserve bounds:
\begin{equation}
\label{eq:area_aggregation}
\tilde A_{\hat i \hat j} = \min\!\Big( A_{\max},\; \sum_{e\in E_{\hat i \hat j}} A_e \Big),
\end{equation}
where $E_{\hat i \hat j} := \{\, e=(i,j)\in E \mid \pi(i)=\hat i,\; \pi(j)=\hat j \,\}$ collects all original members whose endpoints map to the representative pair $(\hat i,\hat j)$ under the node-merging map $\pi$.
This keeps global volume consistent while simplifying the graph.

After merging, we update: (i) the ordered angle list $\boldsymbol{\theta}$ (drop merged entries; keep monotonicity and minimum angular spacing), (ii) symmetry maps (if enforced), (iii) per-node step caps $\kappa_j$ used by the trust region (recomputed from the shortest incident member length), and (iv) cached unit-stiffness contributions $\widehat{\mathbf{K}}_i(\boldsymbol{\theta})$ at the current geometry.

Node merging is applied after an accepted SCP step and before assembling the next subproblem. This ensures the linearized model is built on a well-conditioned geometry and that extremely short members do not persist across iterations.

\subsection{Algorithm}

An overview of the SCP algorithm is summarized in Algorithm~1. The method integrates the linearization framework developed in Section~3.2 with the trust-region management strategy in Section~3.3. Each iteration performs: (i) joint linearization of the stiffness matrix and the design-dependent load vector around the current design; (ii) solution of the resulting SDP subproblem; and (iii) trust-region update based on the step-quality ratio. The algorithm simultaneously optimizes cross-sectional areas $\mathbf{A}$ and nodal positions $\boldsymbol{\theta}$, with the trust-region constraint $\|\boldsymbol{\theta} - \boldsymbol{\theta}^{(k)}\|_2 \leq \Delta^{(k)}$ controlling model fidelity. convergence is declared when the relative change in compliance across accepted iterations falls below a prescribed tolerance while constraints remain satisfied.

\begin{algorithm}
\caption{Joint Sequential Convex Programming for Truss Optimization}
\begin{algorithmic}[1]
\STATE Initialize $\boldsymbol{\theta}^{(0)}$, $\mathbf{A}^{(0)}$, $\Delta^{(0)}$
\STATE Set iteration counter $k \leftarrow 0$
\WHILE{stopping criteria is not satisfied}
    \STATE \textbf{Joint Linearization:}
    \STATE Compute gradients $\frac{\partial \mathbf{K}}{\partial \theta_j}|_{(\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)})}$ and $\frac{\partial \mathbf{f}}{\partial \theta_j}|_{\boldsymbol{\theta}^{(k)}}$ using equation and \eqref{eq:total_gradient}
    \STATE Construct linearized system matrices $\mathbf{K}_{\text{lin}}(\mathbf{A}, \boldsymbol{\theta})$ and $\mathbf{f}_{\text{lin}}(\boldsymbol{\theta})$ using equations~(25) and~(26)
    \STATE \textbf{Joint SDP Solution:}
    \STATE Solve joint SDP subproblem \eqref{eq:joint_objective}--\eqref{eq:joint_trust_region} for $(\mathbf{A}^{(k+1)}, \boldsymbol{\theta}^{(k+1)})$
    \STATE \textbf{Trust Region Update:}
    \STATE Evaluate step quality $\rho^{(k)}$ using equation~\eqref{eq:trust_region_update}
    \IF{$\rho^{(k)} < \eta_1$}
        \STATE Reject step: $(\mathbf{A}^{(k+1)}, \boldsymbol{\theta}^{(k+1)}) = (\mathbf{A}^{(k)}, \boldsymbol{\theta}^{(k)})$
        \STATE Reduce trust region radius
    \ELSE
        \STATE Accept step and update trust region radius
        \STATE \textbf{Node merge and refresh:}
        \STATE Apply node\_merge on updated geometry; rewire connectivity, aggregate areas via Eq.~\eqref{eq:area_aggregation}
        \STATE Refresh caches (unit stiffness $\widehat{\mathbf{K}}_i$, step caps $\kappa_j$, symmetry maps)
    \ENDIF
    \STATE Check convergence based on compliance stabilization
    \STATE $k \leftarrow k + 1$
\ENDWHILE
\STATE \textbf{return} Optimal design $(\mathbf{A}^*, \boldsymbol{\theta}^*)$
\end{algorithmic}
\end{algorithm}

\section{Numerical Results}

The Sequential Convex Programming framework has been implemented in Python(3.12.7). The semidefinite programming subproblems within each SCP iteration are formulated using the CVXPY optimization modeling language and solved by MOSEK. All numerical experiments have been performed on a laptop equipped with an Intel(R) Core(TM) i7-9750H CPU running at 2.60 GHz with 32 GB RAM.

The truss material uses steel parameters with Young's modulus $E_{\text{truss}} = 2.10\times10^{8}\,\mathrm{Pa}$ and density $\rho_{\text{steel}} = 7850\,\mathrm{kg/m^3}$. The auxiliary shell model used to transfer hydrostatic loads is set with $E_{\text{shell}} = 2.10\times10^{11}\,\mathrm{Pa}$ and Poisson ratio $\nu = 0.3$ (thickness specified per case). For hydrostatic loading, we adopt seawater density $\rho_w = 1025\,\mathrm{kg/m^3}$ and gravitational acceleration $g = 9.81\,\mathrm{m/s^2}$. The reference base pressure at depth $H$ is $p_0 = \rho_w g H$, and the nodal loads applied to the truss are obtained from the coupled shell FEA by mapping support reactions to the truss load nodes. The structural depth $H$ is specified for each example case.

Trust region and stopping parameters are as follows. The trust‑region radius is initialized at $\Delta^{(0)} = \pi/180$ and bounded within $[\pi/720,\, \mathrm{deg}(2.0)]$. Step acceptance is governed by the step‑quality ratio $\rho$: we accept when $\rho > 0.01$ and expand the radius when $\rho > 0.75$; otherwise the step is rejected and the radius shrinks. Expansion and shrink factors are $1.5$ and $0.5$, respectively. Convergence is declared when the relative changes of geometry $\boldsymbol{\theta}$ and areas $\mathbf{A}$ between accepted iterations are both below $10^{-3}$. In addition, we terminate when the compliance variation over the last three accepted iterations falls below $0.1\%$. The maximum number of iterations is set to $30$.




\subsection{Fixed Geometry (SDP) vs Moving Nodes (SCP)}

In this section, we present a semicircular truss structure under hydrostatic loading  case to demonstrate the advantages of geometry optimization in dealing with  design-dependent pressure conditions. This is done by comparing solutions obtained with fixed geometry (SDP) and movable nodes (SCP). Moreover, we 
demonstrate the performance improvements in structural compliance achieved  through the proposed SCP framework.

To comprehensively evaluate the performance of SCP and SDP, we conduct comparisons across different discretization levels using 16, 12, and 8 sectors. This parametric study reveals how the benefits of geometry optimization vary with mesh density and helps identify scenarios where joint topology-geometry optimization provides the most significant improvements.


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
