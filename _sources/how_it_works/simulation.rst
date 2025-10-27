Simulation
==========

This page describes the forward physics simulation in torch-diffsim, including the finite element method formulation, time integration scheme, and material model.

Finite Element Method (FEM)
----------------------------

torch-diffsim uses the **finite element method** with linear tetrahedral elements to discretize the continuous elastic body.

Mesh Discretization
~~~~~~~~~~~~~~~~~~~

The domain :math:`\Omega \subset \mathbb{R}^3` is discretized into :math:`M` tetrahedral elements:

.. math::

   \Omega \approx \bigcup_{e=1}^{M} \Omega_e

Each tetrahedron :math:`\Omega_e` is defined by 4 vertices with positions :math:`\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3 \in \mathbb{R}^3`.

Deformation Gradient
~~~~~~~~~~~~~~~~~~~~

The deformation gradient :math:`\mathbf{F}` maps from the reference (rest) configuration to the current (deformed) configuration. For each element, it is computed as:

.. math::

   \mathbf{F} = \mathbf{D}_s \mathbf{D}_m^{-1}

where:

- :math:`\mathbf{D}_m = [\mathbf{X}_1 - \mathbf{X}_0, \mathbf{X}_2 - \mathbf{X}_0, \mathbf{X}_3 - \mathbf{X}_0] \in \mathbb{R}^{3 \times 3}` contains edge vectors in the **rest configuration**
- :math:`\mathbf{D}_s = [\mathbf{x}_1 - \mathbf{x}_0, \mathbf{x}_2 - \mathbf{x}_0, \mathbf{x}_3 - \mathbf{x}_0] \in \mathbb{R}^{3 \times 3}` contains edge vectors in the **current configuration**

For **linear tetrahedral elements**, :math:`\mathbf{F}` is constant within each element (constant strain elements).

Stable Neo-Hookean Material
----------------------------

The material model defines the relationship between deformation and stress. We use the **Stable Neo-Hookean** hyperelastic model from Smith et al. (2018).

Energy Density
~~~~~~~~~~~~~~

The strain energy density :math:`\Psi: \mathbb{R}^{3 \times 3} \to \mathbb{R}` is:

.. math::

   \Psi(\mathbf{F}) = \frac{\mu}{2}(I_C - 3) - \mu \log J + \frac{\lambda}{2}(J - 1)^2

where:

- :math:`I_C = \text{tr}(\mathbf{F}^T \mathbf{F}) = \|\mathbf{F}\|_F^2` is the **first invariant** (measures stretch)
- :math:`J = \det(\mathbf{F})` is the **Jacobian determinant** (measures volume change)
- :math:`\mu = \frac{E}{2(1+\nu)}` is the **shear modulus** (first Lamé parameter)
- :math:`\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}` is the **bulk modulus** (second Lamé parameter)

**Why "Stable"?** The logarithmic term :math:`-\mu \log J` (instead of :math:`\lambda \log^2 J`) ensures the energy remains bounded even under large compression or inversion, providing numerical stability.

Total Elastic Energy
~~~~~~~~~~~~~~~~~~~~

The total elastic energy is obtained by integrating over all elements:

.. math::

   E_{\text{elastic}} = \sum_{e=1}^{M} \Psi(\mathbf{F}_e) V_e^0

where :math:`V_e^0 = \frac{1}{6}|\det(\mathbf{D}_m^e)|` is the rest volume of element :math:`e`.

Elastic Forces
~~~~~~~~~~~~~~

The elastic force on each vertex is the negative gradient of the total energy:

.. math::

   \mathbf{f}_{\text{elastic}} = -\frac{\partial E_{\text{elastic}}}{\partial \mathbf{x}}

This is computed using the **first Piola-Kirchhoff stress tensor** :math:`\mathbf{P}`:

.. math::

   \mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}} = \mu \mathbf{F} - \mu \mathbf{F}^{-T} + \lambda (J-1) J \mathbf{F}^{-T}

where :math:`\mathbf{F}^{-T} = (\mathbf{F}^{-1})^T` is the inverse transpose.

The force on vertex :math:`i` of element :math:`e` is:

.. math::

   \mathbf{f}_i^e = -V_e^0 \mathbf{P}_e \frac{\partial \mathbf{F}_e}{\partial \mathbf{x}_i}

Semi-Implicit (Symplectic Euler) Time Integration
--------------------------------------------------

Time integration advances the simulation forward in time. We use **semi-implicit Euler**, also known as **symplectic Euler**.

Integration Scheme
~~~~~~~~~~~~~~~~~~

Given state :math:`(\mathbf{x}^n, \mathbf{v}^n)` at time :math:`t_n`, compute the next state:

.. math::

   \mathbf{v}^{n+1} &= \mathbf{v}^n + \Delta t \, \mathbf{M}^{-1} \mathbf{f}(\mathbf{x}^n)

   \mathbf{x}^{n+1} &= \mathbf{x}^n + \Delta t \, \mathbf{v}^{n+1}

**Key property**: Velocities are updated using forces at the **current position** :math:`\mathbf{x}^n`, then positions are updated using the **new velocities** :math:`\mathbf{v}^{n+1}`.

Why Semi-Implicit?
~~~~~~~~~~~~~~~~~~

This scheme is:

1. **Symplectic**: Preserves phase space volume, leading to better energy conservation
2. **First-order accurate**: Error is :math:`O(\Delta t)`
3. **Conditionally stable**: More stable than explicit Euler for stiff systems
4. **Simple**: No iterative solver needed (unlike fully implicit methods)

The scheme is called "semi-implicit" because velocities use the current position (explicit) but positions use the new velocity (implicit dependency).

Total Force Computation
~~~~~~~~~~~~~~~~~~~~~~~

The total force includes multiple components:

.. math::

   \mathbf{f}(\mathbf{x}) = \mathbf{f}_{\text{elastic}}(\mathbf{x}) + \mathbf{f}_{\text{gravity}} + \mathbf{f}_{\text{contact}}(\mathbf{x}) + \mathbf{f}_{\text{damping}}(\mathbf{v})

**Elastic forces**: Computed from strain energy as described above

**Gravity**: :math:`\mathbf{f}_{\text{gravity}} = \mathbf{M} \mathbf{g}` where :math:`\mathbf{g} = [0, -9.8, 0]^T`

**Contact forces**: Either smooth barrier forces (differentiable solver) or projection with friction (standard solver)

**Damping**: :math:`\mathbf{v} \leftarrow \alpha \mathbf{v}` with :math:`\alpha \approx 0.99`

Substepping
~~~~~~~~~~~

For stability, each timestep :math:`\Delta t` is subdivided into :math:`n_{\text{sub}}` substeps:

.. math::

   h = \frac{\Delta t}{n_{\text{sub}}}

The integration scheme is applied :math:`n_{\text{sub}}` times with step size :math:`h`. Typical values: :math:`\Delta t = 0.01`, :math:`n_{\text{sub}} = 4`.

Contact Handling
----------------

We use two approaches depending on the solver:

- **Differentiable solver**: Smooth barrier functions to maintain differentiability
- **Standard solver**: Ground-plane projection with restitution and simple Coulomb-like friction

Barrier Potential
~~~~~~~~~~~~~~~~~

For ground contact at :math:`y = 0`, the barrier potential for vertex :math:`i` is:

.. math::

   b(d_i) = \begin{cases}
   -\kappa (d_i - \hat{d})^2 \log(d_i / \hat{d}) & \text{if } d_i < \hat{d} \\
   0 & \text{otherwise}
   \end{cases}

where:

- :math:`d_i = y_i` is the distance to the ground
- :math:`\hat{d}` is the barrier activation distance (e.g., 0.01 m)
- :math:`\kappa` is the barrier stiffness (e.g., :math:`10^4`)

Contact Force (Differentiable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contact force is the negative gradient of the barrier potential:

.. math::

   \mathbf{f}_{\text{contact}} = -\nabla_{\mathbf{x}} \sum_{i=1}^{N} b(d_i)

This creates a smooth repulsive force that increases as the vertex approaches the ground, preventing penetration while maintaining :math:`C^2` continuity.

**Why smooth barriers?** Hard constraints (projections) introduce discontinuities that break gradient flow. Smooth barriers maintain differentiability while still preventing penetration.

Ground Projection (Standard)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the non-differentiable solver, vertices below the ground are projected back to :math:`y=0`, normal velocity is adjusted by restitution, and tangential velocity is damped to emulate friction. See ``SemiImplicitSolver._handle_ground_collision`` in the code.

Mass Matrix
-----------

The mass matrix :math:`\mathbf{M}` is diagonal (lumped mass):

.. math::

   M_{ii} = \sum_{e \in \text{adj}(i)} \frac{\rho V_e^0}{4}

where :math:`\rho` is the material density and the sum is over all elements adjacent to vertex :math:`i`. Each element's mass is distributed equally to its 4 vertices.

Boundary Conditions
-------------------

**Fixed vertices**: Vertices can be constrained by setting their velocity to zero:

.. math::

   \mathbf{v}_i^{n+1} = \mathbf{0} \quad \text{for } i \in \mathcal{F}

where :math:`\mathcal{F}` is the set of fixed vertex indices.

Implementation Notes
--------------------

In the code:

1. :math:`\mathbf{D}_m^{-1}` is precomputed and cached for efficiency
2. Forces are accumulated using ``index_add_`` for parallelism
3. All operations use PyTorch tensors for GPU acceleration
4. Clamping is applied to :math:`J` to prevent numerical issues: :math:`J \leftarrow \text{clamp}(J, J_{\min}, J_{\max})`
5. The standard solver optionally adds simplified self-collision forces; the differentiable solver includes smooth ground contact forces.
