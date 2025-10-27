How does torch-diffsim work?
=============================

This section provides a description of how torch-diffsim implements differentiable physics simulation. We cover both the forward simulation (physics) and the backward differentiation (gradients).

Overview
--------

torch-diffsim combines two key components:

1. **Physics Simulation**: A semi-implicit (symplectic Euler) time integrator for tetrahedral finite element method (FEM) using the Stable Neo-Hookean hyperelastic material model
2. **Automatic Differentiation**: Gradient computation through the simulation using PyTorch's autograd, enabling gradient-based optimization of material properties, initial conditions, and other parameters

.. toctree::
   :maxdepth: 2
   
   how_it_works/simulation
   how_it_works/differentiation

Core Principles
---------------

* Symplectic Integration

The simulator uses a first-order symplectic method (semi-implicit Euler) that provides better energy conservation than explicit methods.

* Smooth Operations

All operations are smooth (differentiable) - no hard constraints, projections, or conditional branching that would break gradient flow.

* Energy-Based Formulation

Forces are derived as negative gradients of the total elastic energy, which naturally integrates with PyTorch's autograd system.

* Barrier Functions

Contact handling uses smooth barrier potentials instead of hard constraints, maintaining :math:`C^2` continuity for gradient computation.

Mathematical Notation
---------------------

Throughout this documentation, we use the following notation:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\mathbf{x}^n`
     - Vertex positions at time step :math:`n`, :math:`(N \times 3)` tensor
   * - :math:`\mathbf{v}^n`
     - Vertex velocities at time step :math:`n`, :math:`(N \times 3)` tensor
   * - :math:`\mathbf{M}`
     - Mass matrix (diagonal), :math:`(N \times N)` matrix
   * - :math:`\mathbf{f}`
     - Force vector, :math:`(N \times 3)` tensor
   * - :math:`\mathbf{F}`
     - Deformation gradient, :math:`(M \times 3 \times 3)` tensor
   * - :math:`\Psi(\mathbf{F})`
     - Strain energy density function
   * - :math:`\Delta t`
     - Time step size
   * - :math:`N`
     - Number of vertices
   * - :math:`M`
     - Number of tetrahedral elements
   * - :math:`E`
     - Young's modulus (material parameter)
   * - :math:`\nu`
     - Poisson's ratio (material parameter)
