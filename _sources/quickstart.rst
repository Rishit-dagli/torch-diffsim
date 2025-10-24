Quickstart Guide
================

This guide will help you get started with torch-diffsim by walking through the basic concepts and providing simple examples.

Basic Concepts
--------------

torch-diffsim is built around a few core concepts:

* **Mesh**: Represents the geometry of your simulated object using tetrahedral elements
* **Material**: Defines the material properties (e.g., Young's modulus :math:`E`, Poisson's ratio :math:`\nu`)
* **Solver**: Handles the semi-implicit time integration and physics computation
* **Simulator**: Coordinates the simulation loop and maintains state

Background
~~~~~~~~~~

**Time Integration**: torch-diffsim uses a semi-implicit (symplectic Euler) integrator:

.. math::

   \mathbf{v}^{n+1} &= \mathbf{v}^n + \Delta t \, \mathbf{M}^{-1} \mathbf{f}(\mathbf{x}^n) \\
   \mathbf{x}^{n+1} &= \mathbf{x}^n + \Delta t \, \mathbf{v}^{n+1}

**Elastic Forces**: Forces are derived from the strain energy density :math:`\Psi(\mathbf{F})`:

.. math::

   \mathbf{f} = -\frac{\partial}{\partial \mathbf{x}} \int_\Omega \Psi(\mathbf{F}(\mathbf{x})) \, dV

where :math:`\mathbf{F} = \frac{\partial \mathbf{x}}{\partial \mathbf{X}}` is the deformation gradient mapping from rest space to current space.

**Stable Neo-Hookean Model**: The energy density is given by:

.. math::

   \Psi(\mathbf{F}) = \frac{\mu}{2}(I_C - 3) - \mu \log J + \frac{\lambda}{2}(J-1)^2

where:

* :math:`I_C = \text{tr}(\mathbf{F}^T\mathbf{F}) = \|\mathbf{F}\|_F^2` is the first invariant
* :math:`J = \det(\mathbf{F})` is the volume ratio (Jacobian determinant)
* :math:`\mu = \frac{E}{2(1+\nu)}` is the first Lamé parameter (shear modulus)
* :math:`\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}` is the second Lamé parameter

This formulation ensures stability even under large deformations and element inversion.

Standard Simulation
-------------------

Creating a Simple Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a minimal example to run a standard (non-differentiable) simulation:

.. code-block:: python

   from diffsim import TetrahedralMesh, StableNeoHookean, SemiImplicitSolver, Simulator

   # Create or load a mesh
   mesh = TetrahedralMesh.create_cube(resolution=5, size=1.0)
   
   # Or load from a file
   # mesh = TetrahedralMesh.load_from_msh("path/to/mesh.msh")
   
   # Define material properties
   material = StableNeoHookean(E=1e5, nu=0.45)  # E: Young's modulus, nu: Poisson's ratio
   
   # Create solver and simulator
   solver = SemiImplicitSolver(mesh, material, dt=0.01)
   simulator = Simulator(solver)
   
   # Run simulation steps
   for i in range(100):
       simulator.step()
       if i % 10 == 0:
           print(f"Step {i}: Energy = {simulator.compute_energy():.4f}")

Loading Custom Meshes
~~~~~~~~~~~~~~~~~~~~~~

You can load tetrahedral meshes in `.msh` format (Gmsh):

.. code-block:: python

   mesh = TetrahedralMesh.load_from_msh("assets/tetmesh/bunny0.msh")
   print(f"Vertices: {mesh.num_vertices}, Tetrahedra: {mesh.num_tetrahedra}")

Visualization
~~~~~~~~~~~~~

torch-diffsim includes built-in visualization using Polyscope:

.. code-block:: python

   from diffsim.visualizer import SimulationVisualizer

   # Create visualizer
   viz = SimulationVisualizer(mesh)
   
   # Run simulation with visualization
   for i in range(200):
       simulator.step()
       if i % 5 == 0:
           viz.update(simulator.solver.positions.cpu().numpy())
   
   # Show the visualization window
   viz.show()

Differentiable Simulation
--------------------------

The key feature of torch-diffsim is its support for differentiable simulation, enabling gradient-based optimization.

Basic Differentiable Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from diffsim import TetrahedralMesh
   from diffsim.diff_physics import DifferentiableMaterial
   from diffsim.diff_simulator import DifferentiableSolver, DifferentiableSimulator

   device = "cuda" if torch.cuda.is_available() else "cpu"
   
   # Create mesh
   mesh = TetrahedralMesh.create_cube(resolution=3, size=0.5, device=device)
   mesh._compute_rest_state()
   
   # Create learnable material
   material = DifferentiableMaterial(E=1e5, nu=0.4, requires_grad=True).to(device)
   
   # Create differentiable solver and simulator
   solver = DifferentiableSolver(dt=0.01, gravity=-9.8, damping=0.98)
   simulator = DifferentiableSimulator(mesh, material, solver, device=device)
   
   # Run simulation
   for _ in range(20):
       simulator.step()
   
   # Compute loss (e.g., match target position)
   target_position = torch.tensor([0.0, -0.5, 0.0], device=device)
   center_of_mass = simulator.positions.mean(dim=0)
   loss = ((center_of_mass - target_position) ** 2).sum()
   
   # Backpropagate
   loss.backward()
   print(f"Gradient w.r.t. Young's modulus: {material.E.grad}")

Material Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common use case is optimizing material properties to match observed behavior:

.. code-block:: python

   import torch
   from diffsim import TetrahedralMesh
   from diffsim.diff_physics import DifferentiableMaterial
   from diffsim.diff_simulator import DifferentiableSolver, DifferentiableSimulator

   device = "cuda" if torch.cuda.is_available() else "cpu"
   
   # Create mesh
   mesh = TetrahedralMesh.create_cube(resolution=4, size=0.5, device=device)
   mesh._compute_rest_state()
   
   # Create target data (simulate with known material)
   true_material = DifferentiableMaterial(8e4, 0.4, requires_grad=False).to(device)
   true_solver = DifferentiableSolver(dt=0.01, gravity=-9.8, damping=0.98, substeps=2)
   true_sim = DifferentiableSimulator(mesh, true_material, true_solver, device=device)
   
   for _ in range(20):
       true_sim.step()
   target = true_sim.positions.detach()
   
   # Initialize learnable material with wrong guess
   material = DifferentiableMaterial(2e5, 0.4, requires_grad=True).to(device)
   solver = DifferentiableSolver(dt=0.01, gravity=-9.8, damping=0.98, substeps=2)
   simulator = DifferentiableSimulator(mesh, material, solver, device=device)
   
   # Optimize
   optimizer = torch.optim.Adam([material.E], lr=5e3)
   
   for iteration in range(50):
       optimizer.zero_grad()
       simulator.reset()
       
       # Forward simulation
       for _ in range(20):
           simulator.step()
       
       # Compute loss
       loss = torch.mean((simulator.positions - target) ** 2)
       
       # Backward pass
       loss.backward()
       optimizer.step()
       
       # Clamp values to physical range
       with torch.no_grad():
           material.E.clamp_(1e4, 5e5)
       
       if iteration % 10 == 0:
           print(f"Iter {iteration}: Loss = {loss.item():.6f}, E = {material.E.item():.2e}")

Spatially Varying Materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also optimize materials that vary across the mesh:

.. code-block:: python

   from diffsim.diff_physics import SpatiallyVaryingMaterial

   # Create spatially varying material
   # Grid resolution: number of control points in each dimension
   spatial_material = SpatiallyVaryingMaterial(
       base_E=1e5,
       base_nu=0.4,
       grid_resolution=(4, 4, 4),
       E_range=(5e4, 2e5),
       requires_grad=True
   ).to(device)
   
   # The material parameters will be interpolated based on position
   simulator = DifferentiableSimulator(mesh, spatial_material, solver, device=device)
   
   # Optimize the spatial distribution
   optimizer = torch.optim.Adam(spatial_material.parameters(), lr=1e3)
   # ... run optimization loop

Memory-Efficient Rollouts
~~~~~~~~~~~~~~~~~~~~~~~~~~

For long simulations, use checkpointed rollouts to save memory:

.. code-block:: python

   from diffsim.diff_physics import CheckpointedRollout

   # Create checkpointed rollout function
   rollout = CheckpointedRollout(simulator, checkpoint_every=10)
   
   # Run for many steps with gradient tracking
   final_state = rollout(num_steps=1000)
   
   # Compute loss and backpropagate
   loss = compute_loss(final_state)
   loss.backward()  # Uses checkpointing to save memory

Contact and Collision
---------------------

torch-diffsim includes barrier-based collision handling:

.. code-block:: python

   from diffsim.diff_physics import DifferentiableBarrierContact

   # Create contact handler
   contact = DifferentiableBarrierContact(
       ground_height=-1.0,
       barrier_stiffness=1e6,
       barrier_distance=0.01
   )
   
   # Apply contact forces during simulation
   positions = simulator.positions
   velocities = simulator.velocities
   
   contact_forces = contact.compute_forces(positions, velocities)
   # These forces are automatically included in the differentiable simulator

Next Steps
----------

* Check out the :doc:`examples` page for more complete examples
* Explore the :doc:`api/diff_physics` API documentation for advanced features
* Look at the example scripts in the ``examples/`` directory of the repository
