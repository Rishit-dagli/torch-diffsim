Examples
========

This page provides detailed walkthroughs of the example scripts included with torch-diffsim. All examples can be found in the ``examples/`` directory of the repository.

Basic Differentiable Simulation
--------------------------------

**File**: `examples/demo_diff_simple.py <https://github.com/Rishit-dagli/torch-diffsim/blob/main/examples/demo_diff_simple.py>`_

This example demonstrates the fundamentals of differentiable simulation in torch-diffsim. It covers three key concepts:

1. Energy and Gradients
   
   Shows how gradients flow through elastic energy computation with respect to both material parameters and vertex positions.

   .. code-block:: python

      # Compute elastic energy
      F = mesh.compute_deformation_gradient(positions)
      energy_density = material.energy_density(F)
      total_energy = torch.sum(energy_density * mesh.rest_volume)
      
      # Backward pass
      total_energy.backward()
      
      # Gradients are now available
      print(f"∂E/∂E_young: {material.E.grad}")
      print(f"∂E/∂x norm: {positions.grad.norm()}")

2. Material Parameter Optimization
   
   Demonstrates a simple optimization loop to recover material properties from observed deformation:

   .. code-block:: python

      optimizer = torch.optim.Adam([learned_material.E], lr=5e3)
      
      for iteration in range(50):
          optimizer.zero_grad()
          simulator.reset()
          
          # Run simulation
          for _ in range(20):
              simulator.step()
          
          # Compute loss and optimize
          loss = torch.mean((simulator.positions - target) ** 2)
          loss.backward()
          optimizer.step()

3. Different Loss Functions
   
   Compares various loss functions for matching simulated and target positions:
   
   - Mean squared error (MSE)
   - L1 loss
   - Per-vertex distance metrics

**Running the example**:

.. code-block:: bash

   python examples/demo_diff_simple.py

**Expected output**: The script will print gradient information, optimization progress, and final material parameter estimates.

Material Parameter Identification
----------------------------------

**File**: `examples/demo_diff_material.py <https://github.com/Rishit-dagli/torch-diffsim/blob/main/examples/demo_diff_material.py>`_

This example tackles the inverse problem of identifying material properties (Young's modulus and Poisson's ratio) from observed deformation.

Problem Setup:

1. Create a ground truth simulation with known material parameters
2. Run the simulation to generate target observations
3. Initialize a learnable material with incorrect parameters
4. Optimize the parameters to match the target deformation

Key Features:

* Fixed boundary conditions: Bottom vertices are constrained to simulate a fixed support
* Multi-parameter optimization: Jointly optimizes both Young's modulus (:math:`E`) and Poisson's ratio (:math:`\nu`)
* Parameter clamping: Ensures physical validity by clamping to reasonable ranges
* Convergence visualization: Generates plots showing loss, parameter recovery, and displacement matching
* Convergence visualization: Generates plots showing loss, parameter recovery, and displacement matching

Code Snippet:

.. code-block:: python

   # Create learnable material
   learned_material = DifferentiableMaterial(
       initial_E, initial_nu, requires_grad=True
   )
   
   # Optimize both parameters
   optimizer = torch.optim.Adam(
       [learned_material.E, learned_material.nu], 
       lr=1e3
   )
   
   for iter in range(100):
       optimizer.zero_grad()
       learned_sim.reset()
       
       # Forward simulation
       for _ in range(30):
           learned_sim.step()
       
       # Match target
       loss = torch.mean((learned_sim.positions - target_positions) ** 2)
       loss.backward()
       optimizer.step()
       
       # Clamp to physical range
       with torch.no_grad():
           learned_material.E.clamp_(1e4, 1e7)
           learned_material.nu.clamp_(0.0, 0.49)

**Running the example**:

.. code-block:: bash

   python examples/demo_diff_material.py

**Output**: The script generates a figure ``material_identification.png`` showing:

* Loss convergence over iterations
* Young's modulus recovery trajectory
* Poisson's ratio recovery trajectory
* True vs. learned displacement comparison

Spatially Varying Material Optimization
----------------------------------------

**File**: `examples/demo_diff_spatial.py <https://github.com/Rishit-dagli/torch-diffsim/blob/main/examples/demo_diff_spatial.py>`_

This advanced example demonstrates optimization of spatially heterogeneous material properties. Instead of uniform material parameters, each element has its own Young's modulus value.

Problem Setup:

1. Create a target with spatially varying stiffness (e.g., stiffer on left, softer on right)
2. Initialize a learnable material with uniform (incorrect) stiffness
3. Optimize per-element stiffness to match target deformation
4. Apply spatial smoothness regularization

Key Features:

* Per-element parameters: Each tetrahedral element has independent material properties
* Spatial regularization: Tikhonov smoothness penalty prevents noisy solutions
* Correlation metrics: Tracks how well the learned distribution matches the target
* Visualization: Plots spatial stiffness distribution along coordinate axes

Code Snippet:

.. code-block:: python

   # Create spatially varying material
   learned_material = SpatiallyVaryingMaterial(
       mesh.num_elements,
       base_youngs=2e5,  # Uniform initial guess
       base_poisson=0.4,
   ).to(device)
   
   # Optimize with regularization
   optimizer = torch.optim.Adam([learned_material.log_E], lr=0.01)
   
   for iter in range(200):
       optimizer.zero_grad()
       
       # Position matching loss
       pos_loss = torch.mean((learned_sim.positions - target_positions) ** 2)
       
       # Spatial smoothness regularization
       smoothness_loss = 0.0
       for i in range(mesh.num_elements - 1):
           smoothness_loss += (
               learned_material.log_E[i] - learned_material.log_E[i + 1]
           ) ** 2
       smoothness_loss = 1e-3 * smoothness_loss / mesh.num_elements
       
       loss = pos_loss + smoothness_loss
       loss.backward()
       optimizer.step()

**Running the example**:

.. code-block:: bash

   python examples/demo_diff_spatial.py

**Output**: Generates ``spatial_material_optimization.png`` showing:

* Loss convergence
* Correlation between learned and target distributions
* Per-element stiffness scatter plot
* Spatial stiffness profile along X-axis

Bunny Demo (Visualization)
---------------------------

**File**: `examples/demo_bunny.py <https://github.com/Rishit-dagli/torch-diffsim/blob/main/examples/demo_bunny.py>`_

This example demonstrates standard (non-differentiable) simulation with 3D visualization using Polyscope.

Features:

* Loading a complex tetrahedral mesh (Stanford Bunny)
* Interactive visualization of deformation
* Collision detection with ground plane
* Real-time simulation display

Running the example:

.. code-block:: bash

   python examples/demo_bunny.py

Output: Opens an interactive Polyscope window where you can view the deformation of the bunny.

Next Steps
----------

* Explore the :doc:`api/diff_physics` documentation for advanced features
* Check out the :doc:`quickstart` guide for more basic examples
* Visit the `GitHub repository <https://github.com/Rishit-dagli/torch-diffsim>`_ for more examples and updates

