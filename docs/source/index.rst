⚙️ torch-diffsim
================

.. raw:: html

   <div style="text-align: center; margin-bottom: 1.5em;">
     <video controls autoplay loop muted playsinline style="width: 100%; max-width: 800px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
       <source src="_static/assets/diffsim_teaser.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>


**torch-diffsim** is a minimal differentiable physics simulator built entirely in PyTorch. It uses semi-implicit (symplectic Euler) time integration for tetrahedral finite element method (FEM) simulations with full automatic differentiation support.

Key Features
------------

* **Fully Differentiable.** Uses automatic differentiation with implicit differentiation, gradient checkpointing and memory efficient backpropagation
* **Fast.** Runs in 30 FPS on RTX 4090 for a single object
* **Semi-Implicit Integration.** Uses symplectic Euler time integration
* **Stable Neo-Hookean Material.** Uses the Stable Neo-Hookean hyperelastic material model
* **Barrier-Based Contact.** Uses differentiable collision handling using smooth barrier functions
* **GPU Accelerated.** Full CUDA support via PyTorch

Installation
------------

Install ``torch-diffsim`` using pip:

.. code-block:: bash

   pip install torch-diffsim

Or install from source:

.. code-block:: bash

   git clone https://github.com/Rishit-dagli/torch-diffsim
   cd torch-diffsim
   pip install -e .

Quick Example
-------------

Here's a simple example of running a differentiable simulation:

.. code-block:: python

   import torch
   from diffsim import TetrahedralMesh, StableNeoHookean, DifferentiableSolver, DifferentiableSimulator

   # Load a tetrahedral mesh
   mesh = TetrahedralMesh.load_from_msh("bunny.msh")
   
   # Create a material with learnable parameters
   material = StableNeoHookean(E=1e5, nu=0.45)
   
   # Set up the differentiable simulator
   solver = DifferentiableSolver(mesh, material, dt=0.01)
   simulator = DifferentiableSimulator(solver)
   
   # Run simulation with gradient tracking
   positions = mesh.vertices.clone().requires_grad_(True)
   final_positions = simulator.rollout(positions, steps=100)
   
   # Compute loss and backpropagate
   loss = ((final_positions - target_positions) ** 2).sum()
   loss.backward()

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   quickstart
   how_it_works
   examples
   citation

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/mesh
   api/material
   api/solver
   api/simulator
   api/diff_physics
   api/diff_simulator
   api/collision
   api/visualizer

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

