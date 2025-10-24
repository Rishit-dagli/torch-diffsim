Differentiation
===============

This page describes how gradients are computed through the simulation, enabling gradient-based optimization of material properties, initial conditions, and other parameters.

Automatic Differentiation Overview
-----------------------------------

torch-diffsim uses **automatic differentiation (autodiff)** via PyTorch's autograd system to compute gradients of outputs with respect to inputs through the simulation.

Problem Setup
~~~~~~~~~~~~~

Given:

- Parameters :math:`\boldsymbol{\theta}` (e.g., material properties :math:`E`, :math:`\nu`)
- Initial state :math:`\mathbf{x}_0, \mathbf{v}_0`
- Loss function :math:`\mathcal{L}` that depends on final or intermediate states

Goal: Compute :math:`\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}`

This enables gradient-based optimization:

.. math::

   \boldsymbol{\theta}^{k+1} = \boldsymbol{\theta}^k - \alpha \frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}

where :math:`\alpha` is the learning rate.

Differentiable Simulation
--------------------------

The simulation can be viewed as a computational graph:

.. math::

   \mathbf{x}_0, \mathbf{v}_0, \boldsymbol{\theta} \xrightarrow{\text{step 1}} \mathbf{x}_1, \mathbf{v}_1 \xrightarrow{\text{step 2}} \cdots \xrightarrow{\text{step } T} \mathbf{x}_T, \mathbf{v}_T \xrightarrow{\mathcal{L}} \text{loss}

For differentiation to work, every operation must:

1. Be differentiable (smooth)
2. Maintain gradient information (no detach operations)
3. Allow backward pass through PyTorch's autograd

Gradient Flow Through Time Steps
---------------------------------

Each simulation step involves:

.. math::

   \mathbf{v}^{n+1} &= \mathbf{v}^n + \Delta t \, \mathbf{M}^{-1} \mathbf{f}(\mathbf{x}^n, \boldsymbol{\theta})

   \mathbf{x}^{n+1} &= \mathbf{x}^n + \Delta t \, \mathbf{v}^{n+1}

Backward Pass
~~~~~~~~~~~~~

By the chain rule, gradients flow backward:

.. math::

   \frac{\partial \mathcal{L}}{\partial \mathbf{x}^n} &= \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{n+1}} \frac{\partial \mathbf{x}^{n+1}}{\partial \mathbf{x}^n} + \frac{\partial \mathcal{L}}{\partial \mathbf{v}^{n+1}} \frac{\partial \mathbf{v}^{n+1}}{\partial \mathbf{x}^n}

   \frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} &= \sum_{n=0}^{T-1} \frac{\partial \mathcal{L}}{\partial \mathbf{v}^{n+1}} \frac{\partial \mathbf{v}^{n+1}}{\partial \boldsymbol{\theta}}

Computing Force Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~

The key computation is :math:`\frac{\partial \mathbf{f}}{\partial \mathbf{x}}` (force Jacobian) and :math:`\frac{\partial \mathbf{f}}{\partial \boldsymbol{\theta}}`.

Since forces come from energy:

.. math::

   \mathbf{f} = -\nabla_{\mathbf{x}} E(\mathbf{x}, \boldsymbol{\theta})

The gradient is:

.. math::

   \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = -\nabla^2_{\mathbf{x}} E = -\mathbf{H}

where :math:`\mathbf{H}` is the **Hessian** of the energy (stiffness matrix in FEM terminology).

Energy-Based Force Computation
-------------------------------

In torch-diffsim, forces are computed as:

.. code-block:: python

   # Compute elastic energy
   E_elastic = sum(psi(F_e) * V_e for all elements)
   
   # Forces as negative energy gradient
   forces = -autograd.grad(E_elastic, positions, create_graph=True)[0]

This approach:

1. **Automatically handles material gradients**: :math:`\frac{\partial \mathbf{f}}{\partial \boldsymbol{\theta}}` is computed by PyTorch
2. **Ensures correctness**: Forces are guaranteed to be energy-consistent
3. **Enables higher-order derivatives**: ``create_graph=True`` maintains the graph

Material Parameter Gradients
-----------------------------

For learnable material parameters :math:`E` and :math:`\nu`:

.. math::

   \frac{\partial \mathcal{L}}{\partial E} = \sum_{n=0}^{T-1} \frac{\partial \mathcal{L}}{\partial \mathbf{f}^n} \frac{\partial \mathbf{f}^n}{\partial E}

The force derivative :math:`\frac{\partial \mathbf{f}}{\partial E}` comes from:

.. math::

   \frac{\partial \mathbf{f}}{\partial E} = -\frac{\partial}{\partial E} \left( \nabla_{\mathbf{x}} \sum_e \Psi(\mathbf{F}_e, E, \nu) V_e \right)

Since :math:`\mu` and :math:`\lambda` depend on :math:`E`:

.. math::

   \frac{\partial \mu}{\partial E} = \frac{1}{2(1+\nu)}, \quad \frac{\partial \lambda}{\partial E} = \frac{\nu}{(1+\nu)(1-2\nu)}

PyTorch autograd handles this automatically when parameters are `torch.nn.Parameter`.

Differentiable Contact
----------------------

Contact forces must also be differentiable. The barrier function:

.. math::

   b(d) = -\kappa (d - \hat{d})^2 \log(d / \hat{d})

has gradient:

.. math::

   \frac{\partial b}{\partial d} = -\kappa \left[ 2(d - \hat{d}) \log(d/\hat{d}) + \frac{(d - \hat{d})^2}{d} \right]

This is smooth (no discontinuities), enabling gradient flow even through contact events.

Smooth Operations for Differentiation
--------------------------------------

Several operations are modified to maintain smoothness:

Velocity Clamping
~~~~~~~~~~~~~~~~~

Instead of hard clamp:

.. math::

   \mathbf{v} \leftarrow \min(\mathbf{v}, v_{\max})  \quad \text{(non-differentiable)}

Use smooth clamp via ``tanh``:

.. math::

   \mathbf{v} \leftarrow \mathbf{v} \cdot \tanh\left(\frac{v_{\max}}{\|\mathbf{v}\|}\right)  \quad \text{(differentiable)}

Fixed Vertices
~~~~~~~~~~~~~~

Instead of hard assignment:

.. math::

   \mathbf{v}_i = \mathbf{0} \quad \text{(breaks gradients)}

Use masking:

.. math::

   \mathbf{v} \leftarrow \mathbf{v} \odot (\mathbf{1} - \mathbf{m})

where :math:`\mathbf{m}` is a binary mask (:math:`m_i = 1` for fixed vertices).

Memory-Efficient Backpropagation
---------------------------------

For long simulations (many time steps), storing the entire computational graph is memory-intensive.

Gradient Checkpointing
~~~~~~~~~~~~~~~~~~~~~~

Gradient checkpointing trades computation for memory:

1. Forward: Store only every :math:`K`-th intermediate state
2. Backward: Recompute intermediate states as needed

For a simulation with :math:`T` steps:

- Without checkpointing :math:`O(T)` memory
- With checkpointing :math:`O(\sqrt{T})` memory (with optimal :math:`K`)

In torch-diffsim:

.. code-block:: python

   from torch.utils.checkpoint import checkpoint
   
   for i in range(num_steps):
       if i % checkpoint_every == 0:
           state = checkpoint(step_fn, state, use_reentrant=False)
       else:
           state = step_fn(state)

Implicit Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

For very long simulations or when only the final state matters, implicit differentiation can be used.

For an equilibrium problem :math:`\mathbf{f}(\mathbf{x}^*, \boldsymbol{\theta}) = 0`, the gradient is:

.. math::

   \frac{\partial \mathbf{x}^*}{\partial \boldsymbol{\theta}} = -\left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right)^{-1} \frac{\partial \mathbf{f}}{\partial \boldsymbol{\theta}}

This requires only:

1. Solving a linear system (CG or direct solve)
2. Computing :math:`\frac{\partial \mathbf{f}}{\partial \boldsymbol{\theta}}` at the solution

Benefit: :math:`O(1)` memory regardless of simulation length.

Tradeoff: Less accurate gradients, only works for steady-state problems.

Spatially Varying Materials
----------------------------

For per-element material properties :math:`E_e`, the gradient is:

.. math::

   \frac{\partial \mathcal{L}}{\partial E_e} = \frac{\partial \mathcal{L}}{\partial \Psi_e} \frac{\partial \Psi_e}{\partial E_e} V_e

Since each element's energy only depends on its own :math:`E_e`, gradients are computed independently per element.

**Log-space parameterization**: To ensure positivity, we parameterize:

.. math::

   E_e = \exp(\log E_e)

Then optimize :math:`\log E_e` instead of :math:`E_e` directly. The gradient transforms as:

.. math::

   \frac{\partial \mathcal{L}}{\partial \log E_e} = E_e \frac{\partial \mathcal{L}}{\partial E_e}

Optimization Example
--------------------

Material parameter optimization:

.. math::

   \min_{E, \nu} \mathcal{L}(\mathbf{x}_T(E, \nu), \mathbf{x}_{\text{target}})

Algorithm:

1. Forward: Run simulation with current :math:`E, \nu` → get :math:`\mathbf{x}_T`
2. Loss: Compute :math:`\mathcal{L} = \|\mathbf{x}_T - \mathbf{x}_{\text{target}}\|^2`
3. Backward: Compute :math:`\frac{\partial \mathcal{L}}{\partial E}, \frac{\partial \mathcal{L}}{\partial \nu}`
4. Update: :math:`E \leftarrow E - \alpha \frac{\partial \mathcal{L}}{\partial E}`

Practical Considerations
------------------------

Gradient Explosion

If gradients become too large:

- Reduce learning rate
- Use gradient clipping: ``torch.nn.utils.clip_grad_norm_(params, max_norm)``
- Increase damping in simulation
- Reduce time step

Gradient Vanishing

If gradients become too small:

- Increase learning rate  
- Normalize loss by number of steps
- Use adaptive optimizers (Adam, AdamW)
- Check for numerical instabilities

Numerical Stability

- Clamp :math:`J` to prevent inversion: :math:`J \in [0.1, 5.0]`
- Use stable material models (log terms instead of polynomials)
- Apply velocity clamping to prevent blow-up
- Use substepping for better stability

Verification
------------

To verify gradients are correct, use finite differences:

.. math::

   \frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}

Compare with autograd gradients. They should match to numerical precision (:math:`\epsilon \sim 10^{-5}`).
