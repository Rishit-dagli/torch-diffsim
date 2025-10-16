"""
Differentiable physics simulation utilities

This module provides advanced techniques for differentiable physics simulation:

1. **Smooth Contact Approximations**: Replace hard constraints with smooth barrier
   functions that maintain gradient flow

2. **Implicit Differentiation**: Compute gradients through iterative solvers using
   the implicit function theorem without storing the full computation graph

3. **Gradient Checkpointing**: Memory-efficient backpropagation through long simulations
   by recomputing forward passes instead of storing all intermediate states

4. **Learnable Materials**: Material models with parameters that can be optimized
   via gradient descent

All components are designed to preserve gradients for automatic differentiation
while maintaining numerical stability.
"""

import torch
from torch.utils.checkpoint import checkpoint


class DifferentiableBarrierContact:
    """
    Smooth, differentiable contact model using barrier functions

    This class implements smooth contact forces that maintain gradient flow for
    automatic differentiation. Instead of hard constraints or projections that
    introduce discontinuities, it uses smooth barrier potential functions that
    activate near contact.

    The barrier potential is:

    .. math::

        b(d) = -(d - \\hat{d})^2 \\log(d/\\hat{d}) \\text{ for } d < \\hat{d}

    where :math:`d` is the distance to the surface and :math:`\\hat{d}` is the
    barrier activation distance. The potential smoothly increases as :math:`d \\to 0`,
    preventing penetration while maintaining :math:`C^2` continuity.

    Parameters:
        barrier_stiffness (float): Stiffness of barrier forces (default: 1e4)
        barrier_width (float): Distance at which barrier activates (default: 0.01)

    Attributes:
        kappa (float): Barrier stiffness coefficient
        d_hat (float): Barrier activation distance
    """

    def __init__(self, barrier_stiffness=1e4, barrier_width=0.01):
        """
        Args:
            barrier_stiffness: controls contact force magnitude
            barrier_width: distance at which barrier activates
        """
        self.kappa = barrier_stiffness
        self.d_hat = barrier_width

    def barrier_potential(self, d):
        """
        Smooth barrier potential: -(d - d_hat)^2 * log(d/d_hat) for d < d_hat

        This is C^2 continuous and always differentiable

        Args:
            d: :math:`(N,)` distances to surface

        Returns:
            energy: :math:`(N,)` barrier energy
        """
        # Smooth activation (no if-statements for AD)
        # Use smooth step function instead of hard threshold
        activation = torch.sigmoid((self.d_hat - d) * 100)  # Smooth 0/1 transition

        # Barrier function (only active when d < d_hat)
        d_safe = torch.clamp(d, min=1e-6)  # Prevent log(0)
        barrier = -((d_safe - self.d_hat) ** 2) * torch.log(d_safe / self.d_hat + 1e-12)

        return self.kappa * activation * barrier

    def ground_contact_force(self, positions, velocities):
        """
        Compute smooth ground contact forces (fully differentiable)

        Args:
            positions: :math:`(N, 3)` positions
            velocities: :math:`(N, 3)` velocities

        Returns:
            forces: :math:`(N, 3)` contact forces
        """
        # Distance to ground (y-coordinate). Use contiguous copy to avoid view versioning issues.
        d = positions[:, 1].contiguous()

        # Compute differentiable barrier energy
        energy = self.barrier_potential(d)  # (N,)

        # Contact force is negative gradient of energy wrt positions
        grad_pos = torch.autograd.grad(
            energy.sum(), positions, create_graph=True, retain_graph=True
        )[0]

        forces = torch.zeros_like(positions)
        forces[:, 1] = -grad_pos[:, 1]

        # Smooth friction (tangential damping proportional to normal force magnitude)
        friction_coeff = 0.3
        tangent_vel = velocities.clone()
        tangent_vel[:, 1] = 0
        normal_mag = forces[:, 1].abs().unsqueeze(-1)
        friction_force = -friction_coeff * normal_mag * tangent_vel
        forces = forces + friction_force

        return forces


class ImplicitDifferentiation:
    """
    Implicit differentiation for backward Euler and Newton solvers

    Instead of differentiating through all Newton iterations,
    use the implicit function theorem:

    If F(x*, p) = 0, then dx*/dp = -(dF/dx)^{-1} @ (dF/dp)

    This avoids storing the full computation graph
    """

    @staticmethod
    def implicit_backward(forward_fn, x_star, params, atol=1e-6):
        """
        Compute gradients using implicit differentiation

        Args:
            forward_fn: function that computes residual F(x, params) = 0
            x_star: solution where F(x_star, params) = 0
            params: parameters to differentiate wrt
            atol: tolerance for linear solve

        Returns:
            x_star: solution with proper gradients attached
        """

        class ImplicitFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *params_tuple):
                # Just return the solution
                ctx.save_for_backward(x_star, *params_tuple)
                return x_star

            @staticmethod
            def backward(ctx, grad_output):
                """
                Compute vjp: grad_output^T @ dx/dp using implicit function theorem
                """
                x_star_saved = ctx.saved_tensors[0]
                params_saved = ctx.saved_tensors[1:]

                # Compute Jacobian dF/dx at x_star (needed for implicit diff)
                # Use Jacobian-free approach: approximate with finite differences
                def matvec(v):
                    """Compute (dF/dx) @ v without forming full Jacobian"""
                    eps = 1e-4
                    x_plus = x_star_saved + eps * v
                    F_plus = forward_fn(x_plus, *params_saved)
                    F_0 = forward_fn(x_star_saved, *params_saved)
                    return (F_plus - F_0) / eps

                # Solve (dF/dx)^T @ lambda = grad_output using CG
                # This is the adjoint equation
                lambda_star = conjugate_gradient(
                    lambda v: matvec(v).detach(),  # Detach to avoid double backprop
                    grad_output.flatten(),
                    max_iters=50,
                    atol=atol,
                ).reshape_as(grad_output)

                # Now compute dL/dp = -lambda^T @ dF/dp
                param_grads = []
                for p in params_saved:
                    if p.requires_grad:
                        # Compute dF/dp
                        F = forward_fn(x_star_saved.detach(), p)
                        dF_dp = torch.autograd.grad(
                            F, p, grad_outputs=torch.ones_like(F), retain_graph=True
                        )[0]

                        # Compute vjp
                        grad_p = -(lambda_star.flatten() @ dF_dp.flatten())
                        param_grads.append(grad_p)
                    else:
                        param_grads.append(None)

                return tuple(param_grads)

        return ImplicitFunction.apply(*params)


def conjugate_gradient(A, b, x0=None, max_iters=100, atol=1e-6):
    """
    Solve Ax = b using conjugate gradient

    Jacobian-free: A is a function that computes matrix-vector products

    Args:
        A: function that computes A @ v
        b: right-hand side
        x0: initial guess
        max_iters: maximum iterations
        atol: absolute tolerance

    Returns:
        x: solution
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A(x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iters):
        Ap = A(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < atol:
            break

        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new

    return x


class CheckpointedRollout:
    """
    Memory-efficient rollout using gradient checkpointing

    Instead of storing all intermediate states, recompute them during backward pass
    This trades computation for memory
    """

    @staticmethod
    def rollout(step_fn, state0, num_steps, checkpoint_every=10):
        """
        Perform rollout with checkpointing

        Args:
            step_fn: function that computes next state: s_{t+1} = step_fn(s_t)
            state0: initial state
            num_steps: number of steps
            checkpoint_every: save state every N steps

        Returns:
            trajectory: list of states [s_0, s_1, ..., s_T]
        """
        trajectory = [state0]
        state = state0

        def _step_fn_tensors(pos, vel):
            return step_fn((pos, vel))

        for i in range(num_steps):
            # Use PyTorch's checkpoint utility
            if i % checkpoint_every == 0:
                pos, vel = state
                state = checkpoint(_step_fn_tensors, pos, vel, use_reentrant=False)
            else:
                # Regular forward pass
                state = step_fn(state)

            trajectory.append(state)

        return trajectory


class DifferentiableMaterial(torch.nn.Module):
    """
    Material model with learnable parameters

    This class wraps material properties as PyTorch parameters, enabling gradient-based
    optimization of material constants. It implements the Stable Neo-Hookean energy
    density with learnable Young's modulus :math:`E` and Poisson's ratio :math:`\\nu`.

    The energy density is:

    .. math::

        \\Psi(\\mathbf{F}) = \\frac{\\mu}{2}(I_C - 3) - \\mu \\log J + \\frac{\\lambda}{2}(J-1)^2

    where :math:`\\mu` and :math:`\\lambda` are computed from :math:`E` and :math:`\\nu`.

    Parameters:
        youngs_modulus (float): Initial Young's modulus value
        poissons_ratio (float): Initial Poisson's ratio value
        requires_grad (bool): Whether parameters should track gradients (default: True)

    Attributes:
        E (torch.nn.Parameter): Learnable Young's modulus
        nu (torch.nn.Parameter): Learnable Poisson's ratio

    Example:
        >>> material = DifferentiableMaterial(1e5, 0.4, requires_grad=True)
        >>> optimizer = torch.optim.Adam(material.parameters(), lr=1e3)
        >>> # Run simulation and optimize material.E and material.nu
    """

    def __init__(self, youngs_modulus, poissons_ratio, requires_grad=True):
        """
        Args:
            youngs_modulus: (scalar or per-element) Young's modulus
            poissons_ratio: (scalar or per-element) Poisson's ratio
            requires_grad: whether to track gradients
        """
        super().__init__()
        # Convert to parameters (requires_grad=True by default)
        self.E = torch.nn.Parameter(
            torch.tensor(youngs_modulus, dtype=torch.float32),
            requires_grad=requires_grad,
        )
        self.nu = torch.nn.Parameter(
            torch.tensor(poissons_ratio, dtype=torch.float32),
            requires_grad=requires_grad,
        )

    @property
    def mu(self):
        """Lamé parameter μ (differentiable)"""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self):
        """Lamé parameter λ (differentiable)"""
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def energy_density(self, F):
        """
        Compute strain energy (fully differentiable)

        Args:
            F: :math:`(M, 3, 3)` deformation gradient

        Returns:
            psi: :math:`(M,)` energy density
        """
        # Stable neo-Hookean energy
        Ic = torch.sum(F * F, dim=(1, 2))
        J = torch.det(F)
        J = torch.clamp(J, min=0.1, max=5.0)  # Smooth clamping

        psi = (
            self.mu / 2.0 * (Ic - 3.0)
            - self.mu * torch.log(J)
            + self.lam / 2.0 * (J - 1.0) ** 2
        )

        return psi


class SpatiallyVaryingMaterial(torch.nn.Module):
    """
    Material with spatially varying properties

    This class allows each element in the mesh to have independent material properties,
    enabling optimization of heterogeneous material distributions. Young's modulus is
    parameterized in log-space to ensure positivity:

    .. math::

        E_i = \\exp(\\log E_i)

    This is particularly useful for:

    - Inverse material identification problems
    - Topology optimization
    - Functionally graded material design
    - Material distribution learning

    Parameters:
        num_elements (int): Number of elements in the mesh
        base_youngs (float): Initial Young's modulus for all elements (default: 1e5)
        base_poisson (float): Poisson's ratio for all elements (default: 0.4)

    Attributes:
        log_E (torch.nn.Parameter): Log-space Young's modulus per element :math:`(M,)`
        nu (torch.nn.Parameter): Poisson's ratio per element :math:`(M,)`

    Example:
        >>> material = SpatiallyVaryingMaterial(mesh.num_elements, 1e5, 0.4)
        >>> optimizer = torch.optim.Adam([material.log_E], lr=0.01)
        >>> # Optimize spatial distribution of stiffness
    """

    def __init__(self, num_elements, base_youngs=1e5, base_poisson=0.4):
        """
        Args:
            num_elements: number of elements in mesh
            base_youngs: base Young's modulus
            base_poisson: base Poisson's ratio
        """
        super().__init__()
        # Per-element moduli (log-space for positivity)
        self.log_E = torch.nn.Parameter(
            torch.ones(num_elements) * torch.log(torch.tensor(base_youngs))
        )
        self.nu = torch.nn.Parameter(torch.ones(num_elements) * base_poisson)

    @property
    def E(self):
        """Young's modulus (always positive via exp)"""
        return torch.exp(self.log_E)

    @property
    def mu(self):
        """Per-element μ"""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self):
        """Per-element λ"""
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def energy_density(self, F):
        """
        Compute strain energy per element (fully differentiable)

        Args:
            F: :math:`(M, 3, 3)` deformation gradient

        Returns:
            psi: :math:`(M,)` energy density
        """
        # Stable neo-Hookean energy
        Ic = torch.sum(F * F, dim=(1, 2))
        J = torch.det(F)
        J = torch.clamp(J, min=0.1, max=5.0)  # Smooth clamping

        # Per-element material properties
        mu = self.mu
        lam = self.lam

        psi = mu / 2.0 * (Ic - 3.0) - mu * torch.log(J) + lam / 2.0 * (J - 1.0) ** 2

        return psi


def smooth_step(x, edge=0.0, width=1.0):
    """
    Smooth step function (differentiable replacement for if/else)

    Args:
        x: input
        edge: center of transition
        width: width of transition

    Returns:
        smooth step from 0 to 1
    """
    t = torch.clamp((x - edge) / width + 0.5, 0, 1)
    return t * t * (3.0 - 2.0 * t)  # Hermite interpolation


def log_barrier(x, eps=1e-3):
    """
    Smooth log barrier: -log(x) for x > eps, quadratic for x < eps

    C^1 continuous approximation of log barrier for AD

    Args:
        x: input (must be positive)
        eps: smoothing parameter

    Returns:
        barrier value
    """
    # Smooth transition
    mask = x > eps

    # Log barrier for x > eps
    log_part = -torch.log(x + 1e-12)

    # Quadratic extension for x < eps (maintains C^1 continuity)
    quad_part = (
        -torch.log(torch.tensor(eps, device=x.device, dtype=x.dtype))
        + (eps - x) / eps
        + 0.5 * ((eps - x) / eps) ** 2
    )

    return torch.where(mask, log_part, quad_part)
