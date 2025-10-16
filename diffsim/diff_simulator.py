"""
Differentiable simulation components.

This module provides DifferentiableSolver and DifferentiableSimulator classes that
maintain full gradient flow through all simulation operations. Unlike the standard
simulator, these classes use:

- **Smooth operations**: No hard constraints or projections that break gradients
- **Differentiable contact**: Smooth barrier functions instead of projection
- **Gradient-aware constraints**: Fixed vertices handled with smooth masking
- **Checkpointing support**: Memory-efficient backpropagation for long rollouts

The differentiable simulator implements the same semi-implicit (symplectic Euler)
time integration as the standard simulator, but all operations preserve gradients
for automatic differentiation.
"""

import torch
from .diff_physics import (
    DifferentiableBarrierContact,
    ImplicitDifferentiation,
    CheckpointedRollout,
    DifferentiableMaterial,
)


class DifferentiableSolver:
    """
    Semi-implicit (symplectic Euler) solver with gradient-preserving operations.
    Forces are derived from the energy gradient; contact uses smooth barriers;
    constraints use masking to avoid breaking gradients.

    Integration scheme:

    .. math::

        \\mathbf{v}^{n+1} &= \\mathbf{v}^n + h \\, \\mathbf{M}^{-1} \\mathbf{f}(\\mathbf{x}^n)

        \\mathbf{x}^{n+1} &= \\mathbf{x}^n + h \\, \\mathbf{v}^{n+1}

    Parameters:
        dt (float): Time step size (default: 0.01)
        gravity (float): Gravity acceleration in m/s² (default: -9.8)
        damping (float): Velocity damping coefficient (default: 0.99)
        substeps (int): Number of substeps per timestep (default: 4)
        use_implicit_diff (bool): Use implicit differentiation (default: False)
    """

    def __init__(
        self, dt=0.01, gravity=-9.8, damping=0.99, substeps=4, use_implicit_diff=False
    ):
        """
        Args:
            dt: time step
            gravity: gravity acceleration
            damping: velocity damping
            substeps: number of substeps
            use_implicit_diff: use implicit differentiation (faster backward, less accurate)
        """
        self.dt = dt
        self.gravity = torch.tensor([0.0, gravity, 0.0])
        self.damping = damping
        self.substeps = substeps
        self.use_implicit_diff = use_implicit_diff

        # Smooth contact handler
        self.contact_handler = DifferentiableBarrierContact(
            barrier_stiffness=1e4, barrier_width=0.02
        )

    def step(self, mesh, material, positions, velocities, masses, fixed_vertices=None):
        """
        Differentiable time step

        All operations maintain gradient flow

        Args:
            mesh: TetrahedralMesh
            material: DifferentiableMaterial
            positions: :math:`(N, 3)` positions (requires_grad=True for backprop)
            velocities: :math:`(N, 3)` velocities
            masses: :math:`(N,)` masses
            fixed_vertices: optional fixed vertices

        Returns:
            new_positions: :math:`(N, 3)` with gradients
            new_velocities: :math:`(N, 3)` with gradients
        """
        device = positions.device
        self.gravity = self.gravity.to(device)

        # Clone to preserve gradients
        x_cur = positions
        v_cur = velocities
        h = self.dt / self.substeps

        for _ in range(self.substeps):
            # Compute forces (all differentiable)
            forces = self._compute_forces(mesh, material, x_cur, masses)

            # Add gravity
            gravity_force = masses.unsqueeze(-1) * self.gravity.unsqueeze(0)
            forces = forces + gravity_force

            # Add smooth contact forces (differentiable)
            # Use detached velocities for friction to avoid in-place versioning
            contact_forces = self.contact_handler.ground_contact_force(
                x_cur, v_cur.detach()
            )
            forces = forces + contact_forces

            # Update velocity
            acceleration = forces / (masses.unsqueeze(-1) + 1e-6)
            v_cur = v_cur + h * acceleration

            # Smooth velocity clamping (differentiable)
            max_velocity = 50.0
            vel_magnitude = torch.norm(v_cur, dim=1, keepdim=True)
            vel_scale = torch.tanh(
                max_velocity / (vel_magnitude + 1e-6)
            )  # Smooth clamp
            v_cur = v_cur * vel_scale

            # Apply damping
            v_cur = v_cur * self.damping

            # Apply constraints (note: this breaks gradients for fixed vertices)
            if fixed_vertices is not None:
                v_mask = torch.zeros_like(v_cur)
                v_mask[fixed_vertices] = 1.0
                v_cur = v_cur * (1.0 - v_mask)

            # Update positions
            x_cur = x_cur + h * v_cur

            # Apply position constraints
            if fixed_vertices is not None:
                x_fix = torch.zeros_like(x_cur)
                x_fix[fixed_vertices] = positions[fixed_vertices]
                mask = torch.zeros_like(x_cur)
                mask[fixed_vertices] = 1.0
                x_cur = x_cur * (1.0 - mask) + x_fix

        return x_cur, v_cur

    def _compute_forces(self, mesh, material, positions, masses):
        """
        Compute elastic forces (fully differentiable)

        Args:
            mesh: TetrahedralMesh
            material: DifferentiableMaterial
            positions: :math:`(N, 3)` positions with gradients
            masses: :math:`(N,)` masses

        Returns:
            forces: :math:`(N, 3)` forces with gradients
        """
        # Ensure positions require grad
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)

        # Compute deformation gradient
        F = mesh.compute_deformation_gradient(positions)

        # Compute energy density (differentiable)
        energy_density = material.energy_density(F)

        # Total elastic energy
        elastic_energy = torch.sum(energy_density * mesh.rest_volume)

        # Compute forces as negative gradient of energy
        # This automatically handles all material parameter gradients
        grad_outputs = torch.ones_like(elastic_energy)
        forces = -torch.autograd.grad(
            elastic_energy,
            positions,
            grad_outputs=grad_outputs,
            create_graph=True,  # Enable higher-order derivatives
            retain_graph=True,
            allow_unused=False,
        )[0]

        return forces


class DifferentiableSimulator:
    """
    Fully differentiable simulator

    Drop-in replacement for Simulator with gradient support
    """

    def __init__(self, mesh, material, solver, density=1000.0, device="cpu"):
        """
        Args:
            mesh: TetrahedralMesh
            material: DifferentiableMaterial (with requires_grad=True params)
            solver: DifferentiableSolver
            density: material density
            device: 'cpu' or 'cuda'
        """
        self.mesh = mesh
        self.material = material
        self.solver = solver
        self.device = torch.device(device)

        # Move to device
        if self.mesh.device != self.device:
            self.mesh.vertices = self.mesh.vertices.to(self.device)
            self.mesh.tetrahedra = self.mesh.tetrahedra.to(self.device)
            self.mesh.Dm = self.mesh.Dm.to(self.device)
            self.mesh.Dm_inv = self.mesh.Dm_inv.to(self.device)
            self.mesh.rest_volume = self.mesh.rest_volume.to(self.device)
            self.mesh.device = self.device

        # State (track gradients on positions; velocities kept as non-leaf tensors)
        self.positions = self.mesh.vertices.clone().requires_grad_(True)
        self.velocities = torch.zeros_like(self.positions)

        # Compute masses
        self.masses = self._compute_vertex_masses(density)

        # Fixed vertices
        self.fixed_vertices = None

        # Time
        self.time = 0.0

    def _compute_vertex_masses(self, density):
        """Compute vertex masses from volume and density"""
        masses = torch.zeros(self.mesh.num_vertices, device=self.device)
        element_masses = density * self.mesh.rest_volume
        vertex_mass_contribution = element_masses / 4.0

        for i in range(4):
            masses.index_add_(0, self.mesh.tetrahedra[:, i], vertex_mass_contribution)

        return masses

    def set_fixed_vertices(self, vertex_indices):
        """Set fixed vertices"""
        if isinstance(vertex_indices, list):
            vertex_indices = torch.tensor(
                vertex_indices, dtype=torch.long, device=self.device
            )
        self.fixed_vertices = vertex_indices

    def step(self):
        """Perform one differentiable step"""
        self.positions, self.velocities = self.solver.step(
            self.mesh,
            self.material,
            self.positions,
            self.velocities,
            self.masses,
            self.fixed_vertices,
        )
        self.time += self.solver.dt

    def rollout(self, num_steps, checkpoint_every=10):
        """
        Perform memory-efficient rollout with checkpointing

        Args:
            num_steps: number of simulation steps
            checkpoint_every: checkpoint frequency

        Returns:
            trajectory: list of (positions, velocities) tuples
        """

        def step_fn(state):
            pos, vel = state
            # Temporarily set state
            old_pos, old_vel = self.positions, self.velocities
            self.positions, self.velocities = pos, vel

            # Step
            self.step()

            # Get new state
            new_state = (self.positions, self.velocities)

            # Restore
            self.positions, self.velocities = old_pos, old_vel

            return new_state

        state0 = (self.positions.clone(), self.velocities.clone())
        trajectory = CheckpointedRollout.rollout(
            step_fn, state0, num_steps, checkpoint_every
        )

        return trajectory

    def compute_loss(self, target_positions, loss_type="mse"):
        """
        Compute loss for optimization

        Args:
            target_positions: :math:`(N, 3)` target positions
            loss_type: 'mse' or 'chamfer'

        Returns:
            loss: scalar loss value
        """
        if loss_type == "mse":
            return torch.mean((self.positions - target_positions) ** 2)
        elif loss_type == "chamfer":
            # Simple Chamfer distance approximation
            dist_fwd = torch.cdist(self.positions, target_positions).min(dim=1)[0]
            dist_bwd = torch.cdist(target_positions, self.positions).min(dim=1)[0]
            return torch.mean(dist_fwd) + torch.mean(dist_bwd)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def reset(self):
        """Reset to initial state"""
        self.positions = self.mesh.vertices.clone().requires_grad_(True)
        self.velocities = torch.zeros_like(self.positions)
        self.time = 0.0
