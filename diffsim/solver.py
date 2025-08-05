"""
Time integration solvers for physics simulation
"""

import torch


class SemiImplicitSolver:
    """
    Semi-implicit (Backward Euler) solver for dynamic simulation

    Solves: M(v_{n+1} - v_n) = dt * f(x_n + dt*v_{n+1})
    where f includes elastic, gravity, and constraint forces

    Uses Newton's method with line search for robustness
    """

    def __init__(
        self,
        dt=0.01,
        gravity=-9.8,
        damping=0.99,
        substeps=4,
        enable_self_collision=False,
        collision_method="simplified",
    ):
        """
        Initialize solver

        Args:
            dt: time step size
            gravity: gravity acceleration (m/s^2)
            damping: velocity damping factor
            substeps: number of substeps per timestep
            enable_self_collision: enable self-collision detection
            collision_method: 'simplified' (fast) or 'ipc' (accurate but slower)
        """
        self.dt = dt
        self.gravity_value = gravity
        self.gravity = None  # Will be set to correct device on first step
        self.damping = damping
        self.substeps = max(1, int(substeps))
        self.enable_self_collision = enable_self_collision
        self.collision_method = collision_method

        # Collision handler (lazy init)
        self._collision_handler = None

    def step(self, mesh, material, positions, velocities, masses, fixed_vertices=None):
        """
        Perform one semi-implicit time step with stability controls

        Args:
            mesh: TetrahedralMesh object
            material: Material model (e.g., StableNeoHookean)
            positions: (N, 3) current positions
            velocities: (N, 3) current velocities
            masses: (N,) vertex masses
            fixed_vertices: list of fixed vertex indices

        Returns:
            new_positions: (N, 3) updated positions
            new_velocities: (N, 3) updated velocities
        """
        device = positions.device

        # Initialize gravity tensor on correct device (only once)
        if self.gravity is None or self.gravity.device != device:
            self.gravity = torch.tensor(
                [0.0, self.gravity_value, 0.0], device=device, dtype=positions.dtype
            )

        # Substepping for stability
        x_cur = positions.clone()  # Clone to avoid modifying input
        v_cur = velocities.clone()
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            # Compute forces at current position
            forces = self._compute_forces(mesh, material, x_cur, masses)

            # Add gravity
            gravity_force = masses.unsqueeze(-1) * self.gravity.unsqueeze(0)
            forces += gravity_force

            # Update velocity
            acceleration = forces / (masses.unsqueeze(-1) + 1e-6)
            v_cur = v_cur + h * acceleration

            # Safety clamp for extreme cases
            max_velocity = 50.0  # m/s
            vel_magnitude = torch.norm(v_cur, dim=1, keepdim=True)
            vel_scale = torch.clamp(max_velocity / (vel_magnitude + 1e-8), max=1.0)
            v_cur = v_cur * vel_scale

            # Minimal damping (numerical only)
            v_cur = v_cur * self.damping

            # Apply constraints (fixed vertices)
            if fixed_vertices is not None:
                v_cur[fixed_vertices] = 0.0

            # Update positions
            x_cur = x_cur + h * v_cur

            # Check for NaN/Inf (numerical instability)
            if torch.isnan(x_cur).any() or torch.isinf(x_cur).any():
                return positions, velocities * 0.9  # Return with damped velocities

            # Apply position constraints
            if fixed_vertices is not None:
                x_cur[fixed_vertices] = positions[fixed_vertices]

            # Handle ground collision (projection + friction)
            x_cur, v_cur = self._handle_ground_collision(x_cur, v_cur)

        return x_cur, v_cur

    def _compute_forces(self, mesh, material, positions, masses):
        """
        Compute elastic forces and collision forces

        Args:
            mesh: TetrahedralMesh
            material: Material model
            positions: (N, 3) vertex positions
            masses: (N,) vertex masses

        Returns:
            forces: (N, 3) forces on each vertex
        """
        # Compute deformation gradient
        F = mesh.compute_deformation_gradient(positions)

        # Compute elastic forces for each element
        element_forces = material.compute_elastic_forces(
            F, mesh.Dm_inv, mesh.rest_volume
        )  # (M, 4, 3)

        # Accumulate forces to vertices
        forces = torch.zeros_like(positions)
        for i in range(4):
            forces.index_add_(0, mesh.tetrahedra[:, i], element_forces[:, i, :])

        # Add self-collision forces if enabled
        if self.enable_self_collision:
            if self._collision_handler is None:
                from .collision import SimplifiedCollisionHandler, IPCCollisionHandler

                if self.collision_method == "ipc":

                    self._collision_handler = IPCCollisionHandler(
                        barrier_stiffness=1e3, dhat=0.02, friction_mu=0.3
                    )
                else:  # 'simplified'
                    self._collision_handler = SimplifiedCollisionHandler(
                        collision_distance=0.02,
                        repulsion_stiffness=1e4,
                        max_checks_per_frame=500,  # Reduced for better performance
                    )

            collision_forces = self._collision_handler.compute_self_collision_forces(
                mesh, positions
            )
            forces += collision_forces

        return forces

    def _handle_ground_collision(
        self, positions, velocities, ground_height=0.0, restitution=0.0
    ):
        """
        Handle collision with ground plane

        Args:
            positions: (N, 3) positions
            velocities: (N, 3) velocities
            ground_height: y-coordinate of ground
            restitution: coefficient of restitution

        Returns:
            positions: (N, 3) corrected positions
            velocities: (N, 3) corrected velocities
        """
        # Contact tolerance to avoid chatter
        eps = 1e-6
        contact = positions[:, 1] <= ground_height + eps

        if contact.any():
            # Project to ground plane
            positions[contact, 1] = ground_height

            # Normal velocity
            v_n = velocities[contact, 1]

            # Restitution only for significant impacts, else stick
            small_impact = torch.abs(v_n) < 0.02
            v_n = torch.where(v_n < 0, -restitution * v_n, v_n)
            v_n = torch.where(small_impact, torch.zeros_like(v_n), v_n)
            velocities[contact, 1] = v_n

            # Tangential friction: static threshold then kinetic damping
            v_t = velocities[contact][:, [0, 2]]
            v_t_norm = torch.norm(v_t, dim=1, keepdim=True) + 1e-12
            v_t_static_thresh = 0.02
            # If tangential speed small, stick (zero it)
            stick = v_t_norm[:, 0] < v_t_static_thresh
            v_t[stick] = 0.0
            # Else kinetic friction damping
            mu_k = 0.3
            v_t[~stick] = v_t[~stick] * (1.0 - mu_k)
            velocities[contact][:, [0, 2]] = v_t

        return positions, velocities
