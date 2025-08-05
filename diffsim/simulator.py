"""
Main simulator class that integrates all components
"""

import torch
import numpy as np


class Simulator:
    """
    Main physics simulator class

    Combines mesh, material model, and solver into a unified interface
    """

    def __init__(
        self, mesh, material, solver, density=1000.0, device="cpu", use_compile=False
    ):
        """
        Initialize simulator

        Args:
            mesh: TetrahedralMesh object
            material: Material model (e.g., StableNeoHookean)
            solver: Time integration solver (e.g., SemiImplicitSolver)
            density: material density (kg/m^3)
            device: 'cpu' or 'cuda'
            use_compile: use torch.compile for speedup (requires PyTorch 2.0+)
        """
        self.mesh = mesh
        self.material = material
        self.solver = solver
        self.device = torch.device(device)
        self.use_compile = use_compile

        # Move mesh to device
        if self.mesh.device != self.device:
            self.mesh.vertices = self.mesh.vertices.to(self.device)
            self.mesh.tetrahedra = self.mesh.tetrahedra.to(self.device)
            self.mesh.Dm = self.mesh.Dm.to(self.device)
            self.mesh.Dm_inv = self.mesh.Dm_inv.to(self.device)
            self.mesh.rest_volume = self.mesh.rest_volume.to(self.device)
            self.mesh.device = self.device

        # Initialize state
        self.positions = self.mesh.vertices.clone()
        self.velocities = torch.zeros_like(self.positions)

        # Compute masses from density and volume
        self.masses = self._compute_vertex_masses(density)

        # Fixed vertices (for boundary conditions)
        self.fixed_vertices = None

        # Simulation time
        self.time = 0.0

        # Compile step function if requested (PyTorch 2.0+ optimization)
        if self.use_compile and device != "cpu":
            try:
                self._compiled_step = torch.compile(
                    self._step_internal, mode="reduce-overhead"
                )
            except Exception:
                self.use_compile = False
                self._compiled_step = self._step_internal
        else:
            self._compiled_step = self._step_internal

    def _compute_vertex_masses(self, density):
        """
        Compute vertex masses from element volumes and density

        Args:
            density: material density (kg/m^3)

        Returns:
            masses: (N,) vertex masses
        """
        masses = torch.zeros(self.mesh.num_vertices, device=self.device)

        # Distribute element mass to vertices (each vertex gets 1/4 of element mass)
        element_masses = density * self.mesh.rest_volume  # (M,)
        vertex_mass_contribution = element_masses / 4.0

        for i in range(4):
            masses.index_add_(0, self.mesh.tetrahedra[:, i], vertex_mass_contribution)

        return masses

    def set_fixed_vertices(self, vertex_indices):
        """
        Set vertices that should remain fixed during simulation

        Args:
            vertex_indices: list or tensor of vertex indices to fix
        """
        if isinstance(vertex_indices, list):
            vertex_indices = torch.tensor(
                vertex_indices, dtype=torch.long, device=self.device
            )
        self.fixed_vertices = vertex_indices

    def fix_bottom_vertices(self, threshold=0.05):
        """
        Fix vertices below a certain height threshold

        Args:
            threshold: height threshold (relative to min y-coordinate)
        """
        min_y = self.positions[:, 1].min()
        fixed = (self.positions[:, 1] <= min_y + threshold).nonzero(as_tuple=True)[0]
        self.set_fixed_vertices(fixed)

    def add_velocity(self, vertex_indices, velocity):
        """
        Add velocity to specific vertices

        Args:
            vertex_indices: indices of vertices
            velocity: (3,) velocity vector to add
        """
        if isinstance(velocity, (list, np.ndarray)):
            velocity = torch.tensor(velocity, dtype=torch.float32, device=self.device)
        self.velocities[vertex_indices] += velocity

    def _step_internal(self, positions, velocities):
        """Internal step function (can be compiled)"""
        return self.solver.step(
            self.mesh,
            self.material,
            positions,
            velocities,
            self.masses,
            self.fixed_vertices,
        )

    def step(self):
        """Perform one simulation step"""
        if self.use_compile:
            self.positions, self.velocities = self._compiled_step(
                self.positions, self.velocities
            )
        else:
            self.positions, self.velocities = self._step_internal(
                self.positions, self.velocities
            )
        self.time += self.solver.dt

    def reset(self):
        """Reset simulation to initial state"""
        self.positions = self.mesh.vertices.clone()
        self.velocities = torch.zeros_like(self.positions)
        self.time = 0.0

    def warmstart(self, num_steps=10):
        """
        Warm start the simulation to reach equilibrium

        This helps prevent initial artifacts by letting the mesh settle
        with reduced forces before the main simulation starts

        Args:
            num_steps: number of warm-start steps
        """

        # Save original solver settings
        original_dt = self.solver.dt
        original_damping = self.solver.damping

        # Use smaller timestep and more damping for warm start
        self.solver.dt = 0.001
        self.solver.damping = 0.9

        # Run a few steps to reach equilibrium
        for _ in range(num_steps):
            self.step()

        # Restore original settings
        self.solver.dt = original_dt
        self.solver.damping = original_damping
        self.time = 0.0  # Reset time after warm start

    def get_surface_mesh(self):
        """
        Extract surface triangles for visualization (boundary faces only)

        Returns:
            vertices: (N, 3) vertex positions
            faces: (F, 3) triangle indices
        """
        # Extract all faces from tetrahedra
        # Each tetrahedron has 4 faces with consistent winding
        tets = self.mesh.tetrahedra.cpu()

        # Create all faces (M*4, 3) - 4 faces per tetrahedron
        all_faces = torch.cat(
            [
                tets[:, [0, 2, 1]],  # Face opposite to vertex 3
                tets[:, [0, 1, 3]],  # Face opposite to vertex 2
                tets[:, [0, 3, 2]],  # Face opposite to vertex 1
                tets[:, [1, 2, 3]],  # Face opposite to vertex 0
            ],
            dim=0,
        )

        # Sort vertices in each face to find duplicates
        sorted_faces, _ = torch.sort(all_faces, dim=1)

        # Convert to tuple for hashing
        face_dict = {}
        for i, face in enumerate(sorted_faces):
            key = tuple(face.tolist())
            if key in face_dict:
                face_dict[key].append(i)
            else:
                face_dict[key] = [i]

        # Boundary faces appear only once
        boundary_indices = [
            indices[0] for indices in face_dict.values() if len(indices) == 1
        ]
        boundary_faces = all_faces[boundary_indices]

        vertices = self.positions.cpu().numpy()
        faces = boundary_faces.numpy()

        return vertices, faces

    def compute_energy(self):
        """
        Compute total energy (kinetic + potential)

        Returns:
            kinetic_energy: kinetic energy
            elastic_energy: elastic potential energy
            gravitational_energy: gravitational potential energy
        """
        # Kinetic energy: 1/2 * m * v^2
        kinetic = 0.5 * torch.sum(self.masses.unsqueeze(-1) * self.velocities**2)

        # Elastic energy
        F = self.mesh.compute_deformation_gradient(self.positions)
        energy_density = self.material.energy_density(F)
        elastic = torch.sum(energy_density * self.mesh.rest_volume)

        # Gravitational energy: m * g * h
        g_value = (
            self.solver.gravity_value
            if self.solver.gravity is None
            else self.solver.gravity[1].item()
        )
        gravitational = torch.sum(self.masses * g_value * self.positions[:, 1])

        return kinetic.item(), elastic.item(), gravitational.item()
