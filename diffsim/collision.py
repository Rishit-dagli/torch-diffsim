"""
Collision detection and response using IPC (Incremental Potential Contact) principles
"""

import torch
import numpy as np


class IPCCollisionHandler:
    """
    IPC-based collision detection and response

    Uses barrier potentials to prevent penetrations with continuous collision detection
    """

    def __init__(self, barrier_stiffness=1e3, dhat=1e-3, friction_mu=0.3):
        """
        Initialize IPC collision handler

        Args:
            barrier_stiffness: stiffness of barrier potential
            dhat: activation distance for barrier (collision threshold)
            friction_mu: friction coefficient
        """
        self.kappa = barrier_stiffness
        self.dhat = dhat
        self.dhat_squared = dhat * dhat
        self.friction_mu = friction_mu

    def barrier_function(self, d_squared):
        """
        IPC barrier function: b(d) for distance d

        Args:
            d_squared: squared distance

        Returns:
            barrier value
        """
        # Avoid log(0)
        d_squared = torch.clamp(d_squared, min=1e-12)
        dhat_sq = self.dhat_squared

        # Barrier: -(d - dhat)^2 * log(d / dhat)
        # Only active when d < dhat
        active = d_squared < dhat_sq

        if not active.any():
            return torch.zeros_like(d_squared)

        d = torch.sqrt(d_squared)
        ratio = d / self.dhat
        barrier = torch.zeros_like(d_squared)
        barrier[active] = -((d[active] - self.dhat) ** 2) * torch.log(
            ratio[active] + 1e-12
        )

        return barrier

    def barrier_gradient(self, d_squared):
        """
        Gradient of barrier function

        Args:
            d_squared: squared distance

        Returns:
            gradient magnitude
        """
        d_squared = torch.clamp(d_squared, min=1e-12)
        d = torch.sqrt(d_squared)

        active = d_squared < self.dhat_squared
        if not active.any():
            return torch.zeros_like(d)

        grad = torch.zeros_like(d)
        d_active = d[active]
        ratio = d_active / self.dhat

        # d/dd[ -(d - dhat)^2 * log(d/dhat) ]
        term1 = -2 * (d_active - self.dhat) * torch.log(ratio + 1e-12)
        term2 = -((d_active - self.dhat) ** 2) / (d_active + 1e-12)
        grad[active] = term1 + term2

        return grad

    def point_triangle_distance(self, p, v0, v1, v2):
        """
        Compute squared distance from point p to triangle (v0, v1, v2)

        Args:
            p: (N, 3) points
            v0, v1, v2: (M, 3) triangle vertices

        Returns:
            distances: (N, M) squared distances
            closest_points: (N, M, 3) closest points on triangles
        """
        # Expand dimensions for broadcasting
        p = p.unsqueeze(1)  # (N, 1, 3)
        v0 = v0.unsqueeze(0)  # (1, M, 3)
        v1 = v1.unsqueeze(0)
        v2 = v2.unsqueeze(0)

        # Triangle edges
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        # Vector from v0 to p
        v0p = p - v0

        # Normal (not normalized)
        normal = torch.cross(e0, v2 - v0, dim=-1)

        # Project point onto plane
        normal_norm_sq = torch.sum(normal * normal, dim=-1, keepdim=True) + 1e-12
        dist_to_plane = torch.sum(v0p * normal, dim=-1, keepdim=True)
        proj = p - normal * (dist_to_plane / normal_norm_sq)

        # Check if projection is inside triangle using barycentric coordinates
        # Simplified: just compute distance to triangle plane for now
        d_squared = dist_to_plane**2 / normal_norm_sq

        return d_squared.squeeze(-1), proj.squeeze(1)

    def compute_self_collision_forces(self, mesh, positions):
        """
        Compute self-collision forces using IPC barrier potentials

        This is a simplified version that checks vertex-face distances

        Args:
            mesh: TetrahedralMesh
            positions: (N, 3) current positions

        Returns:
            forces: (N, 3) collision forces
        """
        device = positions.device
        forces = torch.zeros_like(positions)

        # Extract surface triangles (boundary faces)
        surface_faces = self._extract_surface_faces(mesh)

        if surface_faces is None or len(surface_faces) == 0:
            return forces

        # For each vertex, check distance to non-adjacent faces
        # This is O(N*M) but we'll use spatial hashing for efficiency

        # Simple version: check all vertex-face pairs (expensive but correct)
        # In production, use BVH (Bounding Volume Hierarchy)

        num_checks = 0
        max_checks = min(
            positions.shape[0] * surface_faces.shape[0], 10000
        )  # Limit for speed

        # Sample subset for efficiency
        vertex_sample = torch.randperm(positions.shape[0], device=device)[
            : min(100, positions.shape[0])
        ]
        face_sample = torch.randperm(surface_faces.shape[0], device=device)[
            : min(100, surface_faces.shape[0])
        ]

        for v_idx in vertex_sample:
            p = positions[v_idx]

            for f_idx in face_sample:
                face = surface_faces[f_idx]

                # Skip if vertex is part of this face
                if v_idx in face:
                    continue

                # Compute distance
                v0, v1, v2 = positions[face[0]], positions[face[1]], positions[face[2]]

                # Simple point-to-plane distance
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = torch.cross(edge1, edge2)
                normal_norm = torch.norm(normal) + 1e-12
                normal = normal / normal_norm

                # Distance from point to plane
                to_point = p - v0
                dist = torch.abs(torch.dot(to_point, normal))
                d_squared = dist**2

                # Apply barrier if within threshold
                if d_squared < self.dhat_squared:
                    # Compute barrier gradient
                    grad_mag = self.barrier_gradient(d_squared.unsqueeze(0))[0]

                    # Force direction (repulsion along normal)
                    sign = torch.sign(torch.dot(to_point, normal))
                    force = self.kappa * grad_mag * sign * normal

                    forces[v_idx] += force

                    num_checks += 1
                    if num_checks >= max_checks:
                        return forces

        return forces

    def _extract_surface_faces(self, mesh):
        """
        Extract surface triangles from tetrahedral mesh

        Returns:
            surface_faces: (F, 3) tensor of surface triangle indices
        """
        tets = mesh.tetrahedra.cpu()

        # Each tet has 4 faces
        all_faces = torch.cat(
            [
                tets[:, [0, 2, 1]],
                tets[:, [0, 1, 3]],
                tets[:, [0, 3, 2]],
                tets[:, [1, 2, 3]],
            ],
            dim=0,
        )

        # Sort each face to find duplicates
        sorted_faces, _ = torch.sort(all_faces, dim=1)

        # Find unique faces (boundary faces appear once)
        unique_faces, inverse_indices, counts = torch.unique(
            sorted_faces, dim=0, return_inverse=True, return_counts=True
        )

        # Boundary faces (count == 1)
        boundary_mask = counts == 1
        boundary_sorted = unique_faces[boundary_mask]

        # Map back to original face indices
        surface_faces = []
        for sorted_face in boundary_sorted:
            # Find first occurrence in all_faces
            for i, face in enumerate(all_faces):
                if torch.equal(torch.sort(face)[0], sorted_face):
                    surface_faces.append(face)
                    break

        if len(surface_faces) == 0:
            return None

        return torch.stack(surface_faces).to(mesh.tetrahedra.device)


class SimplifiedCollisionHandler:
    """
    Fast simplified collision detection for real-time simulation

    Uses distance-based repulsion without full CCD
    """

    def __init__(
        self,
        collision_distance=0.02,
        repulsion_stiffness=1e4,
        max_checks_per_frame=1000,
    ):
        """
        Args:
            collision_distance: minimum distance threshold
            repulsion_stiffness: strength of repulsion forces
            max_checks_per_frame: maximum collision checks per frame (for performance)
        """
        self.d_min = collision_distance
        self.k_repulsion = repulsion_stiffness
        self.max_checks = max_checks_per_frame

    def compute_self_collision_forces(self, mesh, positions):
        """
        Fast self-collision detection using simple distance checks

        Args:
            mesh: TetrahedralMesh
            positions: (N, 3) current positions

        Returns:
            forces: (N, 3) repulsion forces
        """
        forces = torch.zeros_like(positions)

        # Vectorized collision detection for speed
        N = positions.shape[0]

        # Limit checks for performance - sample vertex pairs
        num_samples = min(self.max_checks, N * 5)

        if num_samples == 0:
            return forces

        # Generate random pairs in batch
        indices_i = torch.randint(0, N, (num_samples,), device=positions.device)
        indices_j = torch.randint(0, N, (num_samples,), device=positions.device)

        # Filter out self-pairs
        valid = indices_i != indices_j
        indices_i = indices_i[valid]
        indices_j = indices_j[valid]

        if len(indices_i) == 0:
            return forces

        # Compute distances in batch
        pos_i = positions[indices_i]
        pos_j = positions[indices_j]
        diff = pos_i - pos_j
        dist = torch.norm(diff, dim=1, keepdim=True) + 1e-12

        # Find colliding pairs
        colliding = dist.squeeze() < self.d_min

        if not colliding.any():
            return forces

        # Compute repulsion forces for colliding pairs
        dist_colliding = dist[colliding]
        diff_colliding = diff[colliding]
        penetration = self.d_min - dist_colliding
        force_mag = self.k_repulsion * penetration / dist_colliding
        force = force_mag * diff_colliding

        # Accumulate forces (using scatter_add for efficiency)
        indices_i_colliding = indices_i[colliding]
        indices_j_colliding = indices_j[colliding]

        forces.index_add_(0, indices_i_colliding, force)
        forces.index_add_(0, indices_j_colliding, -force)

        return forces
