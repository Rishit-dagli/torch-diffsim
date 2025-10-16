"""
Tetrahedral mesh representation for finite element simulation

This module provides the TetrahedralMesh class, which represents volumetric geometry
using tetrahedral elements. The mesh stores vertex positions and element connectivity,
and computes the deformation gradient for each element during simulation.

The deformation gradient :math:`\\mathbf{F}` maps from the reference (rest) configuration
to the current (deformed) configuration:

.. math::

    \\mathbf{F} = \\mathbf{D}_s \\mathbf{D}_m^{-1}

where :math:`\\mathbf{D}_s` contains edge vectors in the current configuration and
:math:`\\mathbf{D}_m` contains edge vectors in the rest configuration.
"""

import torch
import numpy as np
import meshio


class TetrahedralMesh:
    """
    Tetrahedral mesh for finite element simulation

    This class represents a volumetric mesh using tetrahedral elements. Each tetrahedron
    is defined by four vertices, and the mesh computes quantities needed for FEM simulation
    such as the deformation gradient, element volumes, and shape matrices.

    The mesh automatically computes and caches rest-state quantities when initialized:

    - :math:`\\mathbf{D}_m`: Rest shape matrix for each element (3×3)
    - :math:`\\mathbf{D}_m^{-1}`: Inverse rest shape matrix
    - :math:`V_0`: Rest volume for each element

    Attributes:
        vertices (torch.Tensor): Vertex positions :math:`(N \\times 3)`
        tetrahedra (torch.Tensor): Element connectivity :math:`(M \\times 4)` indices
        num_vertices (int): Number of vertices :math:`N`
        num_elements (int): Number of tetrahedral elements :math:`M`
        Dm (torch.Tensor): Rest shape matrices :math:`(M \\times 3 \\times 3)`
        Dm_inv (torch.Tensor): Inverse rest shape matrices :math:`(M \\times 3 \\times 3)`
        rest_volume (torch.Tensor): Rest volumes :math:`(M,)`
        device (torch.device): Device where tensors are stored
    """

    def __init__(self, vertices, tetrahedra, device="cpu"):
        """
        Initialize tetrahedral mesh

        Args:
            vertices: :math:`(N, 3)` array of vertex positions
            tetrahedra: :math:`(M, 4)` array of tetrahedron indices
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)

        # Convert to torch tensors
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).float()
        if isinstance(tetrahedra, np.ndarray):
            tetrahedra = torch.from_numpy(tetrahedra).long()

        self.vertices = vertices.to(self.device)
        self.tetrahedra = tetrahedra.to(self.device)

        self.num_vertices = self.vertices.shape[0]
        self.num_elements = self.tetrahedra.shape[0]

        # Compute reference shape matrices (Dm in standard FEM notation)
        self._compute_rest_state()

    def _compute_rest_state(self):
        """Compute rest state quantities for each element"""
        # Get vertices for each tetrahedron
        v0 = self.vertices[self.tetrahedra[:, 0]]  # (M, 3)
        v1 = self.vertices[self.tetrahedra[:, 1]]
        v2 = self.vertices[self.tetrahedra[:, 2]]
        v3 = self.vertices[self.tetrahedra[:, 3]]

        # Compute edge vectors in reference configuration
        # Dm = [v1-v0, v2-v0, v3-v0]
        self.Dm = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1)  # (M, 3, 3)

        # Compute inverse of Dm for each element (robust to degeneracy)
        # Use pseudo-inverse to avoid crashes on singular elements
        self.Dm_inv = torch.linalg.pinv(self.Dm)  # (M, 3, 3)

        # Compute rest volume for each tetrahedron
        # Volume = 1/6 * |det(Dm)|
        self.rest_volume = torch.abs(torch.det(self.Dm)) / 6.0  # (M,)

    def compute_deformation_gradient(self, current_vertices):
        """
        Compute deformation gradient F for each element

        Args:
            current_vertices: :math:`(N, 3)` current vertex positions

        Returns:
            F: :math:`(M, 3, 3)` deformation gradient for each element
        """
        # Get current vertices for each tetrahedron
        v0 = current_vertices[self.tetrahedra[:, 0]]
        v1 = current_vertices[self.tetrahedra[:, 1]]
        v2 = current_vertices[self.tetrahedra[:, 2]]
        v3 = current_vertices[self.tetrahedra[:, 3]]

        # Compute current edge vectors
        # Ds = [v1-v0, v2-v0, v3-v0]
        Ds = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1)  # (M, 3, 3)

        # Deformation gradient F = Ds * Dm_inv
        F = torch.bmm(Ds, self.Dm_inv)  # (M, 3, 3)

        return F

    @classmethod
    def from_file(cls, filename, device="cpu"):
        """Load mesh from file using meshio"""
        mesh = meshio.read(filename)

        # Extract vertices
        vertices = mesh.points

        # Extract tetrahedra
        tetrahedra = None
        for cell_block in mesh.cells:
            if cell_block.type == "tetra":
                tetrahedra = cell_block.data
                break

        if tetrahedra is None:
            raise ValueError(f"No tetrahedral elements found in {filename}")

        return cls(vertices, tetrahedra, device)

    @classmethod
    def create_cube(cls, resolution=5, size=1.0, device="cpu"):
        """
        Create a cube mesh with tetrahedral elements

        Args:
            resolution: number of divisions along each axis
            size: size of the cube
            device: 'cpu' or 'cuda'
        """
        # Create grid of points
        n = resolution
        x = np.linspace(0, size, n)
        y = np.linspace(0, size, n)
        z = np.linspace(0, size, n)

        # Create vertices
        vertices = []
        vertex_map = {}
        idx = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    vertices.append([x[i], y[j], z[k]])
                    vertex_map[(i, j, k)] = idx
                    idx += 1

        vertices = np.array(vertices)

        # Create tetrahedra (5 tets per cube)
        tetrahedra = []
        for i in range(n - 1):
            for j in range(n - 1):
                for k in range(n - 1):
                    # Get 8 corners of cube
                    v000 = vertex_map[(i, j, k)]
                    v001 = vertex_map[(i, j, k + 1)]
                    v010 = vertex_map[(i, j + 1, k)]
                    v011 = vertex_map[(i, j + 1, k + 1)]
                    v100 = vertex_map[(i + 1, j, k)]
                    v101 = vertex_map[(i + 1, j, k + 1)]
                    v110 = vertex_map[(i + 1, j + 1, k)]
                    v111 = vertex_map[(i + 1, j + 1, k + 1)]

                    # Subdivide cube into 5 tetrahedra
                    tetrahedra.append([v000, v001, v011, v111])
                    tetrahedra.append([v000, v001, v101, v111])
                    tetrahedra.append([v000, v100, v101, v111])
                    tetrahedra.append([v000, v100, v110, v111])
                    tetrahedra.append([v000, v010, v011, v111])

        tetrahedra = np.array(tetrahedra)

        return cls(vertices, tetrahedra, device)
