"""
torch-diffsim: A minimal differentiable physics simulator in PyTorch

This package provides a fully differentiable finite element method (FEM) simulator
using semi-implicit (symplectic Euler) time integration. All operations support
automatic differentiation for gradient-based optimization.

Main Components:
    - TetrahedralMesh: Tetrahedral mesh representation
    - StableNeoHookean: Stable Neo-Hookean hyperelastic material model
    - SemiImplicitSolver: Semi-implicit time integration solver
    - Simulator: Main simulator combining mesh, material, and solver

Differentiable Components:
    - DifferentiableMaterial: Learnable material parameters
    - SpatiallyVaryingMaterial: Per-element learnable material properties
    - DifferentiableSolver: Fully differentiable semi-implicit solver
    - DifferentiableSimulator: Differentiable simulator with gradient support
    - DifferentiableBarrierContact: Smooth differentiable contact handling
    - CheckpointedRollout: Memory-efficient gradient computation
    - ImplicitDifferentiation: Efficient gradients through implicit function theorem
"""

from .material import StableNeoHookean
from .solver import SemiImplicitSolver
from .mesh import TetrahedralMesh
from .simulator import Simulator

# Differentiable simulation
from .diff_physics import (
    DifferentiableBarrierContact,
    ImplicitDifferentiation,
    CheckpointedRollout,
    DifferentiableMaterial,
    SpatiallyVaryingMaterial,
    smooth_step,
    log_barrier,
)
from .diff_simulator import DifferentiableSolver, DifferentiableSimulator

__version__ = "0.1.0"
__all__ = [
    # Standard simulation
    "StableNeoHookean",
    "SemiImplicitSolver",
    "TetrahedralMesh",
    "Simulator",
    # Differentiable simulation
    "DifferentiableBarrierContact",
    "ImplicitDifferentiation",
    "CheckpointedRollout",
    "DifferentiableMaterial",
    "SpatiallyVaryingMaterial",
    "DifferentiableSolver",
    "DifferentiableSimulator",
    "smooth_step",
    "log_barrier",
]
