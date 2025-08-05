"""
DiffSim - A minimal differentiable physics simulator
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
