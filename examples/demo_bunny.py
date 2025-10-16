import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from diffsim import TetrahedralMesh, StableNeoHookean, SemiImplicitSolver, Simulator
from diffsim.visualizer import PolyscopeVisualizer


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading bunny mesh...")
    mesh_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "tetmesh", "bunny0.msh"
    )
    mesh = TetrahedralMesh.from_file(mesh_path, device=device)

    mesh.vertices *= 2.0

    min_y = mesh.vertices[:, 1].min()
    mesh.vertices[:, 1] -= min_y
    mesh.vertices[:, 1] += 2.0

    mesh._compute_rest_state()

    print(f"Mesh loaded: {mesh.num_vertices} vertices, {mesh.num_elements} elements")

    material = StableNeoHookean(youngs_modulus=5e5, poissons_ratio=0.4)

    solver = SemiImplicitSolver(
        dt=0.003,
        gravity=-9.8,
        damping=0.998,
        substeps=4,
        collision_method="simplified",
        enable_self_collision=False,
    )

    simulator = Simulator(
        mesh=mesh, material=material, solver=solver, density=1000.0, device=device
    )

    visualizer = PolyscopeVisualizer(simulator, "Stanford Bunny Drop")
    visualizer.run(steps_per_frame=2)


if __name__ == "__main__":
    main()
