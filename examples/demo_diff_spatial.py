"""
Demo: Spatially Varying Material Optimization

Shows how to:
1. Optimize per-element material properties
2. Learn heterogeneous material distributions
3. Match target deformation patterns

Task: Design a spatially varying stiffness distribution to match a target shape
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt
import numpy as np
from diffsim import TetrahedralMesh
from diffsim.diff_physics import SpatiallyVaryingMaterial
from diffsim.diff_simulator import DifferentiableSolver, DifferentiableSimulator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 60)
    print("SPATIALLY VARYING MATERIAL OPTIMIZATION")
    print("=" * 60)

    # Create mesh
    print("\n1. Creating mesh...")
    mesh = TetrahedralMesh.create_cube(resolution=5, size=1.0, device=device)
    mesh.vertices[:, 1] += 0.5
    mesh._compute_rest_state()
    print(f"   Mesh: {mesh.num_vertices} vertices, {mesh.num_elements} elements")

    # Create target: stiffer on one side
    print("\n2. Creating target with heterogeneous material...")
    true_material = SpatiallyVaryingMaterial(
        mesh.num_elements, base_youngs=1e5, base_poisson=0.4
    ).to(device)

    # Make left side stiffer than right
    element_centers = torch.zeros((mesh.num_elements, 3), device=device)
    for i in range(mesh.num_elements):
        tet = mesh.tetrahedra[i]
        element_centers[i] = mesh.vertices[tet].mean(dim=0)

    x_pos = element_centers[:, 0]
    x_normalized = (x_pos - x_pos.min()) / (x_pos.max() - x_pos.min() + 1e-6)

    # Stiffness varies linearly: left=5x, right=1x
    with torch.no_grad():
        true_material.log_E[:] = torch.log(
            torch.tensor(1e5, device=device)
        ) + torch.log(1.0 + 4.0 * (1.0 - x_normalized))

    print(
        f"   E range: [{true_material.E.min().item():.2e}, {true_material.E.max().item():.2e}] Pa"
    )

    # Run target simulation
    true_solver = DifferentiableSolver(dt=0.005, gravity=-9.8, damping=0.98, substeps=4)
    true_sim = DifferentiableSimulator(mesh, true_material, true_solver, device=device)

    # Fix bottom
    min_y = true_sim.positions[:, 1].min()
    fixed = (true_sim.positions[:, 1] <= min_y + 0.08).nonzero(as_tuple=True)[0]
    true_sim.set_fixed_vertices(fixed)

    print("   Running target simulation...")
    for _ in range(50):
        true_sim.step()

    target_positions = true_sim.positions.detach().clone()
    target_E = true_material.E.detach().clone()

    # Now try to recover spatial distribution
    print("\n3. Optimizing spatially varying material...")
    learned_material = SpatiallyVaryingMaterial(
        mesh.num_elements,
        base_youngs=2e5,  # Start with uniform wrong guess
        base_poisson=0.4,
    ).to(device)

    learned_solver = DifferentiableSolver(
        dt=0.005, gravity=-9.8, damping=0.98, substeps=4
    )
    learned_sim = DifferentiableSimulator(
        mesh, learned_material, learned_solver, device=device
    )
    learned_sim.set_fixed_vertices(fixed)

    # Optimizer with regularization
    optimizer = torch.optim.Adam([learned_material.log_E], lr=0.01)

    print("   Iter |     Loss     | E_min/E_max  | Correlation")
    print("   " + "-" * 55)

    losses = []
    correlations = []

    for iter in range(200):
        optimizer.zero_grad()

        # Reset
        learned_sim.reset()

        # Forward
        for _ in range(50):
            learned_sim.step()

        # Loss: match deformation
        pos_loss = torch.mean((learned_sim.positions - target_positions) ** 2)

        # Regularization: spatial smoothness (Tikhonov)
        smoothness_loss = 0.0
        for i in range(mesh.num_elements - 1):
            smoothness_loss += (
                learned_material.log_E[i] - learned_material.log_E[i + 1]
            ) ** 2
        smoothness_loss = 1e-3 * smoothness_loss / mesh.num_elements

        loss = pos_loss + smoothness_loss

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        # Project to reasonable range
        with torch.no_grad():
            learned_material.log_E.clamp_(
                torch.log(torch.tensor(1e4, device=device)),
                torch.log(torch.tensor(1e7, device=device)),
            )

        # Log
        losses.append(loss.item())

        E_learned = learned_material.E.detach().cpu().numpy()
        E_target = target_E.cpu().numpy()
        correlation = np.corrcoef(E_learned, E_target)[0, 1]
        correlations.append(correlation)

        if iter % 20 == 0:
            print(
                f"   {iter:4d} | {loss.item():11.6f} | {learned_material.E.min().item():.2e}/{learned_material.E.max().item():.2e} | {correlation:.4f}"
            )

    # Results
    print("\n4. Results:")
    print(
        f"   Target E range:  [{target_E.min().item():.2e}, {target_E.max().item():.2e}] Pa"
    )
    print(
        f"   Learned E range: [{learned_material.E.min().item():.2e}, {learned_material.E.max().item():.2e}] Pa"
    )
    print(f"   Correlation: {correlations[-1]:.4f}")

    # Visualize
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    ax1.plot(losses)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Convergence")
    ax1.set_yscale("log")
    ax1.grid(True)

    ax2.plot(correlations)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Correlation with Target")
    ax2.set_title("Spatial Pattern Recovery")
    ax2.set_ylim([-1, 1])
    ax2.grid(True)

    # Scatter plot
    E_learned_final = learned_material.E.detach().cpu().numpy()
    E_target_final = target_E.cpu().numpy()
    ax3.scatter(E_target_final, E_learned_final, alpha=0.5)
    ax3.plot(
        [E_target_final.min(), E_target_final.max()],
        [E_target_final.min(), E_target_final.max()],
        "r--",
        label="Perfect match",
    )
    ax3.set_xlabel("Target E (Pa)")
    ax3.set_ylabel("Learned E (Pa)")
    ax3.set_title("Per-Element Stiffness")
    ax3.legend()
    ax3.grid(True)

    # Spatial distribution
    x_centers = element_centers[:, 0].cpu().numpy()
    sort_idx = np.argsort(x_centers)
    ax4.plot(
        x_centers[sort_idx], E_target_final[sort_idx], "r-", label="Target", linewidth=2
    )
    ax4.plot(
        x_centers[sort_idx],
        E_learned_final[sort_idx],
        "b--",
        label="Learned",
        linewidth=2,
    )
    ax4.set_xlabel("X Position")
    ax4.set_ylabel("Young's Modulus (Pa)")
    ax4.set_title("Spatial Stiffness Distribution")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("spatial_material_optimization.png", dpi=150)


if __name__ == "__main__":
    main()
