"""
Demo: Differentiable Material Parameter Identification

Shows how to:
1. Learn material parameters from observation
2. Backpropagate through simulation
3. Optimize Young's modulus and Poisson's ratio

Task: Given a target deformation, find the material parameters that produce it
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt
from diffsim import TetrahedralMesh
from diffsim.diff_physics import DifferentiableMaterial
from diffsim.diff_simulator import DifferentiableSolver, DifferentiableSimulator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 60)
    print("DIFFERENTIABLE MATERIAL PARAMETER IDENTIFICATION")
    print("=" * 60)

    # Create simple mesh
    print("\n1. Creating test mesh...")
    mesh = TetrahedralMesh.create_cube(resolution=4, size=0.5, device=device)
    mesh.vertices[:, 1] += 0.3
    mesh._compute_rest_state()
    print(f"   Mesh: {mesh.num_vertices} vertices, {mesh.num_elements} elements")

    # Ground truth material
    print("\n2. Creating ground truth simulation...")
    true_E = 1e5  # 100 kPa
    true_nu = 0.40
    print(f"   True material: E={true_E:.2e} Pa, ν={true_nu:.3f}")

    true_material = DifferentiableMaterial(true_E, true_nu, requires_grad=False)
    true_solver = DifferentiableSolver(dt=0.005, gravity=-9.8, damping=0.98, substeps=4)
    true_sim = DifferentiableSimulator(mesh, true_material, true_solver, device=device)

    # Fix bottom vertices
    min_y = true_sim.positions[:, 1].min()
    fixed = (true_sim.positions[:, 1] <= min_y + 0.05).nonzero(as_tuple=True)[0]
    true_sim.set_fixed_vertices(fixed)

    # Run forward to get target
    print("   Running forward simulation (30 steps)...")
    for _ in range(30):
        true_sim.step()

    target_positions = true_sim.positions.detach().clone()
    print(
        f"   Target displacement: {torch.norm(target_positions - mesh.vertices).item():.4f}"
    )

    # Now try to recover parameters from observation
    print("\n3. Setting up inverse problem...")
    # Start with wrong initial guess
    initial_E = 5e5  # 5x too stiff
    initial_nu = 0.30
    print(f"   Initial guess: E={initial_E:.2e} Pa, ν={initial_nu:.3f}")

    # Create learnable material
    learned_material = DifferentiableMaterial(initial_E, initial_nu, requires_grad=True)
    learned_solver = DifferentiableSolver(
        dt=0.005, gravity=-9.8, damping=0.98, substeps=4
    )
    learned_sim = DifferentiableSimulator(
        mesh, learned_material, learned_solver, device=device
    )
    learned_sim.set_fixed_vertices(fixed)

    # Optimizer
    optimizer = torch.optim.Adam([learned_material.E, learned_material.nu], lr=1e3)

    # Optimize
    print("\n4. Optimizing material parameters...")
    print("   Iter |     Loss     |      E       |      ν      | E_error  ")
    print("   " + "-" * 65)

    losses = []
    E_history = []
    nu_history = []

    for iter in range(100):
        optimizer.zero_grad()

        # Reset simulation
        learned_sim.reset()

        # Forward simulation
        for _ in range(30):
            learned_sim.step()

        # Compute loss
        loss = torch.mean((learned_sim.positions - target_positions) ** 2)

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        # Project to valid range
        with torch.no_grad():
            learned_material.E.clamp_(1e4, 1e7)
            learned_material.nu.clamp_(0.0, 0.49)

        # Log
        losses.append(loss.item())
        E_history.append(learned_material.E.item())
        nu_history.append(learned_material.nu.item())

        if iter % 10 == 0:
            E_error = abs(learned_material.E.item() - true_E) / true_E * 100
            print(
                f"   {iter:4d} | {loss.item():11.6f} | {learned_material.E.item():11.2e} | {learned_material.nu.item():10.4f} | {E_error:7.2f}%"
            )

    # Final results
    print("\n5. Results:")
    print(f"   True:    E={true_E:.2e} Pa, ν={true_nu:.3f}")
    print(
        f"   Learned: E={learned_material.E.item():.2e} Pa, ν={learned_material.nu.item():.3f}"
    )
    print(
        f"   Error:   E={(abs(learned_material.E.item() - true_E) / true_E * 100):.2f}%, "
        + f"ν={(abs(learned_material.nu.item() - true_nu) / true_nu * 100):.2f}%"
    )

    # Plot convergence
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    ax1.plot(losses)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Loss Convergence")
    ax1.set_yscale("log")
    ax1.grid(True)

    ax2.plot(E_history, label="Learned")
    ax2.axhline(true_E, color="r", linestyle="--", label="True")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Young's Modulus (Pa)")
    ax2.set_title("E Recovery")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(nu_history, label="Learned")
    ax3.axhline(true_nu, color="r", linestyle="--", label="True")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Poisson's Ratio")
    ax3.set_title("ν Recovery")
    ax3.legend()
    ax3.grid(True)

    # Show deformation comparison
    true_disp = torch.norm(target_positions - mesh.vertices, dim=1).cpu().numpy()
    learned_disp = (
        torch.norm(learned_sim.positions.detach() - mesh.vertices, dim=1).cpu().numpy()
    )
    ax4.scatter(true_disp, learned_disp, alpha=0.5)
    ax4.plot([0, true_disp.max()], [0, true_disp.max()], "r--", label="Perfect match")
    ax4.set_xlabel("True Displacement")
    ax4.set_ylabel("Learned Displacement")
    ax4.set_title("Displacement Comparison")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("material_identification.png", dpi=150)


if __name__ == "__main__":
    main()
