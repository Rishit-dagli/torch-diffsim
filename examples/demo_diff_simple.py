"""
Simple demo showing differentiable simulation basics

Demonstrates:
1. Gradient flow through energy computation
2. Quick material parameter optimization
3. Loss computation and backpropagation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from diffsim import TetrahedralMesh
from diffsim.diff_physics import DifferentiableMaterial
from diffsim.diff_simulator import DifferentiableSolver, DifferentiableSimulator


def main():
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("SIMPLE DIFFERENTIABLE SIMULATION DEMO")
    print("=" * 70)
    print(f"Device: {device}\n")

    # =========================================================================
    # Part 1: Energy and gradients
    # =========================================================================
    print("[1/3] Testing energy computation and gradients...")

    mesh = TetrahedralMesh.create_cube(resolution=3, size=0.5, device=device)
    mesh._compute_rest_state()

    # Learnable material
    material = DifferentiableMaterial(1e5, 0.4, requires_grad=True).to(device)

    # Deform mesh
    positions = mesh.vertices.clone()
    center = positions.mean(dim=0)
    positions = center + 1.2 * (positions - center)  # 20% expansion
    positions.requires_grad = True

    # Compute energy
    F = mesh.compute_deformation_gradient(positions)
    energy_density = material.energy_density(F)
    total_energy = torch.sum(energy_density * mesh.rest_volume)

    # Backward pass
    total_energy.backward()

    print(f"   Elastic energy: {total_energy.item():.4f} J")
    print(f"   ∂E/∂E_young: {material.E.grad.item():.6e}")
    print(f"   ∂E/∂ν: {material.nu.grad.item():.6e}")
    print(f"   ∂E/∂x norm: {positions.grad.norm().item():.4e}")
    print("   ✓ Gradients flowing correctly!\n")

    # =========================================================================
    # Part 2: Material parameter optimization
    # =========================================================================
    print("[2/3] Testing material parameter optimization...")

    # Create target
    true_E = 8e4
    true_material = DifferentiableMaterial(true_E, 0.4, requires_grad=False).to(device)
    true_solver = DifferentiableSolver(dt=0.01, gravity=-9.8, damping=0.98, substeps=2)
    true_sim = DifferentiableSimulator(mesh, true_material, true_solver, device=device)

    for _ in range(20):
        true_sim.step()
    target = true_sim.positions.detach()

    print(f"   Target material: E = {true_E:.2e} Pa")

    # Optimize
    learned_material = DifferentiableMaterial(2e5, 0.4, requires_grad=True).to(device)
    learned_solver = DifferentiableSolver(
        dt=0.01, gravity=-9.8, damping=0.98, substeps=2
    )
    learned_sim = DifferentiableSimulator(
        mesh, learned_material, learned_solver, device=device
    )

    optimizer = torch.optim.Adam([learned_material.E], lr=5e3)

    print(f"   Initial guess: E = {learned_material.E.item():.2e} Pa\n")
    print("   Optimizing...")

    for i in range(50):
        optimizer.zero_grad()
        learned_sim.reset()

        for _ in range(20):
            learned_sim.step()

        loss = torch.mean((learned_sim.positions - target) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            learned_material.E.clamp_(1e4, 5e5)

        if i % 10 == 0:
            error = abs(learned_material.E.item() - true_E) / true_E * 100
            print(
                f"      Iter {i:2d}: Loss = {loss.item():.6f}, E = {learned_material.E.item():.2e}, Error = {error:.1f}%"
            )

    final_error = abs(learned_material.E.item() - true_E) / true_E * 100
    print(
        f"\n   Final E: {learned_material.E.item():.2e} Pa (error: {final_error:.1f}%)"
    )
    print("   ✓ Optimization converged!\n")

    # =========================================================================
    # Part 3: Different loss functions
    # =========================================================================
    print("[3/3] Testing different loss functions...")

    # MSE loss
    mse_loss = torch.mean((learned_sim.positions - target) ** 2)
    print(f"   MSE loss: {mse_loss.item():.6e}")

    # L1 loss
    l1_loss = torch.mean(torch.abs(learned_sim.positions - target))
    print(f"   L1 loss: {l1_loss.item():.6e}")

    # Vertex-wise distance
    distances = torch.norm(learned_sim.positions - target, dim=1)
    max_distance = distances.max()
    mean_distance = distances.mean()
    print(f"   Max vertex distance: {max_distance.item():.6e}")
    print(f"   Mean vertex distance: {mean_distance.item():.6e}")


if __name__ == "__main__":
    main()
