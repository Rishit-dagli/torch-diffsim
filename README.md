https://github.com/user-attachments/assets/e7d80ce7-af74-4deb-bb5a-aa3f93bb7a6d

# ⚙️ torch-diffsim

![PyPI](https://img.shields.io/pypi/v/torch-diffsim?style=flat-square)

Documentation is hosted at: https://rishit-dagli.github.io/torch-diffsim/

torch-diffsim is an extremely minimal parallelizable differentiable finite element (FEM) simulator written entirely in PyTorch. It uses a semi-implicit (symplectic Euler) integrator, a stable Neo-Hookean material model, and smooth barrier-based contact. All operations preserve gradients to enable optimization of materials and states.

## Install


```bash
pip install torch-diffsim

# or from source
git clone https://github.com/Rishit-dagli/torch-diffsim
cd torch-diffsim
pip install -e .
```

## Quick start (standard simulation)

```python
from diffsim import TetrahedralMesh, StableNeoHookean, SemiImplicitSolver, Simulator

# Load the Stanford Bunny tetrahedral mesh (path in this repository)
mesh = TetrahedralMesh.from_file("assets/tetmesh/bunny0.msh")

material = StableNeoHookean(youngs_modulus=1e5, poissons_ratio=0.45)
solver = SemiImplicitSolver(dt=0.01, damping=0.99, substeps=4)
sim = Simulator(mesh, material, solver)

for _ in range(200):
    sim.step()
```

https://github.com/user-attachments/assets/8e4c1e52-6baf-4806-b8d7-9f87f74c1af5

## Quick start (differentiable simulation)

```python
import torch
from diffsim import TetrahedralMesh
from diffsim.diff_physics import DifferentiableMaterial
from diffsim.diff_simulator import DifferentiableSolver, DifferentiableSimulator

device = "cuda" if torch.cuda.is_available() else "cpu"
mesh = TetrahedralMesh.create_cube(resolution=3, size=0.5, device=device)
mesh._compute_rest_state()

material = DifferentiableMaterial(youngs_modulus=1e5, poissons_ratio=0.4, requires_grad=True).to(device)
solver = DifferentiableSolver(dt=0.01, damping=0.99, substeps=4)
sim = DifferentiableSimulator(mesh, material, solver, device=device)

# Example loss: pull center of mass toward a target
target = torch.tensor([0.0, -0.3, 0.0], device=device)
for _ in range(20):
    sim.step()
loss = (sim.positions.mean(dim=0) - target).pow(2).sum()
loss.backward()
```

## Learn more

- User Guide and API: https://torch-diffsim.github.io
- Examples: see the `examples/` directory
- How it works: simulation and differentiation details in the docs

## Citation

If you use torch-diffsim in academic work, please include a citation or link to the project repository.

```bibtex
@misc{torch-diffsim,
  title  = {torch-diffsim},
  author = {Rishit Dagli},
  year   = {2025},
  howpublished = {\url{https://github.com/Rishit-dagli/torch-diffsim}}
}
```


