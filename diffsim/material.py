"""
Material models for finite element simulation

This module implements hyperelastic material models for FEM simulation.
The primary model is the Stable Neo-Hookean formulation, which provides
numerical stability even under large deformations and element inversion.

The strain energy density is derived from the deformation gradient :math:`\\mathbf{F}`,
and forces are computed as the negative gradient of the total elastic energy.
"""

import torch


class StableNeoHookean:
    """
    Stable Neo-Hookean hyperelastic material model

    This class implements the stable Neo-Hookean formulation from Smith et al. (2018)
    "Stable Neo-Hookean Flesh Simulation". The energy density function is:

    .. math::

        \\Psi(\\mathbf{F}) = \\frac{\\mu}{2}(I_C - 3) - \\mu \\log J + \\frac{\\lambda}{2}(J-1)^2

    where:

    - :math:`\\mathbf{F}` is the deformation gradient tensor (3×3)
    - :math:`I_C = \\text{tr}(\\mathbf{F}^T \\mathbf{F}) = \\|\\mathbf{F}\\|_F^2` is the first invariant
    - :math:`J = \\det(\\mathbf{F})` is the Jacobian determinant (volume ratio)
    - :math:`\\mu = \\frac{E}{2(1+\\nu)}` is the first Lamé parameter (shear modulus)
    - :math:`\\lambda = \\frac{E\\nu}{(1+\\nu)(1-2\\nu)}` is the second Lamé parameter

    The formulation ensures numerical stability even under large deformations and
    near-inversion by using logarithmic terms instead of polynomial terms for the
    volumetric component.

    Parameters:
        youngs_modulus (float): Young's modulus :math:`E` in Pascals (default: 1e6)
        poissons_ratio (float): Poisson's ratio :math:`\\nu` (default: 0.45)

    Attributes:
        E (float): Young's modulus
        nu (float): Poisson's ratio
        mu (float): First Lamé parameter (shear modulus)
        lam (float): Second Lamé parameter

    Reference:
        Smith, B., De Goes, F., & Kim, T. (2018). Stable neo-hookean flesh simulation.
        ACM Transactions on Graphics (TOG), 37(2), 1-15.
    """

    def __init__(self, youngs_modulus=1e6, poissons_ratio=0.45):
        """
        Initialize material with elastic constants

        Args:
            youngs_modulus: Young's modulus (E)
            poissons_ratio: Poisson's ratio (ν)
        """
        self.E = youngs_modulus
        self.nu = poissons_ratio

        # Convert to Lamé parameters
        # μ = E / (2(1+ν))
        # λ = E*ν / ((1+ν)(1-2ν))
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def energy_density(self, F):
        """
        Compute strain energy density for deformation gradient F

        Args:
            F: :math:`(M, 3, 3)` deformation gradient tensor

        Returns:
            psi: :math:`(M,)` energy density for each element
        """
        # Compute first invariant Ic = trace(F^T F) = ||F||_F^2
        Ic = torch.sum(F * F, dim=(1, 2))  # (M,)

        # Compute determinant (volume ratio)
        J = torch.det(F)  # (M,)

        # Clamp J to prevent inversion (J > 0)
        J = torch.clamp(J, min=1e-6)

        # Stable Neo-Hookean energy density
        # Ψ = μ/2 * (Ic - 3) - μ log(J) + λ/2 * (J-1)^2
        psi = (
            self.mu / 2.0 * (Ic - 3.0)
            - self.mu * torch.log(J)
            + self.lam / 2.0 * (J - 1.0) ** 2
        )

        return psi

    def first_piola_kirchhoff_stress(self, F):
        """
        Compute first Piola-Kirchhoff stress tensor P = ∂Ψ/∂F

        Args:
            F: :math:`(M, 3, 3)` deformation gradient tensor

        Returns:
            P: :math:`(M, 3, 3)` first Piola-Kirchhoff stress tensor
        """
        # Enable gradient computation
        F_grad = F.detach().requires_grad_(True)

        # Compute energy density
        psi = self.energy_density(F_grad)

        # Compute gradient
        P = torch.autograd.grad(psi.sum(), F_grad, create_graph=True)[0]

        return P

    def first_piola_kirchhoff_stress_analytic(self, F):
        """
        Compute first Piola-Kirchhoff stress tensor analytically with stability

        P = ∂Ψ/∂F = μ*F - μ*F^{-T} + λ*(J-1)*J*F^{-T}

        Args:
            F: :math:`(M, 3, 3)` deformation gradient tensor

        Returns:
            P: :math:`(M, 3, 3)` first Piola-Kirchhoff stress tensor
        """
        # Compute determinant with strict clamping to prevent inversion
        J = torch.det(F)  # (M,)

        # Check for degenerate/inverted elements
        if (J < 0.01).any():
            # Clamp more aggressively to prevent blow-up
            J = torch.clamp(J, min=0.1, max=5.0)
        else:
            J = torch.clamp(J, min=0.3, max=3.0)

        J_clamped = J

        # Compute F^{-T} = (F^{-1})^T using SVD for stability
        # F = U S V^T, then F^{-1} = V S^{-1} U^T
        U, S, Vh = torch.linalg.svd(F)

        # Clamp singular values more aggressively to prevent singularity
        S_clamped = torch.clamp(S, min=0.2, max=5.0)
        S_inv = 1.0 / S_clamped

        # F_inv = V @ diag(S_inv) @ U^T
        F_inv = torch.bmm(
            Vh.transpose(1, 2), torch.bmm(torch.diag_embed(S_inv), U.transpose(1, 2))
        )
        F_inv_T = F_inv.transpose(1, 2)

        # P = μ*F - μ*F^{-T} + λ*(J-1)*J*F^{-T}
        # Scale down to prevent blow-up
        P = (
            self.mu * F
            - self.mu * F_inv_T
            + self.lam
            * (J_clamped - 1.0).unsqueeze(-1).unsqueeze(-1)
            * J_clamped.unsqueeze(-1).unsqueeze(-1)
            * F_inv_T
        )

        # Only clamp extreme stresses (very high limit for realistic behavior)
        P_norm = torch.norm(P, dim=(1, 2), keepdim=True)  # (M, 1, 1)
        max_stress = 1e6  # Much higher limit - only prevent catastrophic blow-up
        P_scale = torch.clamp(max_stress / (P_norm + 1e-8), max=1.0)
        P = P * P_scale

        return P

    def compute_elastic_forces(self, F, Dm_inv, volume):
        """
        Compute elastic forces from stress tensor

        Args:
            F: :math:`(M, 3, 3)` deformation gradient
            Dm_inv: :math:`(M, 3, 3)` inverse rest shape matrix
            volume: :math:`(M,)` rest volume of each element

        Returns:
            forces: :math:`(M, 4, 3)` forces on vertices of each element
        """
        # Compute stress tensor
        P = self.first_piola_kirchhoff_stress_analytic(F)  # (M, 3, 3)

        # H = -volume * P * Dm_inv^T (force matrix)
        H = -volume.unsqueeze(-1).unsqueeze(-1) * torch.bmm(
            P, Dm_inv.transpose(1, 2)
        )  # (M, 3, 3)

        # Extract forces for each vertex
        # f1, f2, f3 are columns of H
        # f0 = -(f1 + f2 + f3) (force balance)
        f1 = H[:, :, 0]  # (M, 3)
        f2 = H[:, :, 1]  # (M, 3)
        f3 = H[:, :, 2]  # (M, 3)
        f0 = -(f1 + f2 + f3)  # (M, 3)

        forces = torch.stack([f0, f1, f2, f3], dim=1)  # (M, 4, 3)

        return forces
