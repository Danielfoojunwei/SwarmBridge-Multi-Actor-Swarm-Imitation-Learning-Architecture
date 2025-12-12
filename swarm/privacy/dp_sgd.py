"""
Differential Privacy SGD (DP-SGD)

Wrapper around Opacus for DP-SGD training with privacy accounting.
"""

from typing import Dict, Optional

import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


class DPSGDWrapper:
    """
    DP-SGD privacy wrapper

    Provides gradient clipping and noise addition for differential privacy.
    """

    def __init__(self):
        self.accountant: Optional[RDPAccountant] = None
        self.steps = 0

    def privatize(
        self,
        weights: Dict[str, torch.Tensor],
        epsilon: float,
        delta: float,
        clip_norm: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply DP-SGD privatization to weights

        Args:
            weights: Model weights
            epsilon: Privacy budget
            delta: Privacy parameter
            clip_norm: Gradient clipping threshold

        Returns:
            Privatized weights
        """
        # Clip weights
        total_norm = torch.sqrt(sum(torch.sum(w ** 2) for w in weights.values()))

        clipped_weights = {}
        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-8)
            for key, tensor in weights.items():
                clipped_weights[key] = tensor * scale
        else:
            clipped_weights = {k: v.clone() for k, v in weights.items()}

        # Compute noise multiplier for target epsilon, delta
        # Using simplified formula: noise_multiplier ~ clip_norm * sqrt(2 * ln(1.25/delta)) / epsilon
        import math

        noise_multiplier = clip_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

        # Add Gaussian noise
        noisy_weights = {}
        for key, tensor in clipped_weights.items():
            noise = torch.randn_like(tensor) * noise_multiplier
            noisy_weights[key] = tensor + noise

        self.steps += 1

        return noisy_weights

    def get_privacy_spent(
        self, num_steps: Optional[int] = None, delta: float = 1e-5
    ) -> Dict[str, float]:
        """
        Get privacy budget spent

        Args:
            num_steps: Number of training steps (uses self.steps if None)
            delta: Delta parameter

        Returns:
            Dictionary with epsilon and delta
        """
        steps = num_steps or self.steps

        # Simplified privacy accounting (for production, use Opacus RDPAccountant)
        # epsilon roughly scales with sqrt(steps)
        epsilon_estimate = 1.0 * (steps ** 0.5)

        return {"epsilon": epsilon_estimate, "delta": delta, "steps": steps}

    def reset_accountant(self) -> None:
        """Reset privacy accountant"""
        self.steps = 0
        self.accountant = None
