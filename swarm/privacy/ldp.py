"""
Local Differential Privacy (LDP)

Implements edge-first privacy based on:
Zhao et al. (2020) "Local Differential Privacy-based Federated Learning for
Internet of Things"

Key idea: Add calibrated noise at each site before sending updates to coordinator.
"""

import torch
import numpy as np
from typing import Dict


class LocalDifferentialPrivacy:
    """
    Local Differential Privacy for federated learning

    Adds Laplace or Gaussian noise to model updates to achieve epsilon-LDP.
    """

    def __init__(self, mechanism: str = "laplace"):
        """
        Args:
            mechanism: Noise mechanism ("laplace" or "gaussian")
        """
        self.mechanism = mechanism

    def add_noise(
        self,
        weights: Dict[str, torch.Tensor],
        epsilon: float,
        sensitivity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Add LDP noise to model weights

        Args:
            weights: Model weights dictionary
            epsilon: Privacy budget
            sensitivity: L1/L2 sensitivity of the update

        Returns:
            Noisy weights
        """
        noisy_weights = {}

        for key, tensor in weights.items():
            if self.mechanism == "laplace":
                # Laplace mechanism: scale = sensitivity / epsilon
                scale = sensitivity / epsilon
                noise = torch.from_numpy(
                    np.random.laplace(0, scale, tensor.shape)
                ).float()
                noisy_weights[key] = tensor + noise

            elif self.mechanism == "gaussian":
                # Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
                # For simplicity, use delta = 1e-5
                delta = 1e-5
                sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
                noise = torch.randn_like(tensor) * sigma
                noisy_weights[key] = tensor + noise

            else:
                raise ValueError(f"Unknown mechanism: {self.mechanism}")

        return noisy_weights

    def clip_and_add_noise(
        self,
        weights: Dict[str, torch.Tensor],
        epsilon: float,
        clip_norm: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Clip weights and add LDP noise

        Args:
            weights: Model weights
            epsilon: Privacy budget
            clip_norm: Clipping threshold

        Returns:
            Clipped and noisy weights
        """
        # Compute L2 norm
        total_norm = torch.sqrt(
            sum(torch.sum(w ** 2) for w in weights.values())
        ).item()

        # Clip
        clipped_weights = {}
        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-8)
            for key, tensor in weights.items():
                clipped_weights[key] = tensor * scale
        else:
            clipped_weights = weights

        # Add noise (sensitivity = clip_norm after clipping)
        return self.add_noise(clipped_weights, epsilon, sensitivity=clip_norm)
