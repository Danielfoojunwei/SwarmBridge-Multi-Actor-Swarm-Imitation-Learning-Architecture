"""
Privacy Engine

Unified interface for applying privacy mechanisms:
- Local Differential Privacy (LDP)
- Differential Privacy SGD (DP-SGD)
- Homomorphic Encryption (HE)
"""

from enum import Enum
from typing import Any, Dict, List, Optional

import torch

from .ldp import LocalDifferentialPrivacy
from .dp_sgd import DPSGDWrapper
from .he_wrapper import HomomorphicEncryptionWrapper


class PrivacyMode(str, Enum):
    """Supported privacy modes"""

    NONE = "none"
    LDP = "ldp"
    DP_SGD = "dp_sgd"
    HE = "he"
    FHE = "fhe"


class PrivacyEngine:
    """Unified privacy mechanism engine"""

    def __init__(self):
        self.ldp = LocalDifferentialPrivacy()
        self.dp_sgd = DPSGDWrapper()
        self.he = HomomorphicEncryptionWrapper()

    def apply_privacy(
        self,
        weights_list: List[Dict[str, torch.Tensor]],
        mode: PrivacyMode,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        clip_norm: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Apply privacy mechanism to model weights

        Args:
            weights_list: List of weight dictionaries
            mode: Privacy mode to apply
            epsilon: Privacy budget (for DP modes)
            delta: Privacy parameter (for DP modes)
            clip_norm: Gradient clipping norm
            **kwargs: Additional privacy parameters

        Returns:
            Privacy-protected weights
        """
        if mode == PrivacyMode.NONE:
            return weights_list

        elif mode == PrivacyMode.LDP:
            # Local Differential Privacy: Add noise locally before aggregation
            if epsilon is None:
                raise ValueError("epsilon required for LDP mode")
            return [
                self.ldp.add_noise(weights, epsilon=epsilon, sensitivity=clip_norm or 1.0)
                for weights in weights_list
            ]

        elif mode == PrivacyMode.DP_SGD:
            # DP-SGD: Clip and add noise
            if epsilon is None or delta is None:
                raise ValueError("epsilon and delta required for DP-SGD mode")
            return [
                self.dp_sgd.privatize(
                    weights, epsilon=epsilon, delta=delta, clip_norm=clip_norm or 1.0
                )
                for weights in weights_list
            ]

        elif mode in [PrivacyMode.HE, PrivacyMode.FHE]:
            # Homomorphic Encryption: Encrypt weights
            # Note: HE aggregation happens in encrypted space, decryption after
            encrypted_weights = [self.he.encrypt(weights) for weights in weights_list]
            # For now, return encrypted weights (coordinator would aggregate in encrypted space)
            return encrypted_weights

        else:
            raise ValueError(f"Unknown privacy mode: {mode}")

    def get_privacy_accountant(
        self, mode: PrivacyMode, **kwargs: Any
    ) -> Optional[Dict[str, float]]:
        """
        Get privacy accounting information

        Returns:
            Dictionary with privacy budget tracking
        """
        if mode == PrivacyMode.DP_SGD:
            return self.dp_sgd.get_privacy_spent(**kwargs)
        elif mode == PrivacyMode.LDP:
            # LDP budget is simpler (single-shot)
            return {"epsilon": kwargs.get("epsilon", 0.0), "delta": 0.0}
        else:
            return None
