"""
Homomorphic Encryption (HE) Wrapper

Wrapper around Pyfhel/OpenFHE for privacy-preserving validation
and encrypted aggregation.

Note: Full FHE computation is expensive. This is primarily for
encrypted validation/scoring before CSA acceptance.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from Pyfhel import Pyfhel, PyCtxt
    HE_AVAILABLE = True
except ImportError:
    HE_AVAILABLE = False
    print("Warning: Pyfhel not available. HE mode will not work.")


class HomomorphicEncryptionWrapper:
    """
    Homomorphic Encryption wrapper for federated learning

    Provides:
    - Weight encryption
    - Encrypted aggregation
    - Decryption
    """

    def __init__(self, context_params: Optional[Dict] = None):
        """
        Initialize HE context

        Args:
            context_params: HE context parameters (uses defaults if None)
        """
        if not HE_AVAILABLE:
            raise ImportError("Pyfhel not installed. Install with: pip install Pyfhel")

        self.HE = Pyfhel()

        # Default parameters (BFV scheme for integers, moderate security)
        if context_params is None:
            context_params = {
                "scheme": "BFV",
                "n": 2**13,  # Polynomial modulus degree
                "t_bits": 20,  # Plaintext modulus bits
                "sec": 128,  # Security level
            }

        self.HE.contextGen(**context_params)
        self.HE.keyGen()
        self.HE.relinKeyGen()

        print(f"✓ Initialized HE context: {context_params['scheme']}, sec={context_params['sec']}")

    def encrypt(self, weights: Dict[str, torch.Tensor]) -> Dict[str, List]:
        """
        Encrypt model weights

        Args:
            weights: Model weights dictionary

        Returns:
            Dictionary of encrypted weights (list of PyCtxt)
        """
        encrypted = {}

        for key, tensor in weights.items():
            # Flatten and convert to integers (scale for precision)
            scale = 1000  # Scale factor for fixed-point representation
            flat = (tensor.cpu().numpy().flatten() * scale).astype(np.int64)

            # Encrypt in batches (HE has polynomial degree limit)
            batch_size = self.HE.get_nSlots()
            encrypted_batches = []

            for i in range(0, len(flat), batch_size):
                batch = flat[i : i + batch_size]
                # Pad if needed
                if len(batch) < batch_size:
                    batch = np.pad(batch, (0, batch_size - len(batch)))

                # Encrypt
                ctxt = self.HE.encryptInt(batch.tolist())
                encrypted_batches.append(ctxt)

            encrypted[key] = {
                "ciphertexts": encrypted_batches,
                "shape": tensor.shape,
                "scale": scale,
            }

        return encrypted

    def decrypt(self, encrypted_weights: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """
        Decrypt encrypted weights

        Args:
            encrypted_weights: Dictionary of encrypted weights

        Returns:
            Decrypted weights as tensors
        """
        decrypted = {}

        for key, enc_data in encrypted_weights.items():
            # Decrypt batches
            flat_values = []
            for ctxt in enc_data["ciphertexts"]:
                batch = self.HE.decryptInt(ctxt)
                flat_values.extend(batch)

            # Convert back to float and reshape
            scale = enc_data["scale"]
            shape = enc_data["shape"]
            total_elements = np.prod(shape)

            flat_array = np.array(flat_values[:total_elements], dtype=np.float32) / scale
            tensor = torch.from_numpy(flat_array.reshape(shape))

            decrypted[key] = tensor

        return decrypted

    def aggregate_encrypted(
        self, encrypted_weights_list: List[Dict[str, List]]
    ) -> Dict[str, List]:
        """
        Aggregate encrypted weights (homomorphic addition)

        Args:
            encrypted_weights_list: List of encrypted weight dictionaries

        Returns:
            Aggregated encrypted weights
        """
        if not encrypted_weights_list:
            raise ValueError("No weights to aggregate")

        # Initialize with first set of weights
        aggregated = encrypted_weights_list[0]

        # Add remaining weights
        for enc_weights in encrypted_weights_list[1:]:
            for key in aggregated.keys():
                # Add ciphertexts element-wise
                for i in range(len(aggregated[key]["ciphertexts"])):
                    aggregated[key]["ciphertexts"][i] += enc_weights[key]["ciphertexts"][i]

        return aggregated

    def validate_encrypted(self, encrypted_weights: Dict[str, List]) -> float:
        """
        Perform privacy-preserving validation on encrypted weights

        This is a placeholder for encrypted computation.
        In practice, you'd implement specific validation logic.

        Args:
            encrypted_weights: Encrypted model weights

        Returns:
            Validation score (mock for now)
        """
        # Placeholder: count number of parameters
        total_params = sum(
            len(enc_data["ciphertexts"]) for enc_data in encrypted_weights.values()
        )

        # Mock validation score
        return float(total_params > 0)
