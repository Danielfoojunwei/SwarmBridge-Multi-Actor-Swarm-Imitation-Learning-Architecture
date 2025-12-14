"""
Unified Encryption Bridge

Bridges three homomorphic encryption systems:
- Pyfhel (Dynamical-SIL): BFV/CKKS schemes
- N2HE (Edge Platform): 128-bit HE
- OpenFHE (SwarmBrain): BFV/BGV/CKKS schemes

Provides unified interface for cross-system encrypted operations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

import torch
import numpy as np


class EncryptionScheme(Enum):
    """Supported encryption schemes across all systems"""
    PYFHEL_BFV = "pyfhel_bfv"
    PYFHEL_CKKS = "pyfhel_ckks"
    N2HE_128 = "n2he_128"
    OPENFHE_BFV = "openfhe_bfv"
    OPENFHE_BGV = "openfhe_bgv"
    OPENFHE_CKKS = "openfhe_ckks"


@dataclass
class UnifiedEncryptionContext:
    """Unified encryption context for all three systems"""
    source_system: str  # dynamical_sil, edge_platform, swarmbrain
    scheme: EncryptionScheme
    security_bits: int
    poly_modulus_degree: int
    key_id: str


class UnifiedEncryptionBridge:
    """
    Unified encryption bridge for tri-system integration.

    Handles encryption/decryption and format conversion between:
    - Pyfhel (Dynamical-SIL)
    - N2HE (Edge Platform)
    - OpenFHE (SwarmBrain)

    Example:
        bridge = UnifiedEncryptionBridge()

        # Encrypt for any system
        encrypted_sil = bridge.encrypt(weights, target_system="dynamical_sil")
        encrypted_edge = bridge.encrypt(weights, target_system="edge_platform")
        encrypted_swarm = bridge.encrypt(weights, target_system="swarmbrain")

        # Cross-system aggregation
        aggregated = bridge.aggregate_cross_system([
            encrypted_sil,
            encrypted_edge,
            encrypted_swarm,
        ])

        # Decrypt result
        weights = bridge.decrypt(aggregated)
    """

    def __init__(self):
        self.contexts: Dict[str, UnifiedEncryptionContext] = {}

    def encrypt(
        self,
        data: torch.Tensor,
        target_system: str,
        scheme: Optional[EncryptionScheme] = None,
    ) -> Dict[str, Any]:
        """
        Encrypt data for target system.

        Args:
            data: Tensor to encrypt
            target_system: "dynamical_sil", "edge_platform", or "swarmbrain"
            scheme: Optional specific scheme to use

        Returns:
            Encrypted data package with metadata
        """

        # Select default scheme for target system
        if scheme is None:
            if target_system == "dynamical_sil":
                scheme = EncryptionScheme.PYFHEL_CKKS
            elif target_system == "edge_platform":
                scheme = EncryptionScheme.N2HE_128
            elif target_system == "swarmbrain":
                scheme = EncryptionScheme.OPENFHE_BFV
            else:
                raise ValueError(f"Unknown target system: {target_system}")

        # Convert to numpy
        data_np = data.cpu().numpy()

        # Encrypt based on scheme
        if scheme in [EncryptionScheme.PYFHEL_BFV, EncryptionScheme.PYFHEL_CKKS]:
            encrypted_data = self._encrypt_pyfhel(data_np, scheme)
        elif scheme == EncryptionScheme.N2HE_128:
            encrypted_data = self._encrypt_n2he(data_np)
        elif scheme in [EncryptionScheme.OPENFHE_BFV, EncryptionScheme.OPENFHE_BGV, EncryptionScheme.OPENFHE_CKKS]:
            encrypted_data = self._encrypt_openfhe(data_np, scheme)
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")

        # Create context
        context = UnifiedEncryptionContext(
            source_system=target_system,
            scheme=scheme,
            security_bits=128,
            poly_modulus_degree=8192,
            key_id=f"{target_system}_key",
        )

        return {
            "encrypted_data": encrypted_data,
            "context": context,
            "shape": data_np.shape,
            "dtype": str(data_np.dtype),
        }

    def decrypt(
        self,
        encrypted_package: Dict[str, Any],
    ) -> torch.Tensor:
        """Decrypt data from any system"""

        context = encrypted_package["context"]
        encrypted_data = encrypted_package["encrypted_data"]
        shape = encrypted_package["shape"]

        # Decrypt based on scheme
        scheme = context.scheme

        if scheme in [EncryptionScheme.PYFHEL_BFV, EncryptionScheme.PYFHEL_CKKS]:
            data_np = self._decrypt_pyfhel(encrypted_data, scheme)
        elif scheme == EncryptionScheme.N2HE_128:
            data_np = self._decrypt_n2he(encrypted_data)
        elif scheme in [EncryptionScheme.OPENFHE_BFV, EncryptionScheme.OPENFHE_BGV, EncryptionScheme.OPENFHE_CKKS]:
            data_np = self._decrypt_openfhe(encrypted_data, scheme)
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")

        # Reshape and convert to tensor
        data_np = data_np.reshape(shape)
        return torch.from_numpy(data_np)

    def aggregate_cross_system(
        self,
        encrypted_packages: List[Dict[str, Any]],
        strategy: str = "mean",
    ) -> Dict[str, Any]:
        """
        Aggregate encrypted data from multiple systems.

        Handles mixed encryption schemes by:
        1. Converting to common scheme (OpenFHE CKKS)
        2. Performing homomorphic aggregation
        3. Returning in common format

        Args:
            encrypted_packages: List of encrypted packages from different systems
            strategy: Aggregation strategy ("mean", "sum")

        Returns:
            Aggregated encrypted package
        """

        if not encrypted_packages:
            raise ValueError("Empty packages list")

        print(f"Aggregating {len(encrypted_packages)} encrypted packages from multiple systems")

        # Extract schemes
        schemes = [pkg["context"].scheme for pkg in encrypted_packages]
        systems = [pkg["context"].source_system for pkg in encrypted_packages]

        print(f"  Systems: {set(systems)}")
        print(f"  Schemes: {set(s.value for s in schemes)}")

        # For demonstration, convert all to OpenFHE CKKS (most compatible)
        converted_packages = []

        for pkg in encrypted_packages:
            if pkg["context"].scheme != EncryptionScheme.OPENFHE_CKKS:
                # Decrypt and re-encrypt in target scheme
                data = self.decrypt(pkg)
                converted = self.encrypt(
                    data,
                    target_system="swarmbrain",
                    scheme=EncryptionScheme.OPENFHE_CKKS,
                )
                converted_packages.append(converted)
            else:
                converted_packages.append(pkg)

        # Perform homomorphic aggregation
        encrypted_arrays = [pkg["encrypted_data"] for pkg in converted_packages]
        aggregated_data = self._homomorphic_aggregate(
            encrypted_arrays,
            strategy=strategy,
            scheme=EncryptionScheme.OPENFHE_CKKS,
        )

        # Return aggregated package
        return {
            "encrypted_data": aggregated_data,
            "context": UnifiedEncryptionContext(
                source_system="tri_system_aggregated",
                scheme=EncryptionScheme.OPENFHE_CKKS,
                security_bits=128,
                poly_modulus_degree=8192,
                key_id="aggregated_key",
            ),
            "shape": converted_packages[0]["shape"],
            "dtype": converted_packages[0]["dtype"],
        }

    # Internal encryption methods (simulated for now)

    def _encrypt_pyfhel(self, data: np.ndarray, scheme: EncryptionScheme) -> bytes:
        """Encrypt using Pyfhel (Dynamical-SIL)"""
        # In production, use actual Pyfhel library
        encoded = (data * 1000).astype(np.int64)
        return encoded.tobytes()

    def _decrypt_pyfhel(self, encrypted: bytes, scheme: EncryptionScheme) -> np.ndarray:
        """Decrypt using Pyfhel"""
        decoded = np.frombuffer(encrypted, dtype=np.int64)
        return decoded.astype(np.float32) / 1000

    def _encrypt_n2he(self, data: np.ndarray) -> bytes:
        """Encrypt using N2HE (Edge Platform)"""
        # In production, use actual N2HE library
        encoded = (data * (2 ** 16)).astype(np.int64)
        return encoded.tobytes()

    def _decrypt_n2he(self, encrypted: bytes) -> np.ndarray:
        """Decrypt using N2HE"""
        decoded = np.frombuffer(encrypted, dtype=np.int64)
        return decoded.astype(np.float32) / (2 ** 16)

    def _encrypt_openfhe(self, data: np.ndarray, scheme: EncryptionScheme) -> bytes:
        """Encrypt using OpenFHE (SwarmBrain)"""
        # In production, use actual OpenFHE library
        if scheme == EncryptionScheme.OPENFHE_CKKS:
            # CKKS supports floating point
            encoded = (data * 1000).astype(np.int64)
        else:
            # BFV/BGV require integers
            encoded = (data * 1000).astype(np.int64)

        return encoded.tobytes()

    def _decrypt_openfhe(self, encrypted: bytes, scheme: EncryptionScheme) -> np.ndarray:
        """Decrypt using OpenFHE"""
        decoded = np.frombuffer(encrypted, dtype=np.int64)
        return decoded.astype(np.float32) / 1000

    def _homomorphic_aggregate(
        self,
        encrypted_list: List[bytes],
        strategy: str,
        scheme: EncryptionScheme,
    ) -> bytes:
        """Perform homomorphic aggregation"""

        # Convert all to arrays
        arrays = [np.frombuffer(e, dtype=np.int64) for e in encrypted_list]

        # Aggregate (homomorphically - addition supported by all schemes)
        if strategy == "mean":
            aggregated = np.mean(arrays, axis=0).astype(np.int64)
        else:  # sum
            aggregated = np.sum(arrays, axis=0).astype(np.int64)

        return aggregated.tobytes()


class TriSystemPrivacyBudgetTracker:
    """
    Track privacy budgets across all three systems.

    Maintains unified accounting for:
    - Differential Privacy (ε, δ)
    - Homomorphic Encryption (computation depth)
    - System-specific privacy metrics
    """

    def __init__(self):
        self.budgets = {
            "dynamical_sil": {"epsilon": 0.0, "delta": 0.0, "he_depth": 0},
            "edge_platform": {"epsilon": 0.0, "delta": 0.0, "he_depth": 0},
            "swarmbrain": {"epsilon": 0.0, "delta": 0.0, "he_depth": 0},
        }
        self.operations_log = []

    def add_operation(
        self,
        system: str,
        operation_type: str,  # "dp" or "he"
        epsilon: float = 0.0,
        delta: float = 0.0,
        he_depth: int = 0,
    ):
        """Record a privacy operation"""

        self.budgets[system]["epsilon"] += epsilon
        self.budgets[system]["delta"] += delta
        self.budgets[system]["he_depth"] += he_depth

        self.operations_log.append({
            "system": system,
            "type": operation_type,
            "epsilon": epsilon,
            "delta": delta,
            "he_depth": he_depth,
            "timestamp": str(np.datetime64('now')),
        })

    def get_total_budget(self) -> Dict[str, Any]:
        """Get total privacy budget across all systems"""

        return {
            "total_epsilon": sum(b["epsilon"] for b in self.budgets.values()),
            "total_delta": sum(b["delta"] for b in self.budgets.values()),
            "total_he_depth": max(b["he_depth"] for b in self.budgets.values()),
            "per_system": self.budgets,
            "total_operations": len(self.operations_log),
        }

    def is_budget_exceeded(
        self,
        epsilon_limit: float = 10.0,
        delta_limit: float = 1e-5,
        he_depth_limit: int = 10,
    ) -> bool:
        """Check if total privacy budget is exceeded"""

        total = self.get_total_budget()

        return (
            total["total_epsilon"] > epsilon_limit or
            total["total_delta"] > delta_limit or
            total["total_he_depth"] > he_depth_limit
        )
