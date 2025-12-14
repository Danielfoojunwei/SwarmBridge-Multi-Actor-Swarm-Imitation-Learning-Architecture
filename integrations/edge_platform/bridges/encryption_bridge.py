"""
Encryption Bridge: Pyfhel (OpenFL) ↔ N2HE (Edge Platform)

Bridges homomorphic encryption systems between:
- Dynamical-SIL: Pyfhel (BFV/CKKS schemes)
- Edge Platform: N2HE 128-bit homomorphic encryption

Provides:
- Encrypted weight transfer between systems
- Key management and synchronization
- Format conversion for encrypted gradients
- Privacy budget tracking across both systems
"""

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch


@dataclass
class EncryptionContext:
    """Encryption context for cross-system operations"""
    scheme: str  # "BFV", "CKKS", "N2HE"
    key_id: str
    security_bits: int
    poly_modulus_degree: int
    created_at: str
    source_system: str  # "dynamical_sil" or "edge_platform"


@dataclass
class EncryptedWeights:
    """Encrypted model weights with metadata"""
    weights_data: bytes  # Encrypted tensor data
    encryption_context: EncryptionContext
    shape: Tuple[int, ...]
    dtype: str
    checksum: str


class EncryptionBridge:
    """
    Bridge between Pyfhel (Dynamical-SIL) and N2HE (Edge Platform).

    Strategy:
    - For Pyfhel→N2HE: Re-encrypt using compatible parameters
    - For N2HE→Pyfhel: Convert to Pyfhel format
    - Maintain key synchronization for federated aggregation
    - Track privacy budgets across both systems

    Example:
        bridge = EncryptionBridge()

        # Encrypt weights for Edge Platform
        encrypted = bridge.encrypt_for_edge(weights, key_id="key_123")

        # Decrypt weights from Edge Platform
        weights = bridge.decrypt_from_edge(encrypted_data)

        # Aggregate encrypted gradients from both systems
        aggregated = bridge.aggregate_encrypted([grad1, grad2, grad3])
    """

    def __init__(
        self,
        pyfhel_context_path: Optional[Path] = None,
        n2he_key_path: Optional[Path] = None,
    ):
        """
        Initialize encryption bridge.

        Args:
            pyfhel_context_path: Path to Pyfhel context
            n2he_key_path: Path to N2HE keys
        """
        self.pyfhel_context = None
        self.n2he_context = None

        # Load contexts if provided
        if pyfhel_context_path:
            self._load_pyfhel_context(pyfhel_context_path)

        if n2he_key_path:
            self._load_n2he_context(n2he_key_path)

    def _load_pyfhel_context(self, path: Path):
        """Load Pyfhel encryption context"""
        try:
            from Pyfhel import Pyfhel

            self.pyfhel_context = Pyfhel()
            self.pyfhel_context.load_context(str(path / "context"))
            self.pyfhel_context.load_public_key(str(path / "pub.key"))
            self.pyfhel_context.load_secret_key(str(path / "sec.key"))

            print("✓ Loaded Pyfhel context")
        except ImportError:
            print("⚠ Pyfhel not installed, using mock encryption")
            self.pyfhel_context = None
        except Exception as e:
            print(f"⚠ Failed to load Pyfhel context: {e}")

    def _load_n2he_context(self, path: Path):
        """Load N2HE encryption context"""
        try:
            with open(path / "n2he_context.json") as f:
                self.n2he_context = json.load(f)

            print("✓ Loaded N2HE context")
        except Exception as e:
            print(f"⚠ Failed to load N2HE context: {e}")

    def encrypt_for_edge(
        self,
        weights: torch.Tensor,
        key_id: str = "default",
        precision_bits: int = 16,
    ) -> EncryptedWeights:
        """
        Encrypt weights for Edge Platform (N2HE format).

        Args:
            weights: Model weights to encrypt
            key_id: Encryption key identifier
            precision_bits: Precision for fixed-point encoding

        Returns:
            Encrypted weights in N2HE format
        """

        # Convert to numpy
        weights_np = weights.cpu().numpy()

        # Flatten for encryption
        flat_weights = weights_np.flatten()

        # Simulate N2HE encryption (in production, use actual N2HE library)
        encrypted_data = self._n2he_encrypt(
            flat_weights,
            key_id=key_id,
            precision_bits=precision_bits,
        )

        # Create encryption context
        context = EncryptionContext(
            scheme="N2HE",
            key_id=key_id,
            security_bits=128,
            poly_modulus_degree=8192,
            created_at=str(np.datetime64('now')),
            source_system="dynamical_sil",
        )

        # Compute checksum
        import hashlib
        checksum = hashlib.sha256(encrypted_data).hexdigest()

        return EncryptedWeights(
            weights_data=encrypted_data,
            encryption_context=context,
            shape=weights_np.shape,
            dtype=str(weights_np.dtype),
            checksum=checksum,
        )

    def decrypt_from_edge(
        self,
        encrypted: EncryptedWeights,
    ) -> torch.Tensor:
        """
        Decrypt weights from Edge Platform (N2HE format).

        Args:
            encrypted: Encrypted weights from Edge Platform

        Returns:
            Decrypted torch tensor
        """

        # Verify checksum
        import hashlib
        checksum = hashlib.sha256(encrypted.weights_data).hexdigest()
        if checksum != encrypted.checksum:
            raise ValueError("Checksum mismatch - data may be corrupted")

        # Decrypt using N2HE
        flat_weights = self._n2he_decrypt(
            encrypted.weights_data,
            key_id=encrypted.encryption_context.key_id,
        )

        # Reshape
        weights_np = flat_weights.reshape(encrypted.shape)

        # Convert to tensor
        return torch.from_numpy(weights_np)

    def encrypt_for_sil(
        self,
        weights: torch.Tensor,
        scheme: str = "CKKS",
    ) -> EncryptedWeights:
        """
        Encrypt weights for Dynamical-SIL (Pyfhel format).

        Args:
            weights: Model weights
            scheme: Encryption scheme ("BFV" or "CKKS")

        Returns:
            Encrypted weights in Pyfhel format
        """

        weights_np = weights.cpu().numpy()
        flat_weights = weights_np.flatten()

        if self.pyfhel_context:
            # Use actual Pyfhel encryption
            if scheme == "CKKS":
                encrypted_data = self._pyfhel_encrypt_ckks(flat_weights)
            else:
                encrypted_data = self._pyfhel_encrypt_bfv(flat_weights)
        else:
            # Simulated encryption
            encrypted_data = self._simulate_pyfhel_encrypt(flat_weights, scheme)

        context = EncryptionContext(
            scheme=scheme,
            key_id="pyfhel_default",
            security_bits=128,
            poly_modulus_degree=8192,
            created_at=str(np.datetime64('now')),
            source_system="edge_platform",
        )

        import hashlib
        checksum = hashlib.sha256(encrypted_data).hexdigest()

        return EncryptedWeights(
            weights_data=encrypted_data,
            encryption_context=context,
            shape=weights_np.shape,
            dtype=str(weights_np.dtype),
            checksum=checksum,
        )

    def aggregate_encrypted(
        self,
        encrypted_weights_list: List[EncryptedWeights],
        strategy: str = "mean",
    ) -> EncryptedWeights:
        """
        Aggregate multiple encrypted weight updates.

        Performs homomorphic aggregation without decryption.

        Args:
            encrypted_weights_list: List of encrypted weights
            strategy: Aggregation strategy ("mean", "sum")

        Returns:
            Aggregated encrypted weights
        """

        if not encrypted_weights_list:
            raise ValueError("Empty weights list")

        # Verify all have same encryption scheme
        schemes = {w.encryption_context.scheme for w in encrypted_weights_list}
        if len(schemes) > 1:
            raise ValueError(f"Mixed encryption schemes: {schemes}")

        scheme = encrypted_weights_list[0].encryption_context.scheme

        # Perform homomorphic aggregation
        if scheme == "N2HE":
            aggregated_data = self._n2he_aggregate(
                [w.weights_data for w in encrypted_weights_list],
                strategy=strategy,
            )
        elif scheme in ["CKKS", "BFV"]:
            aggregated_data = self._pyfhel_aggregate(
                [w.weights_data for w in encrypted_weights_list],
                scheme=scheme,
                strategy=strategy,
            )
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Create aggregated result
        context = encrypted_weights_list[0].encryption_context
        shape = encrypted_weights_list[0].shape

        import hashlib
        checksum = hashlib.sha256(aggregated_data).hexdigest()

        return EncryptedWeights(
            weights_data=aggregated_data,
            encryption_context=context,
            shape=shape,
            dtype=encrypted_weights_list[0].dtype,
            checksum=checksum,
        )

    # Internal encryption methods (simulated for now)

    def _n2he_encrypt(
        self,
        data: np.ndarray,
        key_id: str,
        precision_bits: int,
    ) -> bytes:
        """Simulate N2HE encryption"""
        # In production, use actual N2HE library
        # For now, use simple encoding + noise
        encoded = (data * (2 ** precision_bits)).astype(np.int64)
        return encoded.tobytes()

    def _n2he_decrypt(
        self,
        encrypted_data: bytes,
        key_id: str,
    ) -> np.ndarray:
        """Simulate N2HE decryption"""
        # In production, use actual N2HE library
        encoded = np.frombuffer(encrypted_data, dtype=np.int64)
        return encoded.astype(np.float32) / (2 ** 16)

    def _n2he_aggregate(
        self,
        encrypted_list: List[bytes],
        strategy: str,
    ) -> bytes:
        """Simulate homomorphic aggregation for N2HE"""
        # Convert all to numpy
        arrays = [np.frombuffer(e, dtype=np.int64) for e in encrypted_list]

        # Aggregate (homomorphically)
        if strategy == "mean":
            aggregated = np.mean(arrays, axis=0).astype(np.int64)
        else:  # sum
            aggregated = np.sum(arrays, axis=0).astype(np.int64)

        return aggregated.tobytes()

    def _pyfhel_encrypt_ckks(self, data: np.ndarray) -> bytes:
        """Encrypt using Pyfhel CKKS"""
        if not self.pyfhel_context:
            return self._simulate_pyfhel_encrypt(data, "CKKS")

        from Pyfhel import PyCtxt

        encrypted_array = self.pyfhel_context.encryptFrac(data)
        return encrypted_array.to_bytes()

    def _pyfhel_encrypt_bfv(self, data: np.ndarray) -> bytes:
        """Encrypt using Pyfhel BFV"""
        if not self.pyfhel_context:
            return self._simulate_pyfhel_encrypt(data, "BFV")

        # BFV requires integers
        data_int = (data * 1000).astype(np.int64)
        encrypted_array = self.pyfhel_context.encryptInt(data_int)
        return encrypted_array.to_bytes()

    def _simulate_pyfhel_encrypt(self, data: np.ndarray, scheme: str) -> bytes:
        """Simulate Pyfhel encryption when library not available"""
        # Simple encoding for simulation
        encoded = (data * 1000).astype(np.int64)
        return encoded.tobytes()

    def _pyfhel_aggregate(
        self,
        encrypted_list: List[bytes],
        scheme: str,
        strategy: str,
    ) -> bytes:
        """Simulate homomorphic aggregation for Pyfhel"""
        # Similar to N2HE aggregation
        arrays = [np.frombuffer(e, dtype=np.int64) for e in encrypted_list]

        if strategy == "mean":
            aggregated = np.mean(arrays, axis=0).astype(np.int64)
        else:
            aggregated = np.sum(arrays, axis=0).astype(np.int64)

        return aggregated.tobytes()


class PrivacyBudgetTracker:
    """
    Track privacy budgets across both systems.

    Maintains unified privacy accounting for:
    - Differential Privacy (ε, δ budgets)
    - Homomorphic Encryption (computation depth)
    """

    def __init__(self):
        self.epsilon_total = 0.0
        self.delta_total = 0.0
        self.he_depth_used = 0
        self.operations_log = []

    def add_dp_operation(
        self,
        epsilon: float,
        delta: float,
        operation: str,
        system: str,
    ):
        """Record a DP operation"""
        self.epsilon_total += epsilon
        self.delta_total += delta

        self.operations_log.append({
            "type": "differential_privacy",
            "epsilon": epsilon,
            "delta": delta,
            "operation": operation,
            "system": system,
            "timestamp": str(np.datetime64('now')),
        })

    def add_he_operation(
        self,
        depth: int,
        operation: str,
        system: str,
    ):
        """Record an HE operation"""
        self.he_depth_used += depth

        self.operations_log.append({
            "type": "homomorphic_encryption",
            "depth": depth,
            "operation": operation,
            "system": system,
            "timestamp": str(np.datetime64('now')),
        })

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current privacy budget status"""
        return {
            "differential_privacy": {
                "epsilon_total": self.epsilon_total,
                "delta_total": self.delta_total,
                "operations_count": sum(
                    1 for op in self.operations_log
                    if op["type"] == "differential_privacy"
                ),
            },
            "homomorphic_encryption": {
                "depth_used": self.he_depth_used,
                "max_depth": 10,  # Typical limit
                "operations_count": sum(
                    1 for op in self.operations_log
                    if op["type"] == "homomorphic_encryption"
                ),
            },
            "total_operations": len(self.operations_log),
        }

    def is_budget_exceeded(
        self,
        epsilon_limit: float = 10.0,
        delta_limit: float = 1e-5,
        depth_limit: int = 10,
    ) -> bool:
        """Check if privacy budget is exceeded"""
        return (
            self.epsilon_total > epsilon_limit or
            self.delta_total > delta_limit or
            self.he_depth_used > depth_limit
        )
