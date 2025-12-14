"""
CSA Cryptographic Signing and Verification

Provides digital signature support for CSA artifacts to ensure authenticity
and integrity during distribution.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class CSASigner:
    """Sign CSA artifacts with private key"""

    def __init__(self, private_key_path: Optional[Path] = None):
        """
        Initialize signer

        Args:
            private_key_path: Path to PEM-encoded RSA private key
                             If None, generates a new key pair
        """
        if private_key_path and private_key_path.exists():
            self.private_key = self._load_private_key(private_key_path)
        else:
            self.private_key = self._generate_key_pair()

        self.public_key = self.private_key.public_key()

    def sign_artifact(self, artifact_path: Path, signer_id: str) -> Path:
        """
        Sign CSA artifact and create detached signature

        Args:
            artifact_path: Path to CSA tarball
            signer_id: Identifier for signing entity (e.g., site name)

        Returns:
            Path to signature file (.sig)
        """
        # Compute artifact hash
        artifact_hash = self._compute_file_hash(artifact_path)

        # Create signature metadata
        sig_metadata = {
            "artifact_name": artifact_path.name,
            "artifact_hash": artifact_hash,
            "signer_id": signer_id,
            "signed_at": datetime.utcnow().isoformat(),
            "algorithm": "RSA-PSS-SHA256",
        }

        # Sign the metadata
        metadata_bytes = json.dumps(sig_metadata, sort_keys=True).encode("utf-8")
        signature = self.private_key.sign(
            metadata_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        # Save signature file
        sig_path = artifact_path.with_suffix(artifact_path.suffix + ".sig")
        sig_data = {
            "metadata": sig_metadata,
            "signature": signature.hex(),
            "public_key": self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("utf-8"),
        }

        with open(sig_path, "w") as f:
            json.dump(sig_data, f, indent=2)

        print(f"✓ Signed artifact: {artifact_path.name}")
        print(f"  Signature: {sig_path}")
        return sig_path

    def save_private_key(self, output_path: Path, password: Optional[bytes] = None) -> None:
        """Save private key to PEM file (optionally encrypted)"""
        encryption_algorithm = (
            serialization.BestAvailableEncryption(password)
            if password
            else serialization.NoEncryption()
        )

        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm,
        )

        with open(output_path, "wb") as f:
            f.write(pem)

        print(f"✓ Saved private key: {output_path}")

    def save_public_key(self, output_path: Path) -> None:
        """Save public key to PEM file"""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        with open(output_path, "wb") as f:
            f.write(pem)

        print(f"✓ Saved public key: {output_path}")

    @staticmethod
    def _generate_key_pair() -> rsa.RSAPrivateKey:
        """Generate new RSA key pair (4096-bit)"""
        return rsa.generate_private_key(
            public_exponent=65537, key_size=4096, backend=default_backend()
        )

    @staticmethod
    def _load_private_key(path: Path, password: Optional[bytes] = None) -> rsa.RSAPrivateKey:
        """Load RSA private key from PEM file"""
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password, default_backend())

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class CSAVerifier:
    """Verify CSA artifact signatures"""

    def verify_artifact(self, artifact_path: Path, signature_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Verify CSA artifact signature

        Args:
            artifact_path: Path to CSA tarball
            signature_path: Path to signature file (if None, looks for .sig file)

        Returns:
            (is_valid, message)
        """
        if signature_path is None:
            signature_path = Path(str(artifact_path) + ".sig")

        if not signature_path.exists():
            return False, f"Signature file not found: {signature_path}"

        try:
            # Load signature data
            with open(signature_path) as f:
                sig_data = json.load(f)

            metadata = sig_data["metadata"]
            signature = bytes.fromhex(sig_data["signature"])

            # Load public key from signature
            public_key = serialization.load_pem_public_key(
                sig_data["public_key"].encode("utf-8"), backend=default_backend()
            )

            # Verify artifact hash matches
            actual_hash = CSASigner._compute_file_hash(artifact_path)
            if actual_hash != metadata["artifact_hash"]:
                return False, f"Artifact hash mismatch: expected {metadata['artifact_hash']}, got {actual_hash}"

            # Verify signature
            metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
            public_key.verify(
                signature,
                metadata_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256(),
            )

            msg = (
                f"✓ Signature valid\n"
                f"  Signer: {metadata['signer_id']}\n"
                f"  Signed: {metadata['signed_at']}\n"
                f"  Hash: {actual_hash[:16]}..."
            )
            return True, msg

        except Exception as e:
            return False, f"Signature verification failed: {str(e)}"

    @staticmethod
    def load_public_key(path: Path) -> rsa.RSAPublicKey:
        """Load RSA public key from PEM file"""
        with open(path, "rb") as f:
            return serialization.load_pem_public_key(f.read(), backend=default_backend())
