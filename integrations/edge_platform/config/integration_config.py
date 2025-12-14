"""
Integration Configuration

Unified configuration for connecting Dynamical-SIL with Edge Platform.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class SystemEndpoints:
    """API endpoints for each system"""
    sil_registry: str = "http://localhost:8000"
    sil_coordinator: str = "http://localhost:8001"
    edge_platform: str = "http://jetson-orin.local:8001"
    edge_device_discovery: str = "http://jetson-orin.local:8001/api/devices"


@dataclass
class EncryptionConfig:
    """Encryption configuration"""
    sil_scheme: str = "CKKS"  # BFV, CKKS
    edge_scheme: str = "N2HE"
    security_bits: int = 128
    poly_modulus_degree: int = 8192
    enable_cross_system_encryption: bool = True


@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    num_sil_sites: int = 3
    num_edge_devices: int = 2
    rounds_per_sync: int = 5  # How many local rounds before sync
    aggregation_strategy: str = "weighted_average"  # weighted_average, median, trimmed_mean
    sil_weight: float = 0.5
    edge_weight: float = 0.5
    privacy_mode: str = "encrypted"  # encrypted, differential_privacy, hybrid


@dataclass
class SyncConfig:
    """Synchronization configuration"""
    auto_sync: bool = True
    sync_interval_seconds: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay_seconds: int = 10
    enable_bidirectional: bool = True


@dataclass
class IntegrationConfig:
    """Complete integration configuration"""
    endpoints: SystemEndpoints = field(default_factory=SystemEndpoints)
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    auth_token: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "IntegrationConfig":
        """Load configuration from YAML file"""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            endpoints=SystemEndpoints(**data.get("endpoints", {})),
            encryption=EncryptionConfig(**data.get("encryption", {})),
            federated=FederatedConfig(**data.get("federated", {})),
            sync=SyncConfig(**data.get("sync", {})),
            auth_token=data.get("auth_token"),
        )

    def to_yaml(self, path: Path):
        """Save configuration to YAML file"""
        data = {
            "endpoints": self.endpoints.__dict__,
            "encryption": self.encryption.__dict__,
            "federated": self.federated.__dict__,
            "sync": self.sync.__dict__,
            "auth_token": self.auth_token,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


def create_default_config() -> IntegrationConfig:
    """Create default integration configuration"""
    return IntegrationConfig()
