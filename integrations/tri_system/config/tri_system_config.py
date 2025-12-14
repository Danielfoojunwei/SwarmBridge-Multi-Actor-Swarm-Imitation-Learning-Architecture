"""
Tri-System Configuration

Unified configuration for all three systems:
- Dynamical-SIL
- Edge Platform
- SwarmBrain
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class SystemEndpoints:
    """API endpoints for all three systems"""
    # Dynamical-SIL
    sil_registry: str = "http://localhost:8000"
    sil_coordinator: str = "http://localhost:8001"

    # Edge Platform
    edge_platform: str = "http://jetson-orin.local:8002"

    # SwarmBrain
    swarmbrain_orchestrator: str = "http://localhost:8003"
    swarmbrain_robots: str = "http://localhost:8003/api/v1/robots"


@dataclass
class WorkflowConfig:
    """Workflow configuration"""
    # Training (Dynamical-SIL)
    sil_training_rounds: int = 5
    sil_num_sites: int = 3

    # Deployment (Edge Platform)
    edge_num_devices: int = 2

    # Execution (SwarmBrain)
    swarm_num_robots: int = 3
    swarm_coordination_type: str = "handover"  # handover, mutex, barrier, rendezvous

    # Timing
    mission_timeout_s: int = 600
    learning_round_timeout_s: int = 300


@dataclass
class EncryptionConfig:
    """Encryption configuration for all systems"""
    # Dynamical-SIL
    sil_scheme: str = "CKKS"  # BFV, CKKS

    # Edge Platform
    edge_scheme: str = "N2HE"

    # SwarmBrain
    swarm_scheme: str = "BFV"  # BFV, BGV, CKKS

    # Unified settings
    security_bits: int = 128
    poly_modulus_degree: int = 8192
    enable_cross_system_encryption: bool = True


@dataclass
class PrivacyConfig:
    """Privacy budget configuration"""
    epsilon_limit: float = 10.0
    delta_limit: float = 1e-5
    he_depth_limit: int = 10
    track_budgets: bool = True


@dataclass
class TriSystemConfig:
    """Complete tri-system configuration"""
    endpoints: SystemEndpoints = field(default_factory=SystemEndpoints)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    auth_token: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TriSystemConfig":
        """Load configuration from YAML file"""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            endpoints=SystemEndpoints(**data.get("endpoints", {})),
            workflow=WorkflowConfig(**data.get("workflow", {})),
            encryption=EncryptionConfig(**data.get("encryption", {})),
            privacy=PrivacyConfig(**data.get("privacy", {})),
            auth_token=data.get("auth_token"),
        )

    def to_yaml(self, path: Path):
        """Save configuration to YAML file"""
        data = {
            "endpoints": self.endpoints.__dict__,
            "workflow": self.workflow.__dict__,
            "encryption": self.encryption.__dict__,
            "privacy": self.privacy.__dict__,
            "auth_token": self.auth_token,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


def create_default_tri_system_config() -> TriSystemConfig:
    """Create default tri-system configuration"""
    return TriSystemConfig()
