"""
Dynamical Edge Platform Integration

Integrates Dynamical-SIL (multi-actor swarm learning) with the
Dynamical Edge Platform (skill-centric VLA models on edge devices).

Key Integration Points:
- CSA packages ↔ MoE skill experts
- OpenFL federated learning ↔ Edge encrypted aggregation
- Registry API ↔ Skills API
- Privacy mechanisms (Pyfhel HE ↔ N2HE)
"""

from .adapters.csa_to_moe import CSAToMoEAdapter
from .bridges.api_bridge import EdgePlatformAPIBridge
from .bridges.encryption_bridge import EncryptionBridge
from .sync.federated_sync import FederatedSyncService
from .config.integration_config import IntegrationConfig

__all__ = [
    "CSAToMoEAdapter",
    "EdgePlatformAPIBridge",
    "EncryptionBridge",
    "FederatedSyncService",
    "IntegrationConfig",
]
