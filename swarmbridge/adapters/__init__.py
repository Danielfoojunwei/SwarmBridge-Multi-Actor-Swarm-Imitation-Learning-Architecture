"""
SwarmBridge Adapters

Thin adapters for external services:
- Federated Learning Service (replaces direct OpenFL usage)
- Registry Service (CSA upload/download)
- Edge Platform (runtime execution)
"""

from .federated_adapter import FederatedLearningAdapter
from .registry_adapter import RegistryAdapter
from .runtime_adapter import EdgePlatformRuntimeAdapter

__all__ = [
    "FederatedLearningAdapter",
    "RegistryAdapter",
    "EdgePlatformRuntimeAdapter",
]
