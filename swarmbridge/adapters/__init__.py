"""
SwarmBridge Adapters

Thin adapters for external services:
- Federated Learning Service (replaces direct OpenFL usage)
- Registry Service (CSA upload/download)
- Edge Platform (runtime execution)
"""

from .federated_adapter import FederatedLearningAdapter, MockFederatedLearningAdapter
from .registry_adapter import RegistryAdapter, MockRegistryAdapter
from .runtime_adapter import EdgePlatformRuntimeAdapter, MockEdgePlatformRuntimeAdapter

__all__ = [
    "FederatedLearningAdapter",
    "RegistryAdapter",
    "EdgePlatformRuntimeAdapter",
    "MockFederatedLearningAdapter",
    "MockRegistryAdapter",
    "MockEdgePlatformRuntimeAdapter",
]
