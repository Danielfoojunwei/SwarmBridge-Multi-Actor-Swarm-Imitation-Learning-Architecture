"""
SwarmBridge: Multi-Actor Demonstration Capture & Cooperative Imitation Learning

Focused Responsibilities:
1. Multi-actor demonstration capture (ROS 2)
2. Cooperative imitation learning (training)
3. Skill artifact packaging (CSA)
4. Registry publishing

Delegated to External Systems:
- Runtime execution → Edge Platform (Dynamical API)
- Federated learning orchestration → Federated Learning Service
- Mission orchestration → SwarmBrain
"""

from .pipeline import SwarmBridgePipeline
from .adapters import FederatedLearningAdapter, RegistryAdapter
from .schemas import SharedRoleSchema, CoordinationPrimitives

__all__ = [
    "SwarmBridgePipeline",
    "FederatedLearningAdapter",
    "RegistryAdapter",
    "SharedRoleSchema",
    "CoordinationPrimitives",
]

__version__ = "2.0.0"  # Refactored version
