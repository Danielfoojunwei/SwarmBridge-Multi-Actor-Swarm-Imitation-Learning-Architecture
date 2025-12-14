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

# Always available: schemas (no external dependencies)
from .schemas import SharedRoleSchema, CoordinationPrimitives

# Optional imports (gracefully handle missing dependencies)
try:
    from .pipeline import SwarmBridgePipeline
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    SwarmBridgePipeline = None

try:
    from .adapters import FederatedLearningAdapter, RegistryAdapter, EdgePlatformRuntimeAdapter
    _ADAPTERS_AVAILABLE = True
except ImportError:
    _ADAPTERS_AVAILABLE = False
    FederatedLearningAdapter = None
    RegistryAdapter = None
    EdgePlatformRuntimeAdapter = None

__all__ = [
    "SharedRoleSchema",
    "CoordinationPrimitives",
]

if _PIPELINE_AVAILABLE:
    __all__.append("SwarmBridgePipeline")

if _ADAPTERS_AVAILABLE:
    __all__.extend(["FederatedLearningAdapter", "RegistryAdapter", "EdgePlatformRuntimeAdapter"])

__version__ = "2.0.0"  # Refactored version
