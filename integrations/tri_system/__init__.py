"""
Tri-System Integration

Unified orchestration layer connecting:
- Dynamical-SIL (training)
- Edge Platform (deployment)
- SwarmBrain (orchestration)

Provides end-to-end workflow from skill training to mission execution.
"""

from .coordinator.unified_coordinator import TriSystemCoordinator
from .encryption.unified_encryption import UnifiedEncryptionBridge
from .config.tri_system_config import TriSystemConfig

__all__ = [
    "TriSystemCoordinator",
    "UnifiedEncryptionBridge",
    "TriSystemConfig",
]
