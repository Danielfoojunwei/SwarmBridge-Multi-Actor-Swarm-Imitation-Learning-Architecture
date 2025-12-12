"""
Cooperative Imitation Learning Training Pipeline

Implements training workflows for multi-actor cooperative skills using:
- robomimic for behavior cloning foundations
- LeRobot for real-world robotics utilities
- Role-conditioned policy adapters
- Coordination latent encoders
"""

from .train_cooperative_bc import CooperativeBCTrainer
from .models import (
    RoleConditionedPolicy,
    CoordinationLatentEncoder,
    MultiActorDataset,
)
from .config import TrainingConfig

__all__ = [
    "CooperativeBCTrainer",
    "RoleConditionedPolicy",
    "CoordinationLatentEncoder",
    "MultiActorDataset",
    "TrainingConfig",
]
