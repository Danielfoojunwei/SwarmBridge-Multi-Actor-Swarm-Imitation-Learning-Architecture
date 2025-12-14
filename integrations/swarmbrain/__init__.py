"""
SwarmBrain Integration

Integrates Dynamical-SIL with SwarmBrain orchestration system.
Enables CSA skills to be deployed and orchestrated via SwarmBrain's
mission planning and federated learning framework.
"""

from .adapters.csa_to_swarmbrain import CSAToSwarmBrainAdapter
from .orchestration.mission_bridge import SwarmBrainMissionBridge

__all__ = [
    "CSAToSwarmBrainAdapter",
    "SwarmBrainMissionBridge",
]
