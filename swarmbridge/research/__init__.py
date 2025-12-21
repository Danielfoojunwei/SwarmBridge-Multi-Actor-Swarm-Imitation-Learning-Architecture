"""
SwarmBridge Research Module

Novel research contributions for multi-actor swarm imitation learning:
1. Causal Coordination Discovery (CCD)
2. Temporal Coordination Credit Assignment (TCCA)
3. Hierarchical Multi-Actor IL
4. Privacy-Preserving Federated Learning
5. Active Demonstration Sampling
6. Cross-Embodiment Transfer
7. Counterfactual Reasoning

These modules contain academically novel algorithms not found in existing work.
"""

from .causal_coordination_discovery import (
    CausalCoordinationDiscovery,
    DirectedCoordinationGraph,
    StructuralCausalModel,
    CoordinationSkeleton,
)

__all__ = [
    "CausalCoordinationDiscovery",
    "DirectedCoordinationGraph",
    "StructuralCausalModel",
    "CoordinationSkeleton",
]
