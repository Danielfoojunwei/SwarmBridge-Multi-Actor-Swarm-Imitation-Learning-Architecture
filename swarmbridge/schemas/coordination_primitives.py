"""
Coordination Primitives Library

Shared coordination primitives used across all systems.

Primitives:
- Handover: Object transfer between actors
- Mutex: Exclusive access to shared resource
- Barrier: Synchronization point for all actors
- Rendezvous: Meeting at specific location
- Formation: Maintain geometric configuration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class CoordinationType(Enum):
    """Standard coordination types"""
    HANDOVER = "handover"
    MUTEX = "mutex"
    BARRIER = "barrier"
    RENDEZVOUS = "rendezvous"
    FORMATION = "formation"
    COLLABORATIVE_MANIPULATION = "collaborative_manipulation"


@dataclass
class CoordinationPrimitive:
    """Unified coordination primitive definition"""
    coordination_type: CoordinationType
    participating_roles: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 10.0
    retry_attempts: int = 3


class CoordinationPrimitives:
    """
    Library of shared coordination primitives.

    Example:
        primitives = CoordinationPrimitives()

        # Get handover primitive
        handover = primitives.get_primitive(
            CoordinationType.HANDOVER,
            roles=["giver", "receiver"],
        )

        # Convert to SwarmBrain task graph
        task_graph = primitives.to_swarmbrain_task_graph(handover)
    """

    @staticmethod
    def get_primitive(
        coordination_type: CoordinationType,
        roles: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> CoordinationPrimitive:
        """
        Get coordination primitive definition.

        Args:
            coordination_type: Type of coordination
            roles: Participating roles
            parameters: Custom parameters

        Returns:
            Coordination primitive
        """

        default_params = CoordinationPrimitives._get_default_parameters(coordination_type)

        if parameters:
            default_params.update(parameters)

        return CoordinationPrimitive(
            coordination_type=coordination_type,
            participating_roles=roles,
            parameters=default_params,
            timeout_s=default_params.get("timeout_s", 10.0),
            retry_attempts=default_params.get("retry_attempts", 3),
        )

    @staticmethod
    def _get_default_parameters(coordination_type: CoordinationType) -> Dict[str, Any]:
        """Get default parameters for coordination type"""

        defaults = {
            CoordinationType.HANDOVER: {
                "handover_location": "midpoint",
                "object_id": "object_0",
                "grasp_force": 5.0,
                "transfer_speed": 0.1,
            },
            CoordinationType.MUTEX: {
                "resource_id": "shared_workspace",
                "max_lock_time_s": 30.0,
            },
            CoordinationType.BARRIER: {
                "sync_location": "barrier_point",
                "tolerance_m": 0.05,
            },
            CoordinationType.RENDEZVOUS: {
                "meeting_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                "arrival_tolerance_m": 0.1,
            },
            CoordinationType.FORMATION: {
                "formation_type": "line",
                "spacing_m": 0.5,
                "maintain_orientation": True,
            },
            CoordinationType.COLLABORATIVE_MANIPULATION: {
                "object_id": "object_0",
                "force_distribution": "equal",
                "sync_frequency_hz": 100,
            },
        }

        return defaults.get(coordination_type, {})

    @staticmethod
    def to_swarmbrain_task_graph(primitive: CoordinationPrimitive) -> Dict[str, Any]:
        """
        Convert coordination primitive to SwarmBrain task graph.

        Returns:
            Task graph structure
        """

        task_graph = {
            "coordination_type": primitive.coordination_type.value,
            "tasks": [],
        }

        if primitive.coordination_type == CoordinationType.HANDOVER:
            task_graph["tasks"] = [
                {
                    "task_id": "approach",
                    "assigned_roles": [primitive.participating_roles[0]],
                    "dependencies": [],
                },
                {
                    "task_id": "handover_sync",
                    "assigned_roles": primitive.participating_roles,
                    "dependencies": ["approach"],
                    "coordination": {
                        "type": "rendezvous",
                        "timeout_s": primitive.timeout_s,
                    },
                },
                {
                    "task_id": "transfer",
                    "assigned_roles": [primitive.participating_roles[1]],
                    "dependencies": ["handover_sync"],
                },
            ]
        elif primitive.coordination_type == CoordinationType.BARRIER:
            task_graph["tasks"] = [
                {
                    "task_id": "parallel_work",
                    "assigned_roles": primitive.participating_roles,
                    "dependencies": [],
                },
                {
                    "task_id": "sync_point",
                    "assigned_roles": primitive.participating_roles,
                    "dependencies": ["parallel_work"],
                    "coordination": {
                        "type": "barrier",
                        "timeout_s": primitive.timeout_s,
                    },
                },
            ]

        return task_graph

    @staticmethod
    def to_csa_coordination_encoder_config(primitive: CoordinationPrimitive) -> Dict[str, Any]:
        """
        Convert coordination primitive to CSA coordination encoder config.

        Returns:
            Coordination encoder configuration
        """

        return {
            "coordination_type": primitive.coordination_type.value,
            "encoder_type": "transformer",  # or "rnn", "mlp"
            "latent_dim": 64,
            "num_actors": len(primitive.participating_roles),
            "parameters": primitive.parameters,
        }

    @staticmethod
    def validate_primitive(primitive: CoordinationPrimitive) -> tuple[bool, str]:
        """
        Validate coordination primitive.

        Returns:
            (is_valid, error_message)
        """

        if not primitive.participating_roles:
            return False, "No participating roles specified"

        if primitive.coordination_type == CoordinationType.HANDOVER:
            if len(primitive.participating_roles) != 2:
                return False, "Handover requires exactly 2 roles"

        if primitive.timeout_s <= 0:
            return False, "Timeout must be positive"

        return True, "Valid"
