"""
Shared Role Schema

Unified role definitions used across all three systems:
- SwarmBridge: Training role-conditioned policies
- Edge Platform: MoE expert specialization
- SwarmBrain: Robot role assignment

Ensures compatibility and interoperability.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class RoleType(Enum):
    """Standard role types"""
    LEADER = "leader"
    FOLLOWER = "follower"
    OBSERVER = "observer"
    SUPPORTER = "supporter"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


@dataclass
class RoleDefinition:
    """
    Unified role definition.

    Compatible with:
    - CSA role configs
    - MoE expert specializations
    - SwarmBrain robot roles
    """
    role_id: str
    role_type: RoleType
    capabilities: List[str] = field(default_factory=list)
    observation_space: Dict[str, int] = field(default_factory=dict)
    action_space: Dict[str, int] = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    can_coordinate_with: List[str] = field(default_factory=list)

    # System-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class SharedRoleSchema:
    """
    Manages shared role schema across systems.

    Example:
        schema = SharedRoleSchema()

        # Define roles for multi-actor skill
        roles = schema.create_role_set(
            num_actors=2,
            coordination_type="handover",
        )

        # roles = [
        #     RoleDefinition(role_id="leader", role_type=RoleType.LEADER, ...),
        #     RoleDefinition(role_id="follower", role_type=RoleType.FOLLOWER, ...)
        # ]
    """

    @staticmethod
    def create_role_set(
        num_actors: int,
        coordination_type: str,
        custom_roles: Optional[List[RoleDefinition]] = None,
    ) -> List[RoleDefinition]:
        """
        Create standard role set for multi-actor skill.

        Args:
            num_actors: Number of actors
            coordination_type: Coordination primitive
            custom_roles: Optional custom role definitions

        Returns:
            List of role definitions
        """

        if custom_roles:
            return custom_roles

        # Standard role sets based on num_actors and coordination
        if coordination_type == "handover":
            if num_actors == 2:
                return [
                    RoleDefinition(
                        role_id="giver",
                        role_type=RoleType.LEADER,
                        capabilities=["grasp", "handover"],
                        priority=1,
                        can_coordinate_with=["receiver"],
                    ),
                    RoleDefinition(
                        role_id="receiver",
                        role_type=RoleType.FOLLOWER,
                        capabilities=["receive", "grasp"],
                        priority=0,
                        can_coordinate_with=["giver"],
                    ),
                ]
        elif coordination_type == "collaborative_manipulation":
            if num_actors == 2:
                return [
                    RoleDefinition(
                        role_id="left_arm",
                        role_type=RoleType.LEADER,
                        capabilities=["grasp", "stabilize"],
                        priority=1,
                        can_coordinate_with=["right_arm"],
                    ),
                    RoleDefinition(
                        role_id="right_arm",
                        role_type=RoleType.FOLLOWER,
                        capabilities=["grasp", "manipulate"],
                        priority=1,
                        can_coordinate_with=["left_arm"],
                    ),
                ]

        # Default: leader-follower hierarchy
        roles = []
        for i in range(num_actors):
            if i == 0:
                role_type = RoleType.LEADER
                priority = num_actors - 1
            else:
                role_type = RoleType.FOLLOWER
                priority = num_actors - i - 1

            roles.append(
                RoleDefinition(
                    role_id=f"actor_{i}",
                    role_type=role_type,
                    capabilities=["grasp", "manipulate"],
                    priority=priority,
                    can_coordinate_with=[f"actor_{j}" for j in range(num_actors) if j != i],
                )
            )

        return roles

    @staticmethod
    def to_csa_format(role: RoleDefinition) -> Dict[str, Any]:
        """Convert to CSA role config format"""
        return {
            "role_id": role.role_id,
            "capabilities": role.capabilities,
            "observation_dim": sum(role.observation_space.values()) if role.observation_space else 15,
            "action_dim": sum(role.action_space.values()) if role.action_space else 7,
            "metadata": {
                "role_type": role.role_type.value,
                "priority": role.priority,
                **role.metadata,
            },
        }

    @staticmethod
    def to_moe_format(role: RoleDefinition) -> Dict[str, Any]:
        """Convert to MoE expert format"""
        return {
            "expert_id": role.role_id,
            "specialization": role.role_type.value,
            "capabilities": role.capabilities,
            "priority": role.priority,
        }

    @staticmethod
    def to_swarmbrain_format(role: RoleDefinition) -> Dict[str, Any]:
        """Convert to SwarmBrain role format"""
        return {
            "role_id": role.role_id,
            "role_type": role.role_type.value,
            "capabilities": role.capabilities,
            "coordination_partners": role.can_coordinate_with,
            "priority": role.priority,
        }
