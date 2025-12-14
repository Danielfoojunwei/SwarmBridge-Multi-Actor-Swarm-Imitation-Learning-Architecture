"""
Shared Schemas

Consolidated role and coordination schemas used across:
- SwarmBridge (training)
- Edge Platform (deployment)
- SwarmBrain (orchestration)

Ensures compatibility across all systems.
"""

from .role_schema import SharedRoleSchema, RoleDefinition
from .coordination_primitives import CoordinationPrimitives, CoordinationType

__all__ = [
    "SharedRoleSchema",
    "RoleDefinition",
    "CoordinationPrimitives",
    "CoordinationType",
]
