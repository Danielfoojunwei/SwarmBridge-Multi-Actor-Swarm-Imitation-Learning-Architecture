"""
Cooperative Skill Artefact (CSA) Packaging and Management

This module provides the core schema and tools for packaging, signing, versioning,
and validating cooperative skill artefacts in the Dynamical-SIL system.
"""

from .schema import (
    CooperativeSkillArtefact,
    RoleConfig,
    PolicyAdapter,
    CoordinationEncoder,
    SafetyEnvelope,
    CSAMetadata,
)
from .packager import CSAPackager, CSALoader
from .signing import CSASigner, CSAVerifier
from .validator import CSAValidator

__all__ = [
    "CooperativeSkillArtefact",
    "RoleConfig",
    "PolicyAdapter",
    "CoordinationEncoder",
    "SafetyEnvelope",
    "CSAMetadata",
    "CSAPackager",
    "CSALoader",
    "CSASigner",
    "CSAVerifier",
    "CSAValidator",
]
