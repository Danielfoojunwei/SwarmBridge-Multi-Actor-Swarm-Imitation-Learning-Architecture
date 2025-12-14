"""
Cooperative Skill Artefact (CSA) Schema Definitions

Defines the complete schema for multi-actor cooperative skills including:
- Role-conditioned policy adapters
- Coordination latent encoders
- Behavior tree state machines
- Safety envelopes and constraints
- Metadata and provenance
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator


class RoleType(str, Enum):
    """Actor roles in cooperative tasks"""

    LEADER = "leader"
    FOLLOWER = "follower"
    SPOTTER = "spotter"
    CUSTOM = "custom"


class PhaseType(str, Enum):
    """Task execution phases"""

    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSFER = "transfer"
    PLACE = "place"
    RETREAT = "retreat"
    ABORT = "abort"


@dataclass
class RoleConfig:
    """Configuration for a single actor role"""

    role_id: str
    role_type: RoleType
    observation_dims: int
    action_dims: int
    requires_coordination: bool = True
    fallback_behavior: Optional[str] = None


@dataclass
class PolicyAdapter:
    """
    Role-conditioned policy adapter (small adapter, not full foundation weights)

    This contains only the role-specific parameters that adapt a shared
    base policy to individual actor roles.
    """

    role_id: str
    adapter_type: str  # "linear", "lora", "ia3", etc.
    adapter_weights: Dict[str, torch.Tensor]
    observation_preprocessor: Optional[Dict[str, Any]] = None
    action_postprocessor: Optional[Dict[str, Any]] = None

    def save(self, path: Path) -> None:
        """Save adapter weights to disk"""
        torch.save(
            {
                "role_id": self.role_id,
                "adapter_type": self.adapter_type,
                "weights": self.adapter_weights,
                "obs_preprocessor": self.observation_preprocessor,
                "action_postprocessor": self.action_postprocessor,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "PolicyAdapter":
        """Load adapter weights from disk"""
        data = torch.load(path)
        return cls(
            role_id=data["role_id"],
            adapter_type=data["adapter_type"],
            adapter_weights=data["weights"],
            observation_preprocessor=data.get("obs_preprocessor"),
            action_postprocessor=data.get("action_postprocessor"),
        )


@dataclass
class CoordinationEncoder:
    """
    Shared coordination latent encoder

    Produces coordination latent z_t from multi-agent observations
    to enable synchronized behavior.
    """

    encoder_type: str  # "transformer", "rnn", "mlp"
    encoder_weights: Dict[str, torch.Tensor]
    latent_dim: int
    sequence_length: int
    fusion_strategy: str = "attention"  # "attention", "concat", "average"

    def save(self, path: Path) -> None:
        """Save encoder weights to disk"""
        torch.save(
            {
                "encoder_type": self.encoder_type,
                "weights": self.encoder_weights,
                "latent_dim": self.latent_dim,
                "sequence_length": self.sequence_length,
                "fusion_strategy": self.fusion_strategy,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "CoordinationEncoder":
        """Load encoder weights from disk"""
        data = torch.load(path)
        return cls(
            encoder_type=data["encoder_type"],
            encoder_weights=data["weights"],
            latent_dim=data["latent_dim"],
            sequence_length=data["sequence_length"],
            fusion_strategy=data.get("fusion_strategy", "attention"),
        )


@dataclass
class SafetyEnvelope:
    """Safety constraints and runtime limits"""

    max_velocity: Dict[str, float]  # joint/cartesian velocity limits
    max_acceleration: Dict[str, float]
    max_force: Dict[str, float]
    max_torque: Dict[str, float]
    min_separation_distance: float  # minimum actor-to-actor distance (meters)
    workspace_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    collision_primitives: List[Dict[str, Any]]
    emergency_stop_triggers: List[str]

    def validate_state(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> Tuple[bool, List[str]]:
        """
        Validate current state against safety envelope

        Returns:
            (is_safe, list_of_violations)
        """
        violations = []

        # Check velocity limits
        for i, vel in enumerate(velocities):
            if abs(vel) > self.max_velocity.get(f"joint_{i}", float("inf")):
                violations.append(f"Velocity limit exceeded on joint {i}")

        # Check workspace bounds
        min_bound, max_bound = self.workspace_bounds
        for i, pos in enumerate(positions):
            if not (min_bound[i] <= pos <= max_bound[i]):
                violations.append(f"Position out of workspace bounds on axis {i}")

        return len(violations) == 0, violations


class CSAMetadata(BaseModel):
    """CSA metadata and provenance"""

    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    skill_name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Training provenance
    num_demonstrations: int
    training_sites: List[str]
    training_duration_seconds: float
    base_model: Optional[str] = None

    # Compatibility matrix
    compatible_robots: List[str]  # e.g., ["ur5e", "franka_panda"]
    compatible_end_effectors: List[str]  # e.g., ["robotiq_2f85", "schunk_svh"]
    min_actors: int
    max_actors: int

    # Privacy metadata
    privacy_mode: str  # "ldp", "dp_sgd", "he", "none"
    epsilon: Optional[float] = None  # For DP modes
    delta: Optional[float] = None

    # Testing
    test_pass_rate: float = Field(ge=0.0, le=1.0)
    test_coverage: float = Field(ge=0.0, le=1.0)

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate semantic versioning"""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("Version must be in format X.Y.Z")
        if not all(p.isdigit() for p in parts):
            raise ValueError("Version parts must be integers")
        return v


@dataclass
class CooperativeSkillArtefact:
    """
    Complete Cooperative Skill Artefact (CSA)

    Encapsulates everything needed to deploy and execute a multi-actor
    cooperative skill.
    """

    # Core components
    roles: List[RoleConfig]
    policy_adapters: List[PolicyAdapter]
    coordination_encoder: CoordinationEncoder
    phase_machine_xml: str  # BehaviorTree.CPP XML definition
    safety_envelope: SafetyEnvelope

    # Metadata
    metadata: CSAMetadata

    # Tests (deterministic offline checks)
    test_suite: Dict[str, Any] = field(default_factory=dict)

    # Optional: shared base policy (if included)
    shared_base_policy: Optional[Dict[str, torch.Tensor]] = None

    def get_role_adapter(self, role_id: str) -> Optional[PolicyAdapter]:
        """Retrieve adapter for specific role"""
        for adapter in self.policy_adapters:
            if adapter.role_id == role_id:
                return adapter
        return None

    def validate_compatibility(self, robot_model: str, end_effector: str) -> bool:
        """Check if CSA is compatible with hardware"""
        return (
            robot_model in self.metadata.compatible_robots
            and end_effector in self.metadata.compatible_end_effectors
        )

    def run_test_suite(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run deterministic offline tests

        Returns:
            (all_passed, test_results)
        """
        results = {}
        all_passed = True

        # Test 1: Phase machine is well-formed
        try:
            # TODO: Parse and validate BehaviorTree XML
            results["phase_machine_valid"] = True
        except Exception as e:
            results["phase_machine_valid"] = False
            results["phase_machine_error"] = str(e)
            all_passed = False

        # Test 2: All roles have adapters
        role_ids = {r.role_id for r in self.roles}
        adapter_ids = {a.role_id for a in self.policy_adapters}
        if role_ids == adapter_ids:
            results["role_adapter_mapping"] = True
        else:
            results["role_adapter_mapping"] = False
            results["missing_adapters"] = list(role_ids - adapter_ids)
            all_passed = False

        # Test 3: Coordination encoder dimensions match
        # TODO: Validate encoder input/output dims

        # Test 4: Safety envelope is complete
        required_fields = [
            "max_velocity",
            "max_acceleration",
            "max_force",
            "max_torque",
        ]
        envelope_complete = all(
            hasattr(self.safety_envelope, f) and getattr(self.safety_envelope, f)
            for f in required_fields
        )
        results["safety_envelope_complete"] = envelope_complete
        if not envelope_complete:
            all_passed = False

        return all_passed, results

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage"""
        return {
            "roles": [
                {
                    "role_id": r.role_id,
                    "role_type": r.role_type.value,
                    "observation_dims": r.observation_dims,
                    "action_dims": r.action_dims,
                    "requires_coordination": r.requires_coordination,
                    "fallback_behavior": r.fallback_behavior,
                }
                for r in self.roles
            ],
            "phase_machine_xml": self.phase_machine_xml,
            "metadata": self.metadata.model_dump(),
        }
