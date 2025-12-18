"""
CooperativeSkillArtifact Schema

Skill export format aligned with Dynamical v0.3.3 MoE layer.
Ensures SwarmBridge outputs are directly compatible with Dynamical's
skill-centric architecture.

Key Design Principles:
1. skill_id namespace shared with Dynamical skill catalog
2. input_embedding_type matches Dynamical perception (MOAI)
3. encryption_scheme compatible with N2HE/Pyfhel
4. Versioning supports federated rounds and site tracking
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path
import json


class InputEmbeddingType(Enum):
    """Input embedding types compatible with Dynamical"""
    MOAI_512 = "moai_512"  # Dynamical v0.3.3 standard
    MOAI_256 = "moai_256"  # Compressed variant
    STATE_VECTOR = "state_vector"  # Fallback for non-MOAI
    STATE_ACTION = "state_action"  # State + action history


class EncryptionScheme(Enum):
    """Encryption schemes compatible with Dynamical ecosystem"""
    N2HE_128 = "n2he_128bit"  # Dynamical standard
    PYFHEL_BFV = "pyfhel_bfv"  # SwarmBridge/SwarmBrain
    PYFHEL_CKKS = "pyfhel_ckks"  # For real-valued weights
    OPENFHE_BFV = "openfhe_bfv"  # SwarmBrain standard
    NONE = "none"  # Unencrypted (dev only)


class CoordinationPrimitiveType(Enum):
    """Multi-actor coordination patterns"""
    HANDOVER = "handover"
    COLLABORATIVE_MANIPULATION = "collaborative_manipulation"
    LEADER_FOLLOWER = "leader_follower"
    BARRIER = "barrier"
    FORMATION = "formation"
    MUTEX = "mutex"
    RENDEZVOUS = "rendezvous"


@dataclass
class DynamicalCompatibilityMetadata:
    """Metadata ensuring Dynamical compatibility"""
    dynamical_version: str = "0.3.3"
    moai_version: Optional[str] = None
    vla_base_model: str = "pi0_7b"  # or "openvla_7b"
    moe_layer_compatible: bool = True
    skill_catalog_verified: bool = False
    n2he_encryption_compatible: bool = True


@dataclass
class CooperativeSkillArtifact:
    """
    Cooperative skill artifact for multi-actor scenarios.
    
    This format ensures SwarmBridge outputs can be directly consumed
    by Dynamical's MoE skill layer on each robot.
    
    Example:
        artifact = CooperativeSkillArtifact(
            skill_id="handover_box",
            role_id="giver",
            input_embedding_type=InputEmbeddingType.MOAI_512,
            expert_checkpoint_uri="s3://skills/handover_v4/giver.onnx",
            encryption_scheme=EncryptionScheme.N2HE_128,
            version="4.0",
            site_id="warehouse_A",
            round_id=15,
        )
    """
    
    # Core identifiers (must match Dynamical skill catalog)
    skill_id: str  # e.g., "handover_box", "collaborative_assembly"
    role_id: str  # e.g., "giver", "receiver", "leader", "follower"
    
    # Input format (must match Dynamical perception stack)
    input_embedding_type: InputEmbeddingType  # MOAI_512 for Dynamical v0.3.3
    
    # Model checkpoint (ONNX or PyTorch compatible with Dynamical MoE loader)
    expert_checkpoint_uri: str  # S3, local path, or Dynamical registry URI
    
    # Encryption (must be compatible with Dynamical N2HE or ecosystem bridges)
    encryption_scheme: EncryptionScheme
    
    # Versioning (critical for federated learning)
    version: str  # Semantic version (e.g., "4.0", "3.2.1")
    site_id: str  # Originating site/robot fleet ID
    round_id: int  # Federated learning round number
    
    # Multi-actor coordination
    coordination_primitive: CoordinationPrimitiveType
    compatible_roles: List[str] = field(default_factory=list)  # Other roles this can work with
    
    # Dynamical compatibility
    dynamical_compatibility: DynamicalCompatibilityMetadata = field(
        default_factory=DynamicalCompatibilityMetadata
    )
    
    # Extended metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate Dynamical compatibility"""
        # Ensure skill_id is valid
        if not self.skill_id:
            raise ValueError("skill_id is required")
        
        # Validate MOAI embedding if used
        if self.input_embedding_type in [InputEmbeddingType.MOAI_512, InputEmbeddingType.MOAI_256]:
            if not self.dynamical_compatibility.moai_version:
                raise ValueError("moai_version required when using MOAI embeddings")
        
        # Ensure encryption scheme is set
        if not self.encryption_scheme:
            self.encryption_scheme = EncryptionScheme.NONE
        
        # Set default compatible roles
        if not self.compatible_roles:
            self.compatible_roles = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        
        # Convert enums to strings
        data["input_embedding_type"] = self.input_embedding_type.value
        data["encryption_scheme"] = self.encryption_scheme.value
        data["coordination_primitive"] = self.coordination_primitive.value
        
        return data
    
    def to_json(self, path: Path) -> None:
        """Save to JSON manifest"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CooperativeSkillArtifact":
        """Load from dictionary"""
        # Convert string enums back to enum types
        data = data.copy()
        
        data["input_embedding_type"] = InputEmbeddingType(data["input_embedding_type"])
        data["encryption_scheme"] = EncryptionScheme(data["encryption_scheme"])
        data["coordination_primitive"] = CoordinationPrimitiveType(data["coordination_primitive"])
        
        if "dynamical_compatibility" in data:
            data["dynamical_compatibility"] = DynamicalCompatibilityMetadata(
                **data["dynamical_compatibility"]
            )
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Path) -> "CooperativeSkillArtifact":
        """Load from JSON manifest"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def validate_dynamical_compatibility(self) -> tuple[bool, str]:
        """
        Validate compatibility with Dynamical v0.3.3.
        
        Returns:
            (is_valid, message)
        """
        issues = []
        
        # Check MOAI version
        if self.input_embedding_type == InputEmbeddingType.MOAI_512:
            if not self.dynamical_compatibility.moai_version:
                issues.append("MOAI version not specified")
            elif self.dynamical_compatibility.moai_version != "0.3.3":
                issues.append(f"MOAI version {self.dynamical_compatibility.moai_version} may not match Dynamical v0.3.3")
        
        # Check encryption compatibility
        if self.encryption_scheme not in [EncryptionScheme.N2HE_128, EncryptionScheme.PYFHEL_CKKS, EncryptionScheme.NONE]:
            issues.append(f"Encryption scheme {self.encryption_scheme.value} may not be compatible with Dynamical")
        
        # Check MoE compatibility flag
        if not self.dynamical_compatibility.moe_layer_compatible:
            issues.append("MoE layer compatibility not verified")
        
        # Check checkpoint URI format
        if not self.expert_checkpoint_uri:
            issues.append("Expert checkpoint URI is required")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Compatible with Dynamical v0.3.3"
    
    def get_dynamical_skill_key(self) -> str:
        """
        Get the key used in Dynamical's skill registry.
        
        Format: {skill_id}/{role_id}/v{version}
        """
        return f"{self.skill_id}/{self.role_id}/v{self.version}"


@dataclass
class CooperativeSkillManifest:
    """
    Manifest for a complete cooperative skill (all roles).
    
    This groups multiple CooperativeSkillArtifacts (one per role)
    into a single multi-actor skill that can be deployed to Dynamical.
    """
    
    skill_id: str
    skill_name: str
    coordination_primitive: CoordinationPrimitiveType
    
    # All role artifacts
    role_artifacts: Dict[str, CooperativeSkillArtifact] = field(default_factory=dict)
    
    # Shared metadata
    training_episodes: int = 0
    federated_sites: List[str] = field(default_factory=list)
    global_version: str = "1.0"
    
    # Dynamical deployment config
    dynamical_deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    def add_role_artifact(self, artifact: CooperativeSkillArtifact):
        """Add a role-specific artifact"""
        if artifact.skill_id != self.skill_id:
            raise ValueError(f"Artifact skill_id {artifact.skill_id} doesn't match manifest {self.skill_id}")
        
        self.role_artifacts[artifact.role_id] = artifact
    
    def validate_all_roles(self) -> tuple[bool, List[str]]:
        """Validate all role artifacts for Dynamical compatibility"""
        issues = []
        
        for role_id, artifact in self.role_artifacts.items():
            is_valid, msg = artifact.validate_dynamical_compatibility()
            if not is_valid:
                issues.append(f"{role_id}: {msg}")
        
        return len(issues) == 0, issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "coordination_primitive": self.coordination_primitive.value,
            "role_artifacts": {
                role_id: artifact.to_dict()
                for role_id, artifact in self.role_artifacts.items()
            },
            "training_episodes": self.training_episodes,
            "federated_sites": self.federated_sites,
            "global_version": self.global_version,
            "dynamical_deployment_config": self.dynamical_deployment_config,
        }
    
    def to_json(self, path: Path):
        """Save manifest to JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> "CooperativeSkillManifest":
        """Load manifest from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        manifest = cls(
            skill_id=data["skill_id"],
            skill_name=data["skill_name"],
            coordination_primitive=CoordinationPrimitiveType(data["coordination_primitive"]),
            training_episodes=data.get("training_episodes", 0),
            federated_sites=data.get("federated_sites", []),
            global_version=data.get("global_version", "1.0"),
            dynamical_deployment_config=data.get("dynamical_deployment_config", {}),
        )
        
        # Load role artifacts
        for role_id, artifact_data in data.get("role_artifacts", {}).items():
            artifact = CooperativeSkillArtifact.from_dict(artifact_data)
            manifest.add_role_artifact(artifact)
        
        return manifest
