"""
CSA to MoE Skill Adapter

Converts Cooperative Skill Artefacts (CSA) from Dynamical-SIL into
Mixture-of-Experts (MoE) skill format for the Edge Platform.

Integration Strategy:
- CSA policy adapters → MoE expert networks
- CSA coordination encoder → Skill router context
- CSA metadata → Skill metadata headers
- Multi-actor roles → Expert specialization
"""

import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn


@dataclass
class MoESkillMetadata:
    """Metadata format for Edge Platform MoE skills"""
    skill_name: str
    version: str
    num_experts: int
    expert_specializations: List[str]  # Maps to CSA roles
    input_dim: int
    output_dim: int
    router_dim: int
    created_at: str
    source_csa_id: Optional[str] = None
    privacy_level: str = "encrypted"  # encrypted, differential_privacy, plaintext


@dataclass
class MoEExpertWeights:
    """Weights for a single MoE expert"""
    expert_id: str
    specialization: str  # e.g., "leader", "follower", "observer"
    weights: Dict[str, torch.Tensor]
    input_dim: int
    output_dim: int


class CSAToMoEAdapter:
    """
    Adapts Cooperative Skill Artefacts to MoE skill format.

    Conversion Strategy:
    1. Extract CSA package (tarball)
    2. Load policy adapters for each role
    3. Convert each role-specific adapter to an MoE expert
    4. Create skill router from coordination encoder
    5. Package as Edge Platform skill format

    Example:
        adapter = CSAToMoEAdapter()
        moe_skill = adapter.convert_csa_to_moe(
            csa_path="artifacts/cooperative_assembly_v1.0.tar.gz",
            output_path="skills/cooperative_assembly_moe.pt"
        )
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def convert_csa_to_moe(
        self,
        csa_path: Path,
        output_path: Path,
        router_hidden_dim: int = 256,
    ) -> MoESkillMetadata:
        """
        Convert a CSA package to MoE skill format.

        Args:
            csa_path: Path to CSA tarball
            output_path: Path to save MoE skill
            router_hidden_dim: Hidden dimension for skill router

        Returns:
            Metadata for the created MoE skill
        """
        print(f"Converting CSA {csa_path} to MoE skill format...")

        # Step 1: Extract CSA package
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with tarfile.open(csa_path, "r:gz") as tar:
                tar.extractall(tmpdir_path)

            # Step 2: Load CSA manifest
            with open(tmpdir_path / "manifest.json") as f:
                manifest = json.load(f)

            # Step 3: Extract roles and policy adapters
            roles = manifest.get("roles", [])
            num_experts = len(roles)

            experts = []
            for role in roles:
                role_id = role["role_id"]
                adapter_path = tmpdir_path / f"roles/{role_id}/policy_adapter.pt"

                if adapter_path.exists():
                    expert = self._create_expert_from_adapter(
                        adapter_path=adapter_path,
                        role_config=role,
                    )
                    experts.append(expert)

            # Step 4: Create skill router from coordination encoder
            coord_encoder_path = tmpdir_path / "coordination_encoder.pt"
            router = self._create_router_from_encoder(
                encoder_path=coord_encoder_path,
                num_experts=num_experts,
                hidden_dim=router_hidden_dim,
            )

            # Step 5: Package as MoE skill
            metadata = MoESkillMetadata(
                skill_name=manifest["skill_name"],
                version=manifest["version"],
                num_experts=num_experts,
                expert_specializations=[r["role_id"] for r in roles],
                input_dim=manifest.get("observation_dim", 15),
                output_dim=manifest.get("action_dim", 7),
                router_dim=router_hidden_dim,
                created_at=manifest.get("created_at", ""),
                source_csa_id=manifest.get("csa_id"),
            )

            # Step 6: Save MoE skill
            self._save_moe_skill(
                output_path=output_path,
                experts=experts,
                router=router,
                metadata=metadata,
            )

        print(f"✓ MoE skill saved to {output_path}")
        return metadata

    def _create_expert_from_adapter(
        self,
        adapter_path: Path,
        role_config: Dict[str, Any],
    ) -> MoEExpertWeights:
        """Convert CSA policy adapter to MoE expert weights"""

        # Load adapter weights
        adapter_state = torch.load(adapter_path, map_location=self.device)

        expert = MoEExpertWeights(
            expert_id=role_config["role_id"],
            specialization=role_config["role_id"],
            weights=adapter_state,
            input_dim=role_config.get("observation_dim", 15),
            output_dim=role_config.get("action_dim", 7),
        )

        return expert

    def _create_router_from_encoder(
        self,
        encoder_path: Path,
        num_experts: int,
        hidden_dim: int,
    ) -> nn.Module:
        """
        Create MoE router from CSA coordination encoder.

        The router determines which expert(s) to use based on context.
        CSA coordination encoder provides multi-actor context that
        maps naturally to expert selection.
        """

        # Load coordination encoder
        if encoder_path.exists():
            encoder_state = torch.load(encoder_path, map_location=self.device)
            coord_dim = encoder_state.get("latent_dim", 64)
        else:
            coord_dim = 64  # default

        # Create router network
        router = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Softmax(dim=-1),  # Expert selection probabilities
        )

        return router

    def _save_moe_skill(
        self,
        output_path: Path,
        experts: List[MoEExpertWeights],
        router: nn.Module,
        metadata: MoESkillMetadata,
    ):
        """Save MoE skill in Edge Platform format"""

        skill_package = {
            "metadata": {
                "skill_name": metadata.skill_name,
                "version": metadata.version,
                "num_experts": metadata.num_experts,
                "expert_specializations": metadata.expert_specializations,
                "input_dim": metadata.input_dim,
                "output_dim": metadata.output_dim,
                "router_dim": metadata.router_dim,
                "created_at": metadata.created_at,
                "source_csa_id": metadata.source_csa_id,
                "privacy_level": metadata.privacy_level,
                "format_version": "1.0",
                "integration_source": "dynamical_sil",
            },
            "experts": [
                {
                    "expert_id": exp.expert_id,
                    "specialization": exp.specialization,
                    "weights": exp.weights,
                    "input_dim": exp.input_dim,
                    "output_dim": exp.output_dim,
                }
                for exp in experts
            ],
            "router": router.state_dict(),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(skill_package, output_path)

    def convert_moe_to_csa(
        self,
        moe_path: Path,
        output_path: Path,
        num_actors: int = 2,
    ):
        """
        Reverse conversion: MoE skill → CSA package.

        This allows skills trained on Edge Platform to be imported
        back into Dynamical-SIL for multi-actor swarm learning.
        """

        print(f"Converting MoE skill {moe_path} to CSA format...")

        # Load MoE skill
        skill_package = torch.load(moe_path, map_location=self.device)
        metadata = skill_package["metadata"]
        experts = skill_package["experts"]
        router = skill_package["router"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create CSA structure
            roles_dir = tmpdir_path / "roles"
            roles_dir.mkdir(parents=True, exist_ok=True)

            # Convert each expert to a role-specific adapter
            roles_config = []
            for expert in experts:
                role_id = expert["expert_id"]
                role_dir = roles_dir / role_id
                role_dir.mkdir(parents=True, exist_ok=True)

                # Save adapter weights
                torch.save(
                    expert["weights"],
                    role_dir / "policy_adapter.pt"
                )

                roles_config.append({
                    "role_id": role_id,
                    "observation_dim": expert["input_dim"],
                    "action_dim": expert["output_dim"],
                    "capabilities": [expert["specialization"]],
                })

            # Save coordination encoder (from router)
            coord_encoder = {
                "latent_dim": metadata.get("router_dim", 64),
                "encoder_type": "transformer",
                "state_dict": router,
            }
            torch.save(coord_encoder, tmpdir_path / "coordination_encoder.pt")

            # Create manifest
            manifest = {
                "skill_name": metadata["skill_name"],
                "version": metadata["version"],
                "roles": roles_config,
                "observation_dim": metadata["input_dim"],
                "action_dim": metadata["output_dim"],
                "num_actors": num_actors,
                "created_at": metadata.get("created_at", ""),
                "source_format": "edge_platform_moe",
                "csa_version": "1.0",
            }

            with open(tmpdir_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # Package as CSA tarball
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(tmpdir_path, arcname=".")

        print(f"✓ CSA package saved to {output_path}")


class MoESkillValidator:
    """Validates MoE skills converted from CSA"""

    def validate_skill(self, skill_path: Path) -> tuple[bool, str]:
        """
        Validate MoE skill package.

        Returns:
            (is_valid, message)
        """

        try:
            skill = torch.load(skill_path)

            # Check required fields
            required_fields = ["metadata", "experts", "router"]
            for field in required_fields:
                if field not in skill:
                    return False, f"Missing required field: {field}"

            # Validate metadata
            metadata = skill["metadata"]
            num_experts = metadata.get("num_experts", 0)
            if num_experts != len(skill["experts"]):
                return False, f"Metadata declares {num_experts} experts but found {len(skill['experts'])}"

            # Validate each expert
            for i, expert in enumerate(skill["experts"]):
                if "weights" not in expert:
                    return False, f"Expert {i} missing weights"
                if "expert_id" not in expert:
                    return False, f"Expert {i} missing expert_id"

            return True, "Skill validation passed"

        except Exception as e:
            return False, f"Validation error: {str(e)}"
