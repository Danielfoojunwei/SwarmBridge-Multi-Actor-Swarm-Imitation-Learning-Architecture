"""
Data Format Converters

Converts data formats between Dynamical-SIL and Edge Platform:
- Observations/actions format
- Metadata schemas
- Deployment records
"""

from typing import Dict, Any, List
import numpy as np
import torch


class ObservationConverter:
    """Convert observation formats between systems"""

    @staticmethod
    def sil_to_edge(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert Dynamical-SIL observations to Edge Platform format.

        SIL format: Dict with multi-actor observations
        Edge format: Single-actor with VLA context
        """
        # Extract first actor's observations as example
        if "multi_actor_observations" in obs:
            actor_obs = obs["multi_actor_observations"][0]  # [T, D]
        else:
            actor_obs = obs.get("observations", obs)

        return {
            "visual": torch.from_numpy(actor_obs) if isinstance(actor_obs, np.ndarray) else actor_obs,
            "proprioception": torch.zeros(7),  # Placeholder
            "language_instruction": "",
        }

    @staticmethod
    def edge_to_sil(obs: Dict[str, torch.Tensor], num_actors: int = 2) -> Dict[str, np.ndarray]:
        """
        Convert Edge Platform observations to Dynamical-SIL format.

        Edge format: Single-actor with VLA context
        SIL format: Multi-actor observations
        """
        visual = obs["visual"].cpu().numpy()

        # Replicate for multi-actor (in practice, would have actual multi-actor data)
        multi_actor_obs = np.stack([visual] * num_actors, axis=0)

        return {
            "multi_actor_observations": multi_actor_obs,
            "coordination_context": np.zeros((num_actors, 64)),  # Placeholder
        }


class ActionConverter:
    """Convert action formats between systems"""

    @staticmethod
    def sil_to_edge(actions: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert multi-actor actions to single-actor format"""
        if "multi_actor_actions" in actions:
            # Take first actor's actions
            return torch.from_numpy(actions["multi_actor_actions"][0])
        else:
            arr = actions.get("actions", actions)
            return torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr

    @staticmethod
    def edge_to_sil(actions: torch.Tensor, num_actors: int = 2) -> Dict[str, np.ndarray]:
        """Convert single-actor actions to multi-actor format"""
        action_np = actions.cpu().numpy()

        # Replicate for multi-actor
        multi_actor_actions = np.stack([action_np] * num_actors, axis=0)

        return {
            "multi_actor_actions": multi_actor_actions,
        }


class MetadataConverter:
    """Convert metadata schemas between systems"""

    @staticmethod
    def csa_metadata_to_skill_metadata(csa_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Convert CSA metadata to Edge Platform skill metadata"""
        return {
            "skill_name": csa_meta.get("skill_name", "unknown"),
            "version": csa_meta.get("version", "1.0"),
            "num_experts": len(csa_meta.get("roles", [])),
            "expert_specializations": [r["role_id"] for r in csa_meta.get("roles", [])],
            "input_dim": csa_meta.get("observation_dim", 15),
            "output_dim": csa_meta.get("action_dim", 7),
            "source": "dynamical_sil",
            "source_csa_id": csa_meta.get("csa_id"),
        }

    @staticmethod
    def skill_metadata_to_csa_metadata(skill_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Edge Platform skill metadata to CSA metadata"""
        return {
            "skill_name": skill_meta.get("skill_name", "unknown"),
            "version": skill_meta.get("version", "1.0"),
            "roles": [
                {"role_id": spec, "capabilities": [spec]}
                for spec in skill_meta.get("expert_specializations", [])
            ],
            "observation_dim": skill_meta.get("input_dim", 15),
            "action_dim": skill_meta.get("output_dim", 7),
            "num_actors": skill_meta.get("num_experts", 2),
            "source": "edge_platform",
            "source_skill_id": skill_meta.get("skill_id"),
        }
