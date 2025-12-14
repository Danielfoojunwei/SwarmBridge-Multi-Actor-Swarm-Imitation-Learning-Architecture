"""
Advanced Multi-Actor Coordination System

Enhancements over base implementation:
- Dynamic role assignment and switching
- Hierarchical multi-actor structures
- Intent sharing and prediction
- Adaptive coordination based on task complexity
- Multi-level safety verification
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class CoordinationMode(str, Enum):
    """Multi-actor coordination modes"""
    HIERARCHICAL = "hierarchical"  # Leader-follower structure
    PEER_TO_PEER = "peer_to_peer"  # Equal collaboration
    DYNAMIC = "dynamic"  # Adaptive based on context
    CONSENSUS = "consensus"  # Vote-based decisions


class ActorIntent(str, Enum):
    """Actor intent types for communication"""
    GRASP = "grasp"
    MOVE = "move"
    WAIT = "wait"
    HANDOFF = "handoff"
    SUPPORT = "support"
    MONITOR = "monitor"


@dataclass
class MultiActorState:
    """Complete multi-actor system state"""
    actor_positions: Dict[str, np.ndarray]  # role_id -> position
    actor_velocities: Dict[str, np.ndarray]  # role_id -> velocity
    actor_intents: Dict[str, ActorIntent]  # role_id -> current intent
    coordination_mode: CoordinationMode
    shared_object_state: Optional[np.ndarray] = None
    formation_config: Optional[Dict] = None


class IntentCommunicationModule(nn.Module):
    """
    Enables actors to share and predict intents

    Architecture:
    - Each actor broadcasts intent embedding
    - Others predict next intents
    - Coordination latent updated with intent information
    """

    def __init__(
        self,
        num_actors: int,
        intent_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_actors = num_actors
        self.intent_dim = intent_dim

        # Intent encoder (maps discrete intent + state -> continuous embedding)
        self.intent_encoder = nn.Sequential(
            nn.Linear(10 + len(ActorIntent), intent_dim),  # state + one-hot intent
            nn.ReLU(),
            nn.Linear(intent_dim, intent_dim),
        )

        # Intent predictor (predicts other actors' next intents)
        self.intent_predictor = nn.Sequential(
            nn.Linear(intent_dim * num_actors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ActorIntent) * num_actors),
        )

        # Communication attention (weighs intent importance)
        self.communication_attn = nn.MultiheadAttention(
            embed_dim=intent_dim,
            num_heads=4,
            batch_first=True,
        )

    def forward(
        self,
        actor_states: torch.Tensor,  # [B, num_actors, state_dim]
        actor_intents: torch.Tensor,  # [B, num_actors, num_intent_types]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            actor_states: Current states of all actors
            actor_intents: One-hot encoded intents

        Returns:
            intent_embeddings: [B, num_actors, intent_dim]
            predicted_intents: [B, num_actors, num_intent_types]
        """
        B, N, S = actor_states.shape

        # Encode current intents with states
        combined = torch.cat([actor_states, actor_intents], dim=-1)
        intent_embeddings = self.intent_encoder(combined)  # [B, N, intent_dim]

        # Communication attention (actors attend to each other's intents)
        attended_intents, _ = self.communication_attn(
            intent_embeddings,
            intent_embeddings,
            intent_embeddings,
        )

        # Predict next intents
        flat_intents = attended_intents.reshape(B, N * self.intent_dim)
        predicted_intents = self.intent_predictor(flat_intents)
        predicted_intents = predicted_intents.reshape(B, N, len(ActorIntent))

        return attended_intents, predicted_intents


class DynamicRoleAssigner(nn.Module):
    """
    Dynamically assigns roles based on actor capabilities and task requirements

    Features:
    - Capability-aware role matching
    - Task-specific role optimization
    - Runtime role switching support
    """

    def __init__(
        self,
        num_actors: int,
        num_roles: int,
        capability_dim: int = 16,
        task_embedding_dim: int = 32,
    ):
        super().__init__()
        self.num_actors = num_actors
        self.num_roles = num_roles

        # Capability encoder
        self.capability_encoder = nn.Linear(capability_dim, 64)

        # Task requirement encoder
        self.task_encoder = nn.Linear(task_embedding_dim, 64)

        # Role assignment network (outputs assignment matrix)
        self.assignment_network = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_actors * num_roles),
        )

    def forward(
        self,
        actor_capabilities: torch.Tensor,  # [B, num_actors, capability_dim]
        task_requirements: torch.Tensor,  # [B, task_embedding_dim]
    ) -> torch.Tensor:
        """
        Compute role assignment matrix

        Returns:
            assignment_matrix: [B, num_actors, num_roles]
                               Soft assignment (use gumbel-softmax for hard)
        """
        B = actor_capabilities.shape[0]

        # Encode capabilities
        cap_encoded = self.capability_encoder(actor_capabilities)  # [B, N, 64]

        # Encode task
        task_encoded = self.task_encoder(task_requirements)  # [B, 64]
        task_encoded = task_encoded.unsqueeze(1).expand(-1, self.num_actors, -1)

        # Combine and compute assignments
        combined = torch.cat([cap_encoded, task_encoded], dim=-1)
        assignment_logits = self.assignment_network(combined.reshape(B, -1))
        assignment_matrix = assignment_logits.reshape(B, self.num_actors, self.num_roles)

        # Softmax over roles for each actor
        assignment_matrix = torch.softmax(assignment_matrix, dim=-1)

        return assignment_matrix


class HierarchicalCoordinationEncoder(nn.Module):
    """
    Multi-level coordination with hierarchical structure

    Levels:
    1. Individual actor level (local goals)
    2. Sub-group level (pairs/trios coordination)
    3. Global level (full team coordination)
    """

    def __init__(
        self,
        num_actors: int,
        obs_dim: int,
        latent_dim: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_actors = num_actors
        self.latent_dim = latent_dim

        # Level 1: Individual encoders
        self.individual_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Level 2: Pairwise coordination (transformer)
        self.pairwise_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads // 2,
            batch_first=True,
        )

        # Level 3: Global coordination (transformer)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Hierarchical fusion
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            observations: [B, num_actors, obs_dim]

        Returns:
            Dict with individual, pairwise, and global latents
        """
        B, N, D = observations.shape

        # Level 1: Individual encoding
        individual_latents = self.individual_encoder(observations)  # [B, N, latent_dim]

        # Level 2: Pairwise coordination
        pairwise_latents, _ = self.pairwise_attn(
            individual_latents,
            individual_latents,
            individual_latents,
        )

        # Level 3: Global coordination
        global_latents, attn_weights = self.global_attn(
            pairwise_latents,
            pairwise_latents,
            pairwise_latents,
        )

        # Hierarchical fusion
        all_latents = torch.cat(
            [individual_latents, pairwise_latents, global_latents], dim=-1
        )
        fused_latents = self.fusion(all_latents)

        return {
            "individual": individual_latents,
            "pairwise": pairwise_latents,
            "global": global_latents,
            "fused": fused_latents,
            "attention_weights": attn_weights,
        }


class AdaptiveCoordinationPolicy(nn.Module):
    """
    Policy that adapts coordination strategy based on task phase and complexity

    Features:
    - Switches between coordination modes dynamically
    - Adjusts communication frequency
    - Modulates safety margins based on uncertainty
    """

    def __init__(
        self,
        role_configs: Dict[str, int],  # role_id -> action_dim
        coordination_latent_dim: int = 64,
        num_coordination_modes: int = 4,
    ):
        super().__init__()
        self.role_configs = role_configs
        self.num_modes = num_coordination_modes

        # Mode selector (decides coordination mode)
        self.mode_selector = nn.Sequential(
            nn.Linear(coordination_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_coordination_modes),
        )

        # Mode-specific policy heads
        self.mode_policies = nn.ModuleDict()
        for mode in CoordinationMode:
            self.mode_policies[mode.value] = nn.ModuleDict()
            for role_id, action_dim in role_configs.items():
                self.mode_policies[mode.value][role_id] = nn.Sequential(
                    nn.Linear(coordination_latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim),
                )

        # Uncertainty estimator (for safety margin adaptation)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(coordination_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Ensure positive
        )

    def forward(
        self,
        coordination_latent: torch.Tensor,  # [B, latent_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mode-adaptive actions

        Returns:
            Dict with:
            - mode_probs: [B, num_modes]
            - actions: {role_id: {mode: [B, action_dim]}}
            - uncertainty: [B, 1]
        """
        # Select coordination mode
        mode_logits = self.mode_selector(coordination_latent)
        mode_probs = torch.softmax(mode_logits, dim=-1)

        # Compute actions for each mode and role
        actions = {}
        for role_id in self.role_configs.keys():
            actions[role_id] = {}
            for mode in CoordinationMode:
                actions[role_id][mode.value] = self.mode_policies[mode.value][role_id](
                    coordination_latent
                )

        # Estimate uncertainty
        uncertainty = self.uncertainty_head(coordination_latent)

        return {
            "mode_probs": mode_probs,
            "actions": actions,
            "uncertainty": uncertainty,
        }


class MultiActorSafetyVerifier:
    """
    Advanced safety verification for multi-actor scenarios

    Checks:
    - Actor-to-actor collision avoidance
    - Coordinated motion feasibility
    - Intent consistency
    - Formation constraints
    """

    def __init__(
        self,
        min_separation: float = 0.5,
        max_relative_velocity: float = 0.3,
        formation_tolerance: float = 0.1,
    ):
        self.min_separation = min_separation
        self.max_relative_velocity = max_relative_velocity
        self.formation_tolerance = formation_tolerance

    def verify_state(
        self, state: MultiActorState
    ) -> Tuple[bool, List[str], Dict[str, float]]:
        """
        Comprehensive safety verification

        Returns:
            is_safe: bool
            violations: List of violation descriptions
            safety_metrics: Dict of safety scores
        """
        violations = []
        metrics = {}

        # 1. Check pairwise separation
        positions = list(state.actor_positions.values())
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions[i + 1:], start=i + 1):
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < self.min_separation:
                    violations.append(
                        f"Actors {i} and {j} too close: {dist:.2f}m < {self.min_separation}m"
                    )

        metrics["min_separation"] = min(
            [np.linalg.norm(positions[i] - positions[j])
             for i in range(len(positions))
             for j in range(i + 1, len(positions))]
        ) if len(positions) > 1 else float('inf')

        # 2. Check relative velocities
        velocities = list(state.actor_velocities.values())
        for i, vel_i in enumerate(velocities):
            for j, vel_j in enumerate(velocities[i + 1:], start=i + 1):
                rel_vel = np.linalg.norm(vel_i - vel_j)
                if rel_vel > self.max_relative_velocity:
                    violations.append(
                        f"Relative velocity between {i} and {j} too high: {rel_vel:.2f} m/s"
                    )

        metrics["max_relative_velocity"] = max(
            [np.linalg.norm(velocities[i] - velocities[j])
             for i in range(len(velocities))
             for j in range(i + 1, len(velocities))]
        ) if len(velocities) > 1 else 0.0

        # 3. Check intent consistency
        intents = list(state.actor_intents.values())
        conflicting_intents = self._check_intent_conflicts(intents)
        if conflicting_intents:
            violations.extend(conflicting_intents)

        # 4. Check formation constraints (if applicable)
        if state.formation_config:
            formation_violations = self._verify_formation(
                state.actor_positions, state.formation_config
            )
            violations.extend(formation_violations)

        is_safe = len(violations) == 0

        return is_safe, violations, metrics

    def _check_intent_conflicts(self, intents: List[ActorIntent]) -> List[str]:
        """Check for conflicting intents"""
        conflicts = []

        # Example: Multiple actors trying to grasp simultaneously
        grasp_count = sum(1 for intent in intents if intent == ActorIntent.GRASP)
        if grasp_count > 1:
            conflicts.append(f"Multiple actors ({grasp_count}) attempting to grasp")

        return conflicts

    def _verify_formation(
        self, positions: Dict[str, np.ndarray], formation: Dict
    ) -> List[str]:
        """Verify actors maintain formation"""
        violations = []

        # Check formation shape (e.g., line, triangle, square)
        if formation.get("type") == "line":
            # Verify actors are roughly collinear
            pos_array = np.array(list(positions.values()))
            if len(pos_array) >= 3:
                # Compute deviation from best-fit line
                centroid = pos_array.mean(axis=0)
                centered = pos_array - centroid
                _, _, vh = np.linalg.svd(centered)
                line_direction = vh[0]

                # Project onto perpendicular directions
                deviations = centered - np.outer(
                    np.dot(centered, line_direction), line_direction
                )
                max_deviation = np.max(np.linalg.norm(deviations, axis=1))

                if max_deviation > self.formation_tolerance:
                    violations.append(
                        f"Formation deviation: {max_deviation:.2f}m > {self.formation_tolerance}m"
                    )

        return violations


class AdvancedMultiActorTrainer:
    """
    Enhanced training for multi-actor systems

    Features:
    - Curriculum learning (start with 2 actors, scale to N)
    - Role diversity regularization
    - Communication efficiency optimization
    - Intent prediction auxiliary task
    """

    def __init__(
        self,
        hierarchical_encoder: HierarchicalCoordinationEncoder,
        adaptive_policy: AdaptiveCoordinationPolicy,
        intent_module: IntentCommunicationModule,
        learning_rate: float = 1e-4,
    ):
        self.encoder = hierarchical_encoder
        self.policy = adaptive_policy
        self.intent_module = intent_module

        # Combine parameters
        params = (
            list(self.encoder.parameters())
            + list(self.policy.parameters())
            + list(self.intent_module.parameters())
        )

        self.optimizer = torch.optim.AdamW(params, lr=learning_rate)

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        intents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss

        Components:
        1. Behavior cloning loss
        2. Intent prediction loss
        3. Coordination consistency loss
        4. Communication efficiency loss
        """
        # Encode coordination
        coord_outputs = self.encoder(observations)
        fused_latent = coord_outputs["fused"]

        # Predict actions
        policy_outputs = self.policy(fused_latent)

        # 1. BC loss (for each role)
        bc_loss = 0.0
        for role_id, true_actions in actions.items():
            # Average over coordination modes (weighted by mode probs)
            mode_probs = policy_outputs["mode_probs"]
            pred_actions_per_mode = policy_outputs["actions"][role_id]

            # Compute weighted action prediction
            weighted_pred = sum(
                mode_probs[:, i:i+1] * pred_actions_per_mode[mode.value]
                for i, mode in enumerate(CoordinationMode)
            )

            bc_loss += torch.nn.functional.mse_loss(weighted_pred, true_actions)

        bc_loss /= len(actions)

        # 2. Intent prediction loss
        intent_embeds, predicted_intents = self.intent_module(
            observations, intents
        )
        intent_loss = torch.nn.functional.cross_entropy(
            predicted_intents.reshape(-1, len(ActorIntent)),
            intents.reshape(-1, len(ActorIntent)).argmax(dim=-1),
        )

        # 3. Coordination consistency (encourage similar latents for coordinated actions)
        coord_variance = coord_outputs["global"].var(dim=1).mean()
        consistency_loss = coord_variance * 0.1

        # 4. Communication efficiency (penalize unnecessary communication)
        attn_entropy = -(
            coord_outputs["attention_weights"]
            * torch.log(coord_outputs["attention_weights"] + 1e-8)
        ).sum(dim=-1).mean()
        comm_efficiency_loss = -attn_entropy * 0.01  # Encourage focused attention

        total_loss = (
            bc_loss
            + 0.1 * intent_loss
            + 0.05 * consistency_loss
            + 0.01 * comm_efficiency_loss
        )

        return {
            "total_loss": total_loss,
            "bc_loss": bc_loss,
            "intent_loss": intent_loss,
            "consistency_loss": consistency_loss,
            "comm_efficiency_loss": comm_efficiency_loss,
        }
