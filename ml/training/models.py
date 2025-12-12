"""
Multi-Actor Cooperative Policy Models

Implements:
- Role-conditioned policy adapters (LoRA, IA3, linear)
- Coordination latent encoders (Transformer, RNN)
- Multi-actor datasets
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CoordinationLatentEncoder(nn.Module):
    """
    Encodes multi-actor observations into shared coordination latent z_t

    Supports:
    - Transformer (attention-based fusion)
    - RNN (sequential fusion)
    - MLP (simple concatenation + projection)
    """

    def __init__(
        self,
        encoder_type: str,
        input_dim_per_actor: int,
        num_actors: int,
        latent_dim: int,
        sequence_length: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.num_actors = num_actors
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        if encoder_type == "transformer":
            # Transformer encoder with multi-head attention
            self.input_proj = nn.Linear(input_dim_per_actor, hidden_dim)
            self.pos_embedding = nn.Parameter(
                torch.randn(1, sequence_length, hidden_dim)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.output_proj = nn.Linear(hidden_dim, latent_dim)

        elif encoder_type in ["rnn", "lstm", "gru"]:
            # RNN-based encoder
            self.input_proj = nn.Linear(input_dim_per_actor * num_actors, hidden_dim)
            rnn_cls = {
                "rnn": nn.RNN,
                "lstm": nn.LSTM,
                "gru": nn.GRU,
            }[encoder_type]
            self.rnn = rnn_cls(
                hidden_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.output_proj = nn.Linear(hidden_dim, latent_dim)

        elif encoder_type == "mlp":
            # Simple MLP encoder
            self.mlp = nn.Sequential(
                nn.Linear(input_dim_per_actor * num_actors * sequence_length, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )

        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(self, multi_actor_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_actor_obs: [B, num_actors, seq_len, obs_dim]

        Returns:
            latent: [B, latent_dim]
        """
        B, N, T, D = multi_actor_obs.shape

        if self.encoder_type == "transformer":
            # Average pool across actors, then process sequence with transformer
            obs_pooled = multi_actor_obs.mean(dim=1)  # [B, T, D]
            x = self.input_proj(obs_pooled)  # [B, T, hidden]
            x = x + self.pos_embedding[:, :T, :]
            x = self.transformer(x)  # [B, T, hidden]
            x = x.mean(dim=1)  # Global average pooling [B, hidden]
            latent = self.output_proj(x)  # [B, latent_dim]

        elif self.encoder_type in ["rnn", "lstm", "gru"]:
            # Concatenate actors and process with RNN
            obs_concat = multi_actor_obs.reshape(B, T, N * D)  # [B, T, N*D]
            x = self.input_proj(obs_concat)  # [B, T, hidden]
            if self.encoder_type == "lstm":
                output, (h_n, c_n) = self.rnn(x)
                x = h_n[-1]  # Use last hidden state
            else:
                output, h_n = self.rnn(x)
                x = h_n[-1]
            latent = self.output_proj(x)  # [B, latent_dim]

        elif self.encoder_type == "mlp":
            # Flatten and process with MLP
            x = multi_actor_obs.reshape(B, -1)  # [B, N*T*D]
            latent = self.mlp(x)  # [B, latent_dim]

        return latent


class PolicyAdapter(nn.Module):
    """
    Role-conditioned policy adapter

    Implements lightweight adapter strategies:
    - LoRA (Low-Rank Adaptation)
    - IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
    - Linear (simple linear projection)
    """

    def __init__(
        self,
        adapter_type: str,
        input_dim: int,
        output_dim: int,
        rank: int = 16,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.adapter_type = adapter_type

        if adapter_type == "lora":
            # LoRA: y = Wx + (BA)x, where B is (d, r) and A is (r, k)
            self.lora_A = nn.Parameter(torch.randn(rank, input_dim) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(output_dim, rank))
            self.scaling = alpha / rank

        elif adapter_type == "ia3":
            # IA3: Element-wise learned gating
            self.gate = nn.Parameter(torch.ones(output_dim))

        elif adapter_type == "linear":
            # Simple linear adapter
            self.linear = nn.Linear(input_dim, output_dim)

        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def forward(self, x: torch.Tensor, base_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features [B, input_dim]
            base_output: Base model output [B, output_dim] (for LoRA/IA3)

        Returns:
            Adapted output [B, output_dim]
        """
        if self.adapter_type == "lora":
            # Add low-rank update to base output
            assert base_output is not None, "LoRA requires base_output"
            delta = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
            return base_output + delta

        elif self.adapter_type == "ia3":
            # Scale base output with learned gate
            assert base_output is not None, "IA3 requires base_output"
            return base_output * self.gate

        elif self.adapter_type == "linear":
            # Direct projection
            return self.linear(x)


class RoleConditionedPolicy(nn.Module):
    """
    Role-conditioned cooperative policy

    Architecture:
    - Shared base encoder
    - Role-specific adapters
    - Coordination latent conditioning
    """

    def __init__(
        self,
        role_configs: Dict[str, Dict],  # role_id -> {obs_dim, action_dim}
        coordination_latent_dim: int,
        adapter_type: str = "lora",
        hidden_dims: List[int] = [512, 512, 256],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.role_configs = role_configs
        self.role_ids = list(role_configs.keys())

        # Shared base encoder (processes raw observations)
        example_role = list(role_configs.values())[0]
        obs_dim = example_role["obs_dim"]

        layers = []
        in_dim = obs_dim + coordination_latent_dim  # Concatenate obs + z
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.shared_encoder = nn.Sequential(*layers)
        self.shared_output_dim = hidden_dims[-1]

        # Role-specific adapters
        self.adapters = nn.ModuleDict()
        for role_id, config in role_configs.items():
            self.adapters[role_id] = PolicyAdapter(
                adapter_type=adapter_type,
                input_dim=self.shared_output_dim,
                output_dim=config["action_dim"],
            )

    def forward(
        self,
        observations: Dict[str, torch.Tensor],  # role_id -> [B, obs_dim]
        coordination_latent: torch.Tensor,  # [B, coord_dim]
    ) -> Dict[str, torch.Tensor]:  # role_id -> [B, action_dim]
        """
        Forward pass for all roles

        Args:
            observations: Dict of per-role observations
            coordination_latent: Shared coordination latent

        Returns:
            Dict of per-role actions
        """
        actions = {}

        for role_id in self.role_ids:
            if role_id not in observations:
                continue

            obs = observations[role_id]  # [B, obs_dim]
            B = obs.shape[0]

            # Concatenate observation with coordination latent
            coord_latent_expanded = coordination_latent[:B]  # Ensure matching batch
            x = torch.cat([obs, coord_latent_expanded], dim=-1)

            # Pass through shared encoder
            shared_features = self.shared_encoder(x)  # [B, shared_dim]

            # Apply role-specific adapter
            if self.adapters[role_id].adapter_type == "linear":
                action = self.adapters[role_id](shared_features)
            else:
                # For LoRA/IA3, we need a base output (here we use a simple projection)
                base_action = torch.zeros(
                    B, self.role_configs[role_id]["action_dim"], device=obs.device
                )
                action = self.adapters[role_id](shared_features, base_action)

            actions[role_id] = action

        return actions


class MultiActorDataset(Dataset):
    """
    Dataset for multi-actor cooperative demonstrations

    Loads demonstrations in robomimic or LeRobot format and provides
    synchronized multi-actor trajectories.
    """

    def __init__(
        self,
        dataset_path: str,
        role_ids: List[str],
        sequence_length: int = 16,
        format: str = "robomimic",
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.role_ids = role_ids
        self.sequence_length = sequence_length
        self.format = format

        # Load demonstrations
        self.demonstrations = self._load_demonstrations()

        # Build index: (demo_id, timestep)
        self.index = []
        for demo_id, demo in enumerate(self.demonstrations):
            demo_len = demo["length"]
            for t in range(demo_len - sequence_length + 1):
                self.index.append((demo_id, t))

    def _load_demonstrations(self) -> List[Dict]:
        """Load demonstrations from file"""
        # TODO: Implement actual loading from robomimic/LeRobot formats
        # For now, return mock structure
        return [
            {
                "length": 100,
                "observations": {role_id: torch.randn(100, 10) for role_id in self.role_ids},
                "actions": {role_id: torch.randn(100, 7) for role_id in self.role_ids},
            }
            for _ in range(10)  # 10 demo trajectories
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "multi_actor_obs": [num_actors, seq_len, obs_dim],
                "observations": {role_id: [seq_len, obs_dim]},
                "actions": {role_id: [seq_len, action_dim]},
                "current_obs": {role_id: [obs_dim]},
                "current_action": {role_id: [action_dim]},
            }
        """
        demo_id, start_t = self.index[idx]
        demo = self.demonstrations[demo_id]

        # Extract sequences
        end_t = start_t + self.sequence_length
        multi_actor_obs_list = []
        observations = {}
        actions = {}
        current_obs = {}
        current_action = {}

        for role_id in self.role_ids:
            obs_seq = demo["observations"][role_id][start_t:end_t]
            action_seq = demo["actions"][role_id][start_t:end_t]

            multi_actor_obs_list.append(obs_seq.unsqueeze(0))  # [1, seq_len, obs_dim]
            observations[role_id] = obs_seq
            actions[role_id] = action_seq
            current_obs[role_id] = obs_seq[-1]  # Last observation
            current_action[role_id] = action_seq[-1]  # Last action

        multi_actor_obs = torch.cat(multi_actor_obs_list, dim=0)  # [num_actors, seq_len, obs_dim]

        return {
            "multi_actor_obs": multi_actor_obs,
            "observations": observations,
            "actions": actions,
            "current_obs": current_obs,
            "current_action": current_action,
        }
