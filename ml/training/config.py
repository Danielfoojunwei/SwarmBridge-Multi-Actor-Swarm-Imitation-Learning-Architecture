"""Training configuration schema"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training configuration for cooperative BC"""

    # Dataset
    dataset_path: Path
    dataset_format: str = "robomimic"  # "robomimic" or "lerobot"
    num_demonstrations: int = Field(ge=1)
    train_split: float = Field(default=0.8, ge=0.0, le=1.0)

    # Model architecture
    shared_encoder_dim: int = 256
    coordination_latent_dim: int = 64
    coordination_encoder_type: str = "transformer"  # transformer, rnn, mlp
    coordination_sequence_length: int = 16
    policy_adapter_type: str = "lora"  # lora, ia3, linear
    policy_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])

    # Roles
    roles: Dict[str, Dict] = field(default_factory=dict)  # role_id -> {obs_dim, action_dim}

    # Training hyperparameters
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    # Optimization
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"  # cosine, step, none
    warmup_steps: int = 1000

    # Coordination loss
    coordination_loss_weight: float = 0.1
    consistency_loss_weight: float = 0.05

    # Regularization
    dropout: float = 0.1
    action_l2_weight: float = 0.01

    # Logging and checkpointing
    output_dir: Path = Path("outputs/training")
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    use_wandb: bool = False
    wandb_project: Optional[str] = None

    # Hardware
    device: str = "cuda"  # cuda, cpu
    num_workers: int = 4
    mixed_precision: bool = True

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    class Config:
        arbitrary_types_allowed = True
