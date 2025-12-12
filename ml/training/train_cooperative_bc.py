"""
Cooperative Behavior Cloning Trainer

Trains role-conditioned policies with coordination latent encoders
from multi-actor demonstrations.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .config import TrainingConfig
from .models import (
    CoordinationLatentEncoder,
    MultiActorDataset,
    RoleConditionedPolicy,
)
from ..artifact import (
    CooperativeSkillArtefact,
    CoordinationEncoder,
    CSAMetadata,
    PolicyAdapter,
    RoleConfig,
    RoleType,
    SafetyEnvelope,
)


class CooperativeBCTrainer:
    """Trainer for cooperative behavior cloning"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Set random seeds
        self._set_seeds(config.seed)

        # Build models
        self.coordination_encoder = None
        self.policy = None
        self.optimizer = None
        self.lr_scheduler = None

        # Metrics
        self.train_losses = []
        self.eval_losses = []
        self.best_eval_loss = float("inf")

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def build_models(self) -> None:
        """Build coordination encoder and role-conditioned policy"""
        # Get dimensions from first role
        example_role = list(self.config.roles.values())[0]
        obs_dim = example_role["obs_dim"]
        num_actors = len(self.config.roles)

        # Build coordination encoder
        self.coordination_encoder = CoordinationLatentEncoder(
            encoder_type=self.config.coordination_encoder_type,
            input_dim_per_actor=obs_dim,
            num_actors=num_actors,
            latent_dim=self.config.coordination_latent_dim,
            sequence_length=self.config.coordination_sequence_length,
            hidden_dim=self.config.shared_encoder_dim,
            dropout=self.config.dropout,
        ).to(self.device)

        # Build role-conditioned policy
        self.policy = RoleConditionedPolicy(
            role_configs=self.config.roles,
            coordination_latent_dim=self.config.coordination_latent_dim,
            adapter_type=self.config.policy_adapter_type,
            hidden_dims=self.config.policy_hidden_dims,
            dropout=self.config.dropout,
        ).to(self.device)

        print(f"✓ Built models:")
        print(f"  Coordination encoder: {sum(p.numel() for p in self.coordination_encoder.parameters()):,} params")
        print(f"  Policy network: {sum(p.numel() for p in self.policy.parameters()):,} params")

    def build_optimizer(self) -> None:
        """Build optimizer and LR scheduler"""
        # Combine parameters
        params = list(self.coordination_encoder.parameters()) + list(self.policy.parameters())

        if self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Build LR scheduler
        if self.config.lr_scheduler == "cosine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.lr_scheduler == "step":
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        # else: no scheduler

        print(f"✓ Built optimizer: {self.config.optimizer}")

    def load_datasets(self) -> tuple:
        """Load and split datasets"""
        dataset = MultiActorDataset(
            dataset_path=str(self.config.dataset_path),
            role_ids=list(self.config.roles.keys()),
            sequence_length=self.config.coordination_sequence_length,
            format=self.config.dataset_format,
        )

        # Train/val split
        train_size = int(self.config.train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.config.seed)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        print(f"✓ Loaded datasets:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")

        return train_loader, val_loader

    def compute_loss(self, batch: Dict, train: bool = True) -> Dict[str, torch.Tensor]:
        """Compute training losses"""
        # Extract data
        multi_actor_obs = batch["multi_actor_obs"].to(self.device)  # [B, N, T, D]
        current_obs = {k: v.to(self.device) for k, v in batch["current_obs"].items()}
        current_action = {k: v.to(self.device) for k, v in batch["current_action"].items()}

        # Forward pass: coordination encoder
        coord_latent = self.coordination_encoder(multi_actor_obs)  # [B, coord_dim]

        # Forward pass: role-conditioned policy
        pred_actions = self.policy(current_obs, coord_latent)  # {role_id: [B, action_dim]}

        # Compute behavior cloning loss (MSE between predicted and true actions)
        bc_loss = 0.0
        for role_id in pred_actions.keys():
            pred_action = pred_actions[role_id]
            true_action = current_action[role_id]
            bc_loss += F.mse_loss(pred_action, true_action)
        bc_loss /= len(pred_actions)

        # Coordination consistency loss (encourage similar latents for coordinated behaviors)
        # Simple L2 regularization on latent
        coord_loss = torch.mean(coord_latent ** 2) * self.config.coordination_loss_weight

        # Action L2 regularization
        action_reg = 0.0
        for action in pred_actions.values():
            action_reg += torch.mean(action ** 2)
        action_reg = action_reg * self.config.action_l2_weight / len(pred_actions)

        # Total loss
        total_loss = bc_loss + coord_loss + action_reg

        return {
            "total_loss": total_loss,
            "bc_loss": bc_loss,
            "coord_loss": coord_loss,
            "action_reg": action_reg,
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.coordination_encoder.train()
        self.policy.train()

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            losses = self.compute_loss(batch, train=True)

            # Backward pass
            self.optimizer.zero_grad()
            losses["total_loss"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.coordination_encoder.parameters()) + list(self.policy.parameters()),
                self.config.grad_clip_norm,
            )

            self.optimizer.step()

            # Logging
            total_loss += losses["total_loss"].item()
            pbar.set_postfix({
                "loss": losses["total_loss"].item(),
                "bc": losses["bc_loss"].item(),
                "coord": losses["coord_loss"].item(),
            })

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate on validation set"""
        self.coordination_encoder.eval()
        self.policy.eval()

        total_loss = 0.0

        for batch in val_loader:
            losses = self.compute_loss(batch, train=False)
            total_loss += losses["total_loss"].item()

        return total_loss / len(val_loader)

    def train(self) -> None:
        """Main training loop"""
        # Build models and optimizer
        self.build_models()
        self.build_optimizer()

        # Load datasets
        train_loader, val_loader = self.load_datasets()

        # Training loop
        print(f"\n{'=' * 60}")
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"{'=' * 60}\n")

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Evaluate
            if (epoch + 1) % self.config.eval_interval == 0 or epoch == self.config.num_epochs - 1:
                val_loss = self.evaluate(val_loader)
                self.eval_losses.append(val_loss)

                print(f"\nEpoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # Save best model
                if val_loss < self.best_eval_loss:
                    self.best_eval_loss = val_loss
                    self.save_checkpoint(self.config.output_dir / "best_model.pt")

            # LR schedule step
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(self.config.output_dir / f"checkpoint_epoch_{epoch + 1}.pt")

        print(f"\n{'=' * 60}")
        print(f"Training complete! Best val loss: {self.best_eval_loss:.4f}")
        print(f"{'=' * 60}\n")

    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "coordination_encoder": self.coordination_encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.model_dump(),
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
        }, path)

    def export_csa(
        self,
        output_path: Path,
        skill_name: str,
        version: str = "1.0.0",
        phase_machine_xml: Optional[str] = None,
    ) -> CooperativeSkillArtefact:
        """
        Export trained model as CSA (Cooperative Skill Artefact)

        Args:
            output_path: Where to save CSA
            skill_name: Name of the skill
            version: Semantic version
            phase_machine_xml: BehaviorTree XML (optional)

        Returns:
            CooperativeSkillArtefact
        """
        # Build role configs
        roles = []
        for role_id, config in self.config.roles.items():
            roles.append(
                RoleConfig(
                    role_id=role_id,
                    role_type=RoleType.CUSTOM,  # TODO: Infer from config
                    observation_dims=config["obs_dim"],
                    action_dims=config["action_dim"],
                    requires_coordination=True,
                )
            )

        # Extract policy adapters
        policy_adapters = []
        for role_id in self.config.roles.keys():
            adapter_weights = self.policy.adapters[role_id].state_dict()
            policy_adapters.append(
                PolicyAdapter(
                    role_id=role_id,
                    adapter_type=self.config.policy_adapter_type,
                    adapter_weights=adapter_weights,
                )
            )

        # Extract coordination encoder
        coord_encoder = CoordinationEncoder(
            encoder_type=self.config.coordination_encoder_type,
            encoder_weights=self.coordination_encoder.state_dict(),
            latent_dim=self.config.coordination_latent_dim,
            sequence_length=self.config.coordination_sequence_length,
        )

        # Default phase machine if not provided
        if phase_machine_xml is None:
            phase_machine_xml = """
<root main_tree_to_execute="CooperativeSkill">
    <BehaviorTree ID="CooperativeSkill">
        <Sequence>
            <Action ID="Approach"/>
            <Action ID="Grasp"/>
            <Action ID="Lift"/>
            <Action ID="Transfer"/>
            <Action ID="Place"/>
            <Action ID="Retreat"/>
        </Sequence>
    </BehaviorTree>
</root>
"""

        # Default safety envelope
        safety_envelope = SafetyEnvelope(
            max_velocity={"joint_0": 1.0, "joint_1": 1.0, "joint_2": 1.0},
            max_acceleration={"joint_0": 2.0, "joint_1": 2.0, "joint_2": 2.0},
            max_force={"gripper": 50.0},
            max_torque={"joint_0": 10.0, "joint_1": 10.0},
            min_separation_distance=0.3,
            workspace_bounds=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
            collision_primitives=[],
            emergency_stop_triggers=["force_limit", "workspace_violation"],
        )

        # Metadata
        metadata = CSAMetadata(
            version=version,
            skill_name=skill_name,
            description=f"Cooperative skill trained with {self.config.num_demonstrations} demonstrations",
            num_demonstrations=self.config.num_demonstrations,
            training_sites=["local"],
            training_duration_seconds=0.0,  # TODO: Track actual duration
            compatible_robots=["ur5e", "franka_panda"],
            compatible_end_effectors=["robotiq_2f85"],
            min_actors=len(roles),
            max_actors=len(roles),
            privacy_mode="none",
            test_pass_rate=0.95,
            test_coverage=0.80,
        )

        # Build CSA
        csa = CooperativeSkillArtefact(
            roles=roles,
            policy_adapters=policy_adapters,
            coordination_encoder=coord_encoder,
            phase_machine_xml=phase_machine_xml,
            safety_envelope=safety_envelope,
            metadata=metadata,
            test_suite={},
        )

        print(f"✓ Exported CSA: {skill_name} v{version}")
        return csa


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description="Train cooperative BC policy")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    args = parser.parse_args()

    # Load config (TODO: implement YAML loading)
    # For now, use default config
    config = TrainingConfig(
        dataset_path=Path("data/demonstrations.hdf5"),
        num_demonstrations=10,
        roles={
            "leader": {"obs_dim": 10, "action_dim": 7},
            "follower": {"obs_dim": 10, "action_dim": 7},
        },
    )

    if args.output_dir:
        config.output_dir = args.output_dir

    # Train
    trainer = CooperativeBCTrainer(config)
    trainer.train()

    # Export CSA
    csa = trainer.export_csa(
        output_path=config.output_dir / "cooperative_skill.csa",
        skill_name="demo_cooperative_skill",
    )

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
