"""
Training Script for Advanced Multi-Actor System

Integrates all enhanced components:
- Hierarchical coordination encoding
- Dynamic role assignment
- Intent communication
- Adaptive policies
- Advanced safety verification
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .advanced_multi_actor import (
    HierarchicalCoordinationEncoder,
    AdaptiveCoordinationPolicy,
    IntentCommunicationModule,
    DynamicRoleAssigner,
    AdvancedMultiActorTrainer,
    MultiActorSafetyVerifier,
    CoordinationMode,
)
from ..datasets.multi_actor_dataset import create_multi_actor_dataloader


class AdvancedMultiActorTrainingPipeline:
    """
    Complete training pipeline for advanced multi-actor IL

    Features:
    - Curriculum learning (2 → N actors)
    - Role diversity optimization
    - Communication efficiency
    - Safety-aware training
    """

    def __init__(
        self,
        dataset_path: Path,
        num_actors: int = 3,
        obs_dim: int = 10,
        action_dim: int = 7,
        coordination_latent_dim: int = 64,
        batch_size: int = 32,
        num_epochs: int = 100,
        device: str = "cuda",
    ):
        self.dataset_path = dataset_path
        self.num_actors = num_actors
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Build models
        self.hierarchical_encoder = HierarchicalCoordinationEncoder(
            num_actors=num_actors,
            obs_dim=obs_dim,
            latent_dim=coordination_latent_dim,
        ).to(device)

        # Role configs (assume uniform for now)
        role_configs = {
            f"actor_{i}": action_dim for i in range(num_actors)
        }

        self.adaptive_policy = AdaptiveCoordinationPolicy(
            role_configs=role_configs,
            coordination_latent_dim=coordination_latent_dim,
            num_coordination_modes=len(CoordinationMode),
        ).to(device)

        self.intent_module = IntentCommunicationModule(
            num_actors=num_actors,
            intent_dim=32,
        ).to(device)

        # Trainer
        self.trainer = AdvancedMultiActorTrainer(
            hierarchical_encoder=self.hierarchical_encoder,
            adaptive_policy=self.adaptive_policy,
            intent_module=self.intent_module,
            learning_rate=1e-4,
        )

        # Safety verifier
        self.safety_verifier = MultiActorSafetyVerifier(
            min_separation=0.5,
            max_relative_velocity=0.3,
        )

        # Dataloader
        self.train_loader = create_multi_actor_dataloader(
            dataset_path=dataset_path,
            batch_size=batch_size,
            num_workers=4,
            formation_balanced=True,
        )

        # Metrics tracking
        self.metrics_history = {
            "total_loss": [],
            "bc_loss": [],
            "intent_loss": [],
            "consistency_loss": [],
            "comm_efficiency_loss": [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.hierarchical_encoder.train()
        self.adaptive_policy.train()
        self.intent_module.train()

        epoch_metrics = {
            "total_loss": 0.0,
            "bc_loss": 0.0,
            "intent_loss": 0.0,
            "consistency_loss": 0.0,
            "comm_efficiency_loss": 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            observations = batch['multi_actor_observations'].to(self.device)
            actions_dict = {
                f"actor_{i}": batch['multi_actor_actions'][:, i, -1, :].to(self.device)
                for i in range(batch['num_actors'][0])
            }
            intents = batch['multi_actor_intents'].to(self.device)

            # Forward pass and compute loss
            losses = self.trainer.compute_loss(observations, actions_dict, intents)

            # Backward pass
            self.trainer.optimizer.zero_grad()
            losses["total_loss"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.hierarchical_encoder.parameters())
                + list(self.adaptive_policy.parameters())
                + list(self.intent_module.parameters()),
                max_norm=1.0,
            )

            self.trainer.optimizer.step()

            # Accumulate metrics
            for key in epoch_metrics.keys():
                epoch_metrics[key] += losses[key].item()

            # Update progress bar
            pbar.set_postfix({
                "loss": losses["total_loss"].item(),
                "bc": losses["bc_loss"].item(),
                "intent": losses["intent_loss"].item(),
            })

        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def train(self) -> None:
        """Main training loop"""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Dataset: {self.dataset_path}")
        print(f"Num actors: {self.num_actors}")
        print(f"Device: {self.device}")
        print()

        for epoch in range(self.num_epochs):
            metrics = self.train_epoch(epoch)

            # Log metrics
            for key, value in metrics.items():
                self.metrics_history[key].append(value)

            # Print summary
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Total Loss:      {metrics['total_loss']:.4f}")
                print(f"  BC Loss:         {metrics['bc_loss']:.4f}")
                print(f"  Intent Loss:     {metrics['intent_loss']:.4f}")
                print(f"  Consistency:     {metrics['consistency_loss']:.4f}")
                print(f"  Comm Efficiency: {metrics['comm_efficiency_loss']:.4f}")

            # Save checkpoint
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        print("\n✓ Training complete!")

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint"""
        checkpoint = {
            "hierarchical_encoder": self.hierarchical_encoder.state_dict(),
            "adaptive_policy": self.adaptive_policy.state_dict(),
            "intent_module": self.intent_module.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
        }

        save_path = Path("outputs") / "advanced_multi_actor" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        print(f"✓ Saved checkpoint: {save_path}")

    def export_for_deployment(self, output_path: Path) -> None:
        """
        Export trained models for deployment

        Creates package with:
        - Hierarchical encoder
        - Adaptive policy
        - Intent module
        - Safety configuration
        """
        deployment_package = {
            "hierarchical_encoder": self.hierarchical_encoder.state_dict(),
            "adaptive_policy": self.adaptive_policy.state_dict(),
            "intent_module": self.intent_module.state_dict(),
            "config": {
                "num_actors": self.num_actors,
                "coordination_latent_dim": self.hierarchical_encoder.latent_dim,
                "num_coordination_modes": len(CoordinationMode),
            },
            "safety_config": {
                "min_separation": self.safety_verifier.min_separation,
                "max_relative_velocity": self.safety_verifier.max_relative_velocity,
                "formation_tolerance": self.safety_verifier.formation_tolerance,
            },
        }

        torch.save(deployment_package, output_path)
        print(f"✓ Exported deployment package: {output_path}")


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="Train advanced multi-actor coordination system"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to multi-actor dataset (HDF5)",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=3,
        help="Number of actors",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=10,
        help="Observation dimension per actor",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action dimension per actor",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training device",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/advanced_multi_actor"),
        help="Output directory",
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = AdvancedMultiActorTrainingPipeline(
        dataset_path=args.dataset,
        num_actors=args.num_actors,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
    )

    # Train
    pipeline.train()

    # Export
    output_path = args.output_dir / "deployment_package.pt"
    pipeline.export_for_deployment(output_path)


if __name__ == "__main__":
    main()
