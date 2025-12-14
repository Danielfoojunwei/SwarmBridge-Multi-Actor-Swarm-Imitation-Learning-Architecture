"""
SwarmBridge End-to-End Pipeline

Modular pipeline for:
1. Multi-actor demonstration capture (ROS 2)
2. Perception processing
3. Cooperative imitation learning
4. CSA packaging
5. Registry publishing

Example:
    pipeline = SwarmBridgePipeline(
        registry_url="http://localhost:8000",
        federated_service_url="http://localhost:8001",
    )

    # Complete pipeline
    csa_id = pipeline.run_complete_pipeline(
        skill_name="cooperative_assembly",
        demonstration_topic="/swarm/demonstrations",
        num_demonstrations=3,
        num_actors=2,
    )
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

import torch
import numpy as np


@dataclass
class PipelineConfig:
    """Configuration for SwarmBridge pipeline"""
    # Capture
    ros2_workspace: Path = Path("ros2_ws")
    demonstration_topic: str = "/swarm/demonstrations"
    perception_topics: List[str] = None

    # Training
    training_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4

    # CSA Packaging
    csa_output_dir: Path = Path("artifacts")

    # Registry
    registry_url: str = "http://localhost:8000"
    federated_service_url: str = "http://localhost:8001"

    def __post_init__(self):
        if self.perception_topics is None:
            self.perception_topics = [
                "/camera/rgb/image_raw",
                "/camera/depth/image_raw",
            ]


class SwarmBridgePipeline:
    """
    End-to-end pipeline for multi-actor skill learning.

    Pipeline Stages:
    1. CAPTURE: Record multi-actor demonstrations via ROS 2
    2. PROCESS: Extract observations and actions from recordings
    3. TRAIN: Cooperative imitation learning
    4. PACKAGE: Create CSA artifact
    5. PUBLISH: Upload to registry

    Optionally:
    6. FEDERATE: Submit to federated learning service
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        registry_url: Optional[str] = None,
        federated_service_url: Optional[str] = None,
    ):
        self.config = config or PipelineConfig()

        if registry_url:
            self.config.registry_url = registry_url
        if federated_service_url:
            self.config.federated_service_url = federated_service_url

    async def run_complete_pipeline(
        self,
        skill_name: str,
        num_demonstrations: int = 3,
        num_actors: int = 2,
        roles: Optional[List[str]] = None,
        coordination_type: str = "handover",
        enable_federated_learning: bool = False,
    ) -> str:
        """
        Execute complete pipeline from capture to registry.

        Args:
            skill_name: Name of skill to learn
            num_demonstrations: Number of demonstrations to collect
            num_actors: Number of actors in demonstrations
            roles: Role names (default: ["leader", "follower"])
            coordination_type: Coordination primitive
            enable_federated_learning: Submit to federated service

        Returns:
            CSA ID in registry
        """

        if roles is None:
            roles = ["leader", "follower"] if num_actors == 2 else [f"actor_{i}" for i in range(num_actors)]

        print(f"\n{'='*60}")
        print(f"SWARMBRIDGE PIPELINE: {skill_name}")
        print(f"{'='*60}\n")

        # STAGE 1: CAPTURE demonstrations
        print("STAGE 1/5: CAPTURE multi-actor demonstrations")
        demonstrations = await self.capture_demonstrations(
            skill_name=skill_name,
            num_demonstrations=num_demonstrations,
            num_actors=num_actors,
        )
        print(f"  ✓ Captured {len(demonstrations)} demonstrations\n")

        # STAGE 2: PROCESS demonstrations
        print("STAGE 2/5: PROCESS demonstrations")
        processed_data = await self.process_demonstrations(
            demonstrations=demonstrations,
            num_actors=num_actors,
        )
        print(f"  ✓ Processed {processed_data['num_trajectories']} trajectories\n")

        # STAGE 3: TRAIN cooperative policy
        print("STAGE 3/5: TRAIN cooperative imitation learning")
        trained_model = await self.train_cooperative_policy(
            dataset=processed_data,
            skill_name=skill_name,
            roles=roles,
            coordination_type=coordination_type,
        )
        print(f"  ✓ Training complete\n")

        # STAGE 4: PACKAGE as CSA
        print("STAGE 4/5: PACKAGE as CSA artifact")
        csa_path = await self.package_csa(
            skill_name=skill_name,
            model=trained_model,
            roles=roles,
            coordination_type=coordination_type,
        )
        print(f"  ✓ CSA packaged: {csa_path}\n")

        # STAGE 5: PUBLISH to registry
        print("STAGE 5/5: PUBLISH to registry")
        csa_id = await self.publish_to_registry(
            csa_path=csa_path,
            skill_name=skill_name,
        )
        print(f"  ✓ Published: {csa_id}\n")

        # OPTIONAL: FEDERATE
        if enable_federated_learning:
            print("OPTIONAL: FEDERATE with learning service")
            await self.submit_to_federated_learning(
                csa_id=csa_id,
                skill_name=skill_name,
            )
            print(f"  ✓ Submitted to federated learning\n")

        print(f"{'='*60}")
        print(f"PIPELINE COMPLETE: {csa_id}")
        print(f"{'='*60}\n")

        return csa_id

    async def capture_demonstrations(
        self,
        skill_name: str,
        num_demonstrations: int,
        num_actors: int,
    ) -> List[Dict[str, Any]]:
        """
        STAGE 1: Capture multi-actor demonstrations via ROS 2.

        Uses:
        - swarm_capture package (ROS 2)
        - swarm_perception package (MMPose, ONVIF)

        Returns:
            List of demonstration recordings
        """

        print(f"  Starting ROS 2 capture for {num_demonstrations} demonstrations...")
        print(f"  Actors: {num_actors}")
        print(f"  Topics: {self.config.demonstration_topic}")

        # Import capture module
        from .capture import ROS2DemonstrationCapture

        capture = ROS2DemonstrationCapture(
            workspace=self.config.ros2_workspace,
            topic=self.config.demonstration_topic,
        )

        demonstrations = await capture.record_demonstrations(
            skill_name=skill_name,
            num_demonstrations=num_demonstrations,
            num_actors=num_actors,
        )

        return demonstrations

    async def process_demonstrations(
        self,
        demonstrations: List[Dict[str, Any]],
        num_actors: int,
    ) -> Dict[str, Any]:
        """
        STAGE 2: Process demonstrations into training data.

        Extracts:
        - Multi-actor observations (joint positions, poses)
        - Multi-actor actions (velocities, commands)
        - Coordination context

        Returns:
            Processed dataset
        """

        print(f"  Processing {len(demonstrations)} demonstrations...")
        print(f"  Extracting multi-actor observations and actions...")

        from .processing import DemonstrationProcessor

        processor = DemonstrationProcessor(num_actors=num_actors)

        processed_data = processor.process_batch(demonstrations)

        print(f"  Extracted {processed_data['num_trajectories']} trajectories")
        print(f"  Total timesteps: {processed_data['total_timesteps']}")

        return processed_data

    async def train_cooperative_policy(
        self,
        dataset: Dict[str, Any],
        skill_name: str,
        roles: List[str],
        coordination_type: str,
    ) -> Dict[str, Any]:
        """
        STAGE 3: Train cooperative imitation learning policy.

        Uses:
        - ml/training/train_cooperative_bc.py
        - Shared role schema
        - Coordination primitives

        Returns:
            Trained model components
        """

        print(f"  Training cooperative policy...")
        print(f"  Roles: {roles}")
        print(f"  Coordination: {coordination_type}")
        print(f"  Epochs: {self.config.training_epochs}")

        # Import training module
        from ml.training.train_cooperative_bc import CooperativeBCTrainer

        trainer = CooperativeBCTrainer(
            dataset=dataset,
            roles=roles,
            coordination_type=coordination_type,
            epochs=self.config.training_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )

        trained_model = trainer.train()

        print(f"  Final loss: {trained_model['final_loss']:.4f}")

        return trained_model

    async def package_csa(
        self,
        skill_name: str,
        model: Dict[str, Any],
        roles: List[str],
        coordination_type: str,
    ) -> Path:
        """
        STAGE 4: Package as Cooperative Skill Artifact (CSA).

        Uses:
        - ml/artifact/packager.py
        - Shared role schema
        - Coordination primitives

        Returns:
            Path to CSA tarball
        """

        print(f"  Packaging CSA: {skill_name}")

        from ml.artifact.packager import CSAPackager
        from ml.artifact.schema import CooperativeSkillArtefact

        packager = CSAPackager()

        # Create CSA
        csa = CooperativeSkillArtefact(
            skill_name=skill_name,
            version="1.0",
            roles=model['roles'],
            policy_adapters=model['adapters'],
            coordination_encoder=model['coordination_encoder'],
            coordination_type=coordination_type,
            metadata=model['metadata'],
        )

        # Package
        output_path = self.config.csa_output_dir / f"{skill_name}_v1.0.tar.gz"
        csa_path = packager.package(csa, output_path)

        print(f"  Packaged: {csa_path}")
        print(f"  Size: {csa_path.stat().st_size / 1024:.1f} KB")

        return csa_path

    async def publish_to_registry(
        self,
        csa_path: Path,
        skill_name: str,
    ) -> str:
        """
        STAGE 5: Publish CSA to shared registry.

        Uses:
        - Registry adapter
        - services/registry API

        Returns:
            CSA ID in registry
        """

        print(f"  Publishing to registry: {self.config.registry_url}")

        from .adapters.registry_adapter import RegistryAdapter

        adapter = RegistryAdapter(registry_url=self.config.registry_url)

        csa_id = await adapter.upload_csa(
            csa_path=csa_path,
            skill_name=skill_name,
        )

        print(f"  Published with ID: {csa_id}")

        return csa_id

    async def submit_to_federated_learning(
        self,
        csa_id: str,
        skill_name: str,
    ):
        """
        OPTIONAL: Submit to federated learning service.

        Uses:
        - Federated learning adapter (abstracts OpenFL)

        This replaces direct OpenFL dependencies.
        """

        print(f"  Submitting to federated learning service...")
        print(f"  Service: {self.config.federated_service_url}")

        from .adapters.federated_adapter import FederatedLearningAdapter

        adapter = FederatedLearningAdapter(
            service_url=self.config.federated_service_url
        )

        await adapter.submit_local_update(
            csa_id=csa_id,
            skill_name=skill_name,
        )

        print(f"  Submitted for federated aggregation")


class PipelineStage:
    """Base class for pipeline stages"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this stage"""
        raise NotImplementedError
