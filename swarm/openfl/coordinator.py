"""
OpenFL Swarm Coordinator

Orchestrates federated training rounds across multiple sites with
privacy-preserving aggregation.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .aggregator import AggregationStrategy, RobustAggregator
from ..privacy import PrivacyMode, PrivacyEngine
from ...ml.artifact import CSALoader, CSAPackager, CooperativeSkillArtefact

logger = logging.getLogger(__name__)


@dataclass
class SwarmRoundConfig:
    """Configuration for a swarm training round"""

    round_id: str
    privacy_mode: str  # "ldp", "dp_sgd", "he", "none"
    aggregation_strategy: str  # "mean", "trimmed_mean", "median", "krum"
    min_participants: int = 2
    max_participants: int = 10
    timeout_seconds: int = 3600
    staleness_tolerance: int = 1  # Max rounds a site can lag

    # Privacy parameters
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    noise_multiplier: Optional[float] = None
    clip_norm: Optional[float] = None


class SwarmCoordinator:
    """
    Coordinate privacy-preserving swarm learning rounds

    Responsibilities:
    - Manage participant registration
    - Orchestrate training rounds
    - Apply privacy mechanisms
    - Perform robust aggregation
    - Emit merged CSA artifacts
    """

    def __init__(
        self,
        registry_url: str,
        workspace_dir: Path,
        privacy_engine: Optional[PrivacyEngine] = None,
    ):
        self.registry_url = registry_url
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.privacy_engine = privacy_engine or PrivacyEngine()
        self.aggregator = RobustAggregator()

        # Round state
        self.current_round_id: Optional[str] = None
        self.participants: Dict[str, Dict] = {}
        self.site_contributions: Dict[str, List[str]] = {}  # site_id -> list of round_ids

        logger.info(f"Initialized SwarmCoordinator with workspace: {workspace_dir}")

    async def register_participant(
        self, site_id: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Register a site as participant

        Args:
            site_id: Unique site identifier
            metadata: Site metadata (capabilities, privacy preferences, etc.)

        Returns:
            Registration response
        """
        if site_id in self.participants:
            logger.warning(f"Site {site_id} already registered, updating metadata")

        self.participants[site_id] = {
            "site_id": site_id,
            "registered_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "active": True,
        }

        logger.info(f"Registered participant: {site_id}")
        return {"status": "registered", "site_id": site_id}

    async def start_round(self, config: SwarmRoundConfig) -> str:
        """
        Start a new swarm training round

        Args:
            config: Round configuration

        Returns:
            Round ID
        """
        if self.current_round_id:
            logger.warning(f"Round {self.current_round_id} still active, completing first")
            await self.complete_round()

        self.current_round_id = config.round_id

        # Create round directory
        round_dir = self.workspace_dir / config.round_id
        round_dir.mkdir(parents=True, exist_ok=True)

        # Save round config
        with open(round_dir / "config.json", "w") as f:
            json.dump(
                {
                    "round_id": config.round_id,
                    "privacy_mode": config.privacy_mode,
                    "aggregation_strategy": config.aggregation_strategy,
                    "min_participants": config.min_participants,
                    "started_at": datetime.utcnow().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(f"Started round: {config.round_id}")
        logger.info(f"  Privacy mode: {config.privacy_mode}")
        logger.info(f"  Aggregation: {config.aggregation_strategy}")

        return config.round_id

    async def submit_update(
        self,
        round_id: str,
        site_id: str,
        csa_delta: CooperativeSkillArtefact,
    ) -> Dict[str, Any]:
        """
        Submit CSA update from a site

        Args:
            round_id: Round identifier
            site_id: Site identifier
            csa_delta: CSA delta (adapters + coordination updates)

        Returns:
            Submission response
        """
        if round_id != self.current_round_id:
            return {"status": "error", "message": f"Round {round_id} not active"}

        if site_id not in self.participants:
            return {"status": "error", "message": f"Site {site_id} not registered"}

        # Save CSA delta
        round_dir = self.workspace_dir / round_id
        delta_path = round_dir / f"{site_id}_delta.tar.gz"

        packager = CSAPackager(output_dir=round_dir)
        packager.package(csa_delta, output_name=delta_path.name)

        # Record contribution
        if site_id not in self.site_contributions:
            self.site_contributions[site_id] = []
        self.site_contributions[site_id].append(round_id)

        logger.info(f"Received update from {site_id} for round {round_id}")

        return {"status": "accepted", "round_id": round_id, "site_id": site_id}

    async def aggregate_round(
        self, round_id: str, config: SwarmRoundConfig
    ) -> CooperativeSkillArtefact:
        """
        Aggregate CSA deltas with privacy and robustness

        Args:
            round_id: Round identifier
            config: Round configuration

        Returns:
            Merged CSA artifact
        """
        round_dir = self.workspace_dir / round_id

        # Load all CSA deltas
        csa_deltas = []
        site_ids = []

        for delta_file in round_dir.glob("*_delta.tar.gz"):
            site_id = delta_file.stem.replace("_delta", "")
            loader = CSALoader()
            csa_delta = loader.load(delta_file)
            csa_deltas.append(csa_delta)
            site_ids.append(site_id)

        if len(csa_deltas) < config.min_participants:
            raise ValueError(
                f"Insufficient participants: {len(csa_deltas)} < {config.min_participants}"
            )

        logger.info(f"Aggregating {len(csa_deltas)} updates for round {round_id}")

        # Extract adapter weights for aggregation
        aggregated_adapters = {}
        for role_id in csa_deltas[0].policy_adapters[0].role_id:
            # Collect weights from all sites for this role
            role_weights = []
            for csa in csa_deltas:
                adapter = csa.get_role_adapter(role_id)
                if adapter:
                    role_weights.append(adapter.adapter_weights)

            # Apply privacy mechanism
            if config.privacy_mode != "none":
                role_weights = self.privacy_engine.apply_privacy(
                    role_weights,
                    mode=PrivacyMode(config.privacy_mode),
                    epsilon=config.epsilon,
                    delta=config.delta,
                    clip_norm=config.clip_norm,
                )

            # Robust aggregation
            aggregated_weights = self.aggregator.aggregate(
                role_weights, strategy=AggregationStrategy(config.aggregation_strategy)
            )

            aggregated_adapters[role_id] = aggregated_weights

        # Build merged CSA (using first delta as template)
        base_csa = csa_deltas[0]

        # Update adapter weights with aggregated values
        for adapter in base_csa.policy_adapters:
            if adapter.role_id in aggregated_adapters:
                adapter.adapter_weights = aggregated_adapters[adapter.role_id]

        # Update metadata
        base_csa.metadata.training_sites = site_ids
        base_csa.metadata.updated_at = datetime.utcnow()
        base_csa.metadata.privacy_mode = config.privacy_mode
        if config.epsilon:
            base_csa.metadata.epsilon = config.epsilon
        if config.delta:
            base_csa.metadata.delta = config.delta

        logger.info(f"✓ Aggregated CSA for round {round_id}")
        logger.info(f"  Privacy mode: {config.privacy_mode}")
        logger.info(f"  Participants: {len(site_ids)}")

        return base_csa

    async def complete_round(self) -> None:
        """Mark current round as complete"""
        if self.current_round_id:
            logger.info(f"Completed round: {self.current_round_id}")
            self.current_round_id = None

    async def run_round(
        self, config: SwarmRoundConfig, csa_deltas: List[CooperativeSkillArtefact]
    ) -> CooperativeSkillArtefact:
        """
        Run complete swarm round (convenience method for testing)

        Args:
            config: Round configuration
            csa_deltas: Pre-loaded CSA deltas

        Returns:
            Merged CSA
        """
        # Start round
        await self.start_round(config)

        # Simulate submissions
        for i, csa_delta in enumerate(csa_deltas):
            site_id = f"site_{i}"
            if site_id not in self.participants:
                await self.register_participant(site_id, {})
            await self.submit_update(config.round_id, site_id, csa_delta)

        # Aggregate
        merged_csa = await self.aggregate_round(config.round_id, config)

        # Complete round
        await self.complete_round()

        return merged_csa


async def main():
    """Example coordinator usage"""
    coordinator = SwarmCoordinator(
        registry_url="http://localhost:8080", workspace_dir=Path("./swarm_workspace")
    )

    config = SwarmRoundConfig(
        round_id="round_001",
        privacy_mode="ldp",
        aggregation_strategy="trimmed_mean",
        epsilon=1.0,
        delta=1e-5,
    )

    # TODO: Load CSA deltas from sites
    # merged_csa = await coordinator.run_round(config, csa_deltas)

    logger.info("✓ Swarm round complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
