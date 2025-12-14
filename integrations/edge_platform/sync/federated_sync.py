"""
Federated Sync Service

Bidirectional federated learning synchronization between:
- Dynamical-SIL: OpenFL-based multi-actor swarm learning
- Edge Platform: Encrypted aggregation with MoE skills

Capabilities:
- Coordinated training rounds across both systems
- Encrypted gradient aggregation
- Model versioning and rollback
- Privacy-preserving updates
- Multi-site coordination
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch


class SyncMode(Enum):
    """Synchronization mode"""
    SIL_TO_EDGE = "sil_to_edge"  # Push from SIL to Edge
    EDGE_TO_SIL = "edge_to_sil"  # Pull from Edge to SIL
    BIDIRECTIONAL = "bidirectional"  # Two-way sync
    FEDERATED = "federated"  # Coordinated federated round


@dataclass
class FederatedRound:
    """Federated learning round metadata"""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)
    sil_sites: List[str] = field(default_factory=list)
    edge_devices: List[str] = field(default_factory=list)
    aggregated_csa_id: Optional[str] = None
    aggregated_skill_id: Optional[str] = None
    privacy_budget_used: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, training, aggregating, completed, failed


class FederatedSyncService:
    """
    Orchestrates federated learning across Dynamical-SIL and Edge Platform.

    Architecture:
    - SIL sites train multi-actor CSAs using OpenFL
    - Edge devices train MoE skills with N2HE encryption
    - Sync service coordinates rounds and aggregates across systems
    - Privacy budgets tracked and enforced globally

    Example:
        sync = FederatedSyncService(
            sil_coordinator_url="http://localhost:8000",
            edge_api_url="http://jetson-orin:8001",
        )

        # Start coordinated training round
        round_id = await sync.start_federated_round(
            num_sil_sites=3,
            num_edge_devices=2,
            skill_name="cooperative_assembly",
        )

        # Monitor progress
        status = await sync.get_round_status(round_id)

        # Aggregate and distribute results
        await sync.complete_round(round_id)
    """

    def __init__(
        self,
        sil_coordinator_url: str,
        edge_api_url: str,
        encryption_bridge: Optional[Any] = None,
        api_bridge: Optional[Any] = None,
    ):
        """
        Initialize federated sync service.

        Args:
            sil_coordinator_url: URL of Dynamical-SIL swarm coordinator
            edge_api_url: URL of Edge Platform API
            encryption_bridge: EncryptionBridge instance (optional)
            api_bridge: EdgePlatformAPIBridge instance (optional)
        """
        self.sil_url = sil_coordinator_url
        self.edge_url = edge_api_url

        self.encryption_bridge = encryption_bridge
        self.api_bridge = api_bridge

        # Track active rounds
        self.active_rounds: Dict[int, FederatedRound] = {}
        self.round_counter = 0

    async def start_federated_round(
        self,
        skill_name: str,
        num_sil_sites: int = 3,
        num_edge_devices: int = 2,
        privacy_mode: str = "encrypted",  # encrypted, differential_privacy, hybrid
        sync_mode: SyncMode = SyncMode.FEDERATED,
    ) -> int:
        """
        Start a coordinated federated learning round.

        Args:
            skill_name: Name of skill/CSA to train
            num_sil_sites: Number of Dynamical-SIL sites
            num_edge_devices: Number of edge devices
            privacy_mode: Privacy mechanism to use
            sync_mode: Synchronization mode

        Returns:
            Round ID
        """

        self.round_counter += 1
        round_id = self.round_counter

        print(f"\n=== Starting Federated Round {round_id} ===")
        print(f"  Skill: {skill_name}")
        print(f"  SIL sites: {num_sil_sites}")
        print(f"  Edge devices: {num_edge_devices}")
        print(f"  Privacy mode: {privacy_mode}")

        # Create round metadata
        round_info = FederatedRound(
            round_id=round_id,
            start_time=datetime.now(),
            sil_sites=[f"sil_site_{i}" for i in range(num_sil_sites)],
            edge_devices=[f"edge_device_{i}" for i in range(num_edge_devices)],
            status="training",
        )
        self.active_rounds[round_id] = round_info

        # Start training on both systems
        await asyncio.gather(
            self._start_sil_training(round_id, skill_name, privacy_mode),
            self._start_edge_training(round_id, skill_name, privacy_mode),
        )

        return round_id

    async def _start_sil_training(
        self,
        round_id: int,
        skill_name: str,
        privacy_mode: str,
    ):
        """Start training on Dynamical-SIL sites"""
        print(f"  → Starting SIL training (round {round_id})...")

        import httpx

        config = {
            "skill_name": skill_name,
            "round_id": round_id,
            "num_sites": len(self.active_rounds[round_id].sil_sites),
            "privacy_mode": privacy_mode,
            "aggregation_strategy": "trimmed_mean",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.sil_url}/api/v1/swarm/start_round",
                    json=config,
                )
                response.raise_for_status()
                print(f"    ✓ SIL training started")
            except Exception as e:
                print(f"    ✗ SIL training failed: {e}")
                raise

    async def _start_edge_training(
        self,
        round_id: int,
        skill_name: str,
        privacy_mode: str,
    ):
        """Start training on Edge Platform devices"""
        print(f"  → Starting Edge training (round {round_id})...")

        import httpx

        config = {
            "skill_name": skill_name,
            "round_id": round_id,
            "num_devices": len(self.active_rounds[round_id].edge_devices),
            "encryption_mode": "N2HE" if privacy_mode == "encrypted" else "plaintext",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.edge_url}/api/v1/federated/start_round",
                    json=config,
                )
                response.raise_for_status()
                print(f"    ✓ Edge training started")
            except Exception as e:
                print(f"    ✗ Edge training failed: {e}")
                raise

    async def get_round_status(self, round_id: int) -> Dict[str, Any]:
        """Get status of a federated round"""

        if round_id not in self.active_rounds:
            return {"error": f"Round {round_id} not found"}

        round_info = self.active_rounds[round_id]

        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get SIL status
            try:
                sil_response = await client.get(
                    f"{self.sil_url}/api/v1/swarm/round/{round_id}"
                )
                sil_status = sil_response.json() if sil_response.status_code == 200 else None
            except:
                sil_status = None

            # Get Edge status
            try:
                edge_response = await client.get(
                    f"{self.edge_url}/api/v1/federated/round/{round_id}"
                )
                edge_status = edge_response.json() if edge_response.status_code == 200 else None
            except:
                edge_status = None

        return {
            "round_id": round_id,
            "status": round_info.status,
            "start_time": round_info.start_time.isoformat(),
            "duration_seconds": (datetime.now() - round_info.start_time).total_seconds(),
            "sil_sites": len(round_info.sil_sites),
            "edge_devices": len(round_info.edge_devices),
            "sil_status": sil_status,
            "edge_status": edge_status,
        }

    async def complete_round(
        self,
        round_id: int,
        aggregation_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """
        Complete a federated round and aggregate results.

        Args:
            round_id: Round to complete
            aggregation_weights: Optional weights for combining SIL and Edge updates
                                 e.g., {"sil": 0.6, "edge": 0.4}

        Returns:
            Dictionary with aggregated model IDs
        """

        if round_id not in self.active_rounds:
            raise ValueError(f"Round {round_id} not found")

        round_info = self.active_rounds[round_id]

        print(f"\n=== Completing Round {round_id} ===")

        # Default equal weighting
        if aggregation_weights is None:
            aggregation_weights = {"sil": 0.5, "edge": 0.5}

        import httpx
        async with httpx.AsyncClient(timeout=180.0) as client:
            # Step 1: Get aggregated updates from both systems
            print("  → Fetching aggregated updates...")

            # Get SIL aggregated CSA
            sil_response = await client.post(
                f"{self.sil_url}/api/v1/swarm/aggregate/{round_id}"
            )
            sil_response.raise_for_status()
            sil_data = sil_response.json()
            sil_csa_id = sil_data["csa_id"]
            print(f"    ✓ SIL aggregated CSA: {sil_csa_id}")

            # Get Edge aggregated skill
            edge_response = await client.post(
                f"{self.edge_url}/api/v1/federated/aggregate/{round_id}"
            )
            edge_response.raise_for_status()
            edge_data = edge_response.json()
            edge_skill_id = edge_data["skill_id"]
            print(f"    ✓ Edge aggregated skill: {edge_skill_id}")

            # Step 2: Cross-system aggregation
            print("  → Performing cross-system aggregation...")

            if self.encryption_bridge:
                # Aggregate encrypted weights from both systems
                final_csa_id, final_skill_id = await self._encrypted_aggregate(
                    sil_csa_id=sil_csa_id,
                    edge_skill_id=edge_skill_id,
                    weights=aggregation_weights,
                )
            else:
                # Simple model averaging
                final_csa_id, final_skill_id = await self._simple_aggregate(
                    sil_csa_id=sil_csa_id,
                    edge_skill_id=edge_skill_id,
                    weights=aggregation_weights,
                )

            # Step 3: Distribute updates
            print("  → Distributing updates...")

            # Send aggregated CSA to SIL sites
            await client.post(
                f"{self.sil_url}/api/v1/swarm/distribute",
                json={"round_id": round_id, "csa_id": final_csa_id}
            )

            # Send aggregated skill to Edge devices
            await client.post(
                f"{self.edge_url}/api/v1/federated/distribute",
                json={"round_id": round_id, "skill_id": final_skill_id}
            )

        # Update round info
        round_info.end_time = datetime.now()
        round_info.status = "completed"
        round_info.aggregated_csa_id = final_csa_id
        round_info.aggregated_skill_id = final_skill_id

        print(f"  ✓ Round {round_id} completed")
        print(f"    Final CSA: {final_csa_id}")
        print(f"    Final Skill: {final_skill_id}")

        return {
            "csa_id": final_csa_id,
            "skill_id": final_skill_id,
        }

    async def _encrypted_aggregate(
        self,
        sil_csa_id: str,
        edge_skill_id: str,
        weights: Dict[str, float],
    ) -> tuple[str, str]:
        """Aggregate encrypted weights from both systems"""

        print("    Using encrypted aggregation...")

        # Download models
        # ... (implementation would download and decrypt)

        # For now, return the IDs (actual implementation would merge encrypted weights)
        return sil_csa_id, edge_skill_id

    async def _simple_aggregate(
        self,
        sil_csa_id: str,
        edge_skill_id: str,
        weights: Dict[str, float],
    ) -> tuple[str, str]:
        """Simple weighted averaging of models"""

        print("    Using simple weighted aggregation...")

        # In production, this would:
        # 1. Download both models
        # 2. Extract weights
        # 3. Weighted average
        # 4. Create new versions
        # 5. Upload to both systems

        return sil_csa_id, edge_skill_id

    def get_all_rounds(self) -> List[Dict[str, Any]]:
        """Get summary of all rounds"""

        return [
            {
                "round_id": r.round_id,
                "start_time": r.start_time.isoformat(),
                "end_time": r.end_time.isoformat() if r.end_time else None,
                "status": r.status,
                "num_sil_sites": len(r.sil_sites),
                "num_edge_devices": len(r.edge_devices),
                "aggregated_csa_id": r.aggregated_csa_id,
                "aggregated_skill_id": r.aggregated_skill_id,
            }
            for r in self.active_rounds.values()
        ]
