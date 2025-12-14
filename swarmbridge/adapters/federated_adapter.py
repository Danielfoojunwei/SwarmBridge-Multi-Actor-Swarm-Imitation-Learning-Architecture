"""
Federated Learning Adapter

Replaces direct OpenFL dependencies with adapter pattern.
Interacts with external federated learning service.

This allows SwarmBridge to:
- Submit local updates
- Request merged models
- Support unlearning
- Track federated rounds

WITHOUT depending directly on OpenFL, Flower, or other FL frameworks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import httpx


@dataclass
class FederatedUpdate:
    """Local update to submit to federated service"""
    csa_id: str
    skill_name: str
    site_id: str
    model_weights: bytes  # Encrypted or plaintext
    privacy_mode: str  # "encrypted", "differential_privacy", "plaintext"
    epsilon: Optional[float] = None
    delta: Optional[float] = None


@dataclass
class FederatedRound:
    """Federated learning round metadata"""
    round_id: int
    skill_name: str
    num_participants: int
    status: str  # "pending", "aggregating", "completed"
    aggregated_csa_id: Optional[str] = None


class FederatedLearningAdapter:
    """
    Adapter for federated learning service.

    Replaces direct OpenFL usage in SwarmBridge core.

    Example:
        adapter = FederatedLearningAdapter(
            service_url="http://localhost:8001"
        )

        # Submit local update
        await adapter.submit_local_update(
            csa_id="csa_123",
            skill_name="cooperative_assembly",
        )

        # Request merged model
        merged_csa_id = await adapter.request_merge(
            skill_name="cooperative_assembly",
        )

        # Unlearning
        await adapter.request_unlearning(
            csa_id="csa_123",
            skill_name="cooperative_assembly",
        )
    """

    def __init__(
        self,
        service_url: str,
        site_id: Optional[str] = None,
    ):
        """
        Initialize federated learning adapter.

        Args:
            service_url: URL of federated learning service
            site_id: Unique site identifier
        """
        self.service_url = service_url.rstrip("/")
        self.site_id = site_id or "swarmbridge_default"

    async def submit_local_update(
        self,
        csa_id: str,
        skill_name: str,
        privacy_mode: str = "encrypted",
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ):
        """
        Submit local CSA update to federated learning service.

        The service handles:
        - Aggregation strategy
        - Privacy mechanisms
        - Round management

        SwarmBridge only provides the local update.
        """

        print(f"  Submitting local update to federated service...")
        print(f"    CSA ID: {csa_id}")
        print(f"    Site ID: {self.site_id}")
        print(f"    Privacy mode: {privacy_mode}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.service_url}/api/v1/federated/submit_update",
                json={
                    "csa_id": csa_id,
                    "skill_name": skill_name,
                    "site_id": self.site_id,
                    "privacy_mode": privacy_mode,
                    "epsilon": epsilon,
                    "delta": delta,
                },
            )
            response.raise_for_status()
            result = response.json()

            print(f"    Round ID: {result.get('round_id')}")

            return result

    async def request_merge(
        self,
        skill_name: str,
        min_participants: int = 2,
    ) -> str:
        """
        Request federated merge of skill updates.

        Returns:
            Merged CSA ID
        """

        print(f"  Requesting federated merge...")
        print(f"    Skill: {skill_name}")
        print(f"    Min participants: {min_participants}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.service_url}/api/v1/federated/request_merge",
                json={
                    "skill_name": skill_name,
                    "min_participants": min_participants,
                },
            )
            response.raise_for_status()
            result = response.json()

            merged_csa_id = result["merged_csa_id"]
            print(f"    Merged CSA: {merged_csa_id}")

            return merged_csa_id

    async def request_unlearning(
        self,
        csa_id: str,
        skill_name: str,
        unlearning_method: str = "influence_removal",
    ):
        """
        Request unlearning of a specific contribution.

        Supported methods:
        - "influence_removal": Remove influence without retraining
        - "retraining": Full retraining without this contribution

        The federated service handles the unlearning logic.
        """

        print(f"  Requesting unlearning...")
        print(f"    CSA ID: {csa_id}")
        print(f"    Method: {unlearning_method}")

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{self.service_url}/api/v1/federated/request_unlearning",
                json={
                    "csa_id": csa_id,
                    "skill_name": skill_name,
                    "site_id": self.site_id,
                    "method": unlearning_method,
                },
            )
            response.raise_for_status()
            result = response.json()

            print(f"    Unlearning request accepted")
            print(f"    New CSA ID: {result.get('new_csa_id')}")

            return result

    async def get_round_status(self, round_id: int) -> FederatedRound:
        """Get status of a federated learning round"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.service_url}/api/v1/federated/round/{round_id}"
            )
            response.raise_for_status()
            data = response.json()

            return FederatedRound(
                round_id=data["round_id"],
                skill_name=data["skill_name"],
                num_participants=data["num_participants"],
                status=data["status"],
                aggregated_csa_id=data.get("aggregated_csa_id"),
            )

    async def list_available_rounds(self, skill_name: Optional[str] = None) -> List[FederatedRound]:
        """List available federated rounds"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {}
            if skill_name:
                params["skill_name"] = skill_name

            response = await client.get(
                f"{self.service_url}/api/v1/federated/rounds",
                params=params,
            )
            response.raise_for_status()
            rounds_data = response.json()

            return [
                FederatedRound(
                    round_id=r["round_id"],
                    skill_name=r["skill_name"],
                    num_participants=r["num_participants"],
                    status=r["status"],
                    aggregated_csa_id=r.get("aggregated_csa_id"),
                )
                for r in rounds_data
            ]
