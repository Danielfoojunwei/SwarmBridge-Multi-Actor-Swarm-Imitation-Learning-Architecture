"""
Federated Learning Service Adapter

Thin adapter for external federated learning service.
Replaces direct OpenFL dependencies in SwarmBridge.

SwarmBridge delegates federated learning to external service:
- Submit local CSA updates
- Request federated merge
- Support unlearning requests
"""

import httpx
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FederatedLearningAdapter:
    """
    Adapter for external federated learning service.

    Supports multiple FL backends:
    - OpenFL
    - Flower
    - FATE

    Example:
        adapter = FederatedLearningAdapter(
            service_url="http://federated-service:8001"
        )

        # Submit local CSA update
        await adapter.submit_local_update(
            csa_id="handover_v1_site1",
            skill_name="handover",
            weights_path="./models/handover.pt",
        )

        # Request federated merge
        global_csa_id = await adapter.request_merge(
            skill_name="handover",
            merge_strategy="fedavg",
        )
    """

    def __init__(
        self,
        service_url: str,
        timeout_s: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize federated learning adapter.

        Args:
            service_url: Base URL of federated learning service
            timeout_s: Request timeout
            api_key: Optional API key for authentication
        """
        self.service_url = service_url.rstrip("/")
        self.timeout_s = timeout_s
        self.headers = {}

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def submit_local_update(
        self,
        csa_id: str,
        skill_name: str,
        weights_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit local CSA update to federated learning service.

        Args:
            csa_id: Local CSA identifier
            skill_name: Skill being trained
            weights_path: Path to model weights (optional)
            metadata: Additional metadata (site_id, training_metrics, etc.)

        Returns:
            Submission response with update_id
        """

        payload = {
            "csa_id": csa_id,
            "skill_name": skill_name,
            "metadata": metadata or {},
        }

        files = {}
        if weights_path and Path(weights_path).exists():
            files["weights"] = open(weights_path, "rb")

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                if files:
                    response = await client.post(
                        f"{self.service_url}/api/v1/federated/submit",
                        data=payload,
                        files=files,
                        headers=self.headers,
                    )
                else:
                    response = await client.post(
                        f"{self.service_url}/api/v1/federated/submit",
                        json=payload,
                        headers=self.headers,
                    )

                response.raise_for_status()
                result = response.json()

                logger.info(
                    f"✅ Submitted local update for {skill_name}: "
                    f"update_id={result.get('update_id')}"
                )

                return result

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to submit local update: {e}")
            raise
        finally:
            for f in files.values():
                f.close()

    async def request_merge(
        self,
        skill_name: str,
        merge_strategy: str = "fedavg",
        min_updates: int = 2,
        privacy_budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Request federated merge of skill updates.

        Args:
            skill_name: Skill to merge
            merge_strategy: Aggregation strategy (fedavg, trimmed_mean, krum)
            min_updates: Minimum number of updates required
            privacy_budget: Optional DP privacy budget

        Returns:
            Merge response with global_csa_id
        """

        payload = {
            "skill_name": skill_name,
            "merge_strategy": merge_strategy,
            "min_updates": min_updates,
        }

        if privacy_budget:
            payload["privacy_budget"] = privacy_budget

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s * 3) as client:
                response = await client.post(
                    f"{self.service_url}/api/v1/federated/merge",
                    json=payload,
                    headers=self.headers,
                )

                response.raise_for_status()
                result = response.json()

                logger.info(
                    f"✅ Federated merge completed for {skill_name}: "
                    f"global_csa_id={result.get('global_csa_id')}"
                )

                return result

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to request merge: {e}")
            raise

    async def request_unlearning(
        self,
        csa_id: str,
        skill_name: str,
        reason: str = "data_deletion_request",
    ) -> Dict[str, Any]:
        """
        Request federated unlearning of specific CSA.

        Supports GDPR Article 17 (Right to Erasure).

        Args:
            csa_id: CSA to unlearn
            skill_name: Skill name
            reason: Reason for unlearning

        Returns:
            Unlearning response
        """

        payload = {
            "csa_id": csa_id,
            "skill_name": skill_name,
            "reason": reason,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s * 2) as client:
                response = await client.post(
                    f"{self.service_url}/api/v1/federated/unlearn",
                    json=payload,
                    headers=self.headers,
                )

                response.raise_for_status()
                result = response.json()

                logger.info(
                    f"✅ Unlearning request submitted for {csa_id}: "
                    f"status={result.get('status')}"
                )

                return result

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to request unlearning: {e}")
            raise

    async def get_merge_status(self, merge_id: str) -> Dict[str, Any]:
        """
        Get status of federated merge operation.

        Args:
            merge_id: Merge operation ID

        Returns:
            Merge status (pending, in_progress, completed, failed)
        """

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(
                    f"{self.service_url}/api/v1/federated/merge/{merge_id}",
                    headers=self.headers,
                )

                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to get merge status: {e}")
            raise

    async def list_contributions(
        self,
        skill_name: str,
    ) -> List[Dict[str, Any]]:
        """
        List all contributions for a skill.

        Args:
            skill_name: Skill name

        Returns:
            List of contributions with metadata
        """

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(
                    f"{self.service_url}/api/v1/federated/contributions",
                    params={"skill_name": skill_name},
                    headers=self.headers,
                )

                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to list contributions: {e}")
            raise


class MockFederatedLearningAdapter(FederatedLearningAdapter):
    """
    Mock adapter for local development and testing.

    Simulates federated learning without external service.
    """

    def __init__(self):
        # Don't call super().__init__()
        self.submissions = []
        self.merges = []

    async def submit_local_update(
        self,
        csa_id: str,
        skill_name: str,
        weights_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        update_id = f"update_{len(self.submissions)}"

        self.submissions.append({
            "update_id": update_id,
            "csa_id": csa_id,
            "skill_name": skill_name,
            "weights_path": weights_path,
            "metadata": metadata,
        })

        logger.info(f"🔄 [MOCK] Submitted local update: {update_id}")

        return {
            "update_id": update_id,
            "status": "accepted",
        }

    async def request_merge(
        self,
        skill_name: str,
        merge_strategy: str = "fedavg",
        min_updates: int = 2,
        privacy_budget: Optional[float] = None,
    ) -> Dict[str, Any]:

        merge_id = f"merge_{len(self.merges)}"
        global_csa_id = f"{skill_name}_global_v{len(self.merges)}"

        self.merges.append({
            "merge_id": merge_id,
            "skill_name": skill_name,
            "global_csa_id": global_csa_id,
            "merge_strategy": merge_strategy,
        })

        logger.info(f"🔄 [MOCK] Federated merge: {global_csa_id}")

        return {
            "merge_id": merge_id,
            "global_csa_id": global_csa_id,
            "status": "completed",
        }

    async def request_unlearning(
        self,
        csa_id: str,
        skill_name: str,
        reason: str = "data_deletion_request",
    ) -> Dict[str, Any]:

        logger.info(f"🔄 [MOCK] Unlearning request: {csa_id}")

        return {
            "status": "completed",
            "csa_id": csa_id,
        }
