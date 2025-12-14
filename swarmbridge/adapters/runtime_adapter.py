"""
Edge Platform Runtime Adapter

Thin wrapper for executing skills via Edge Platform Dynamical API.

SwarmBridge delegates runtime execution to Edge Platform:
- Fetch CSA from registry
- Execute via Edge Platform
- Monitor execution status
"""

import httpx
import asyncio
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class EdgePlatformRuntimeAdapter:
    """
    Adapter for Edge Platform skill execution.

    Example:
        adapter = EdgePlatformRuntimeAdapter(
            edge_api_url="http://edge-platform:8080",
            registry_adapter=registry_adapter,
        )

        # Execute skill
        execution_id = await adapter.execute_skill(
            csa_id="handover_v1",
            robot_id="robot_1",
            task_parameters={"target_object": "cube"},
        )

        # Monitor execution
        status = await adapter.get_execution_status(execution_id)
    """

    def __init__(
        self,
        edge_api_url: str,
        registry_adapter,  # RegistryAdapter instance
        timeout_s: float = 120.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize runtime adapter.

        Args:
            edge_api_url: Edge Platform Dynamical API URL
            registry_adapter: Registry adapter for CSA fetching
            timeout_s: Execution timeout
            api_key: Optional API key
        """
        self.edge_api_url = edge_api_url.rstrip("/")
        self.registry_adapter = registry_adapter
        self.timeout_s = timeout_s
        self.headers = {}

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def execute_skill(
        self,
        csa_id: str,
        robot_id: str,
        task_parameters: Dict[str, Any],
        execution_mode: str = "real",  # or "sim"
    ) -> str:
        """
        Execute skill via Edge Platform.

        Workflow:
        1. Fetch CSA from registry
        2. Convert to Edge Platform MoE format
        3. Submit to Dynamical API
        4. Monitor execution

        Args:
            csa_id: CSA identifier
            robot_id: Robot to execute on
            task_parameters: Execution parameters
            execution_mode: "real" or "sim"

        Returns:
            Execution ID
        """

        # 1. Get CSA metadata from registry
        csa_metadata = await self.registry_adapter.get_csa_metadata(csa_id)

        # 2. Submit execution request to Edge Platform
        payload = {
            "skill_id": csa_id,
            "robot_id": robot_id,
            "task_parameters": task_parameters,
            "execution_mode": execution_mode,
            "skill_metadata": {
                "skill_name": csa_metadata.get("skill_name"),
                "version": csa_metadata.get("version"),
                "num_actors": csa_metadata.get("metadata", {}).get("num_actors", 1),
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(
                    f"{self.edge_api_url}/api/v1/skills/execute",
                    json=payload,
                    headers=self.headers,
                )

                response.raise_for_status()
                result = response.json()

                execution_id = result.get("execution_id")
                logger.info(
                    f"✅ Started skill execution: {execution_id} "
                    f"(skill={csa_id}, robot={robot_id})"
                )

                return execution_id

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to execute skill: {e}")
            raise

    async def get_execution_status(
        self,
        execution_id: str,
    ) -> Dict[str, Any]:
        """
        Get execution status.

        Args:
            execution_id: Execution identifier

        Returns:
            Execution status (pending, running, completed, failed)
        """

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.edge_api_url}/api/v1/executions/{execution_id}",
                    headers=self.headers,
                )

                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to get execution status: {e}")
            raise

    async def wait_for_completion(
        self,
        execution_id: str,
        poll_interval_s: float = 2.0,
        max_wait_s: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Wait for execution to complete.

        Args:
            execution_id: Execution identifier
            poll_interval_s: Polling interval
            max_wait_s: Maximum wait time

        Returns:
            Final execution status

        Raises:
            TimeoutError: If execution doesn't complete in time
        """

        start_time = asyncio.get_event_loop().time()

        while True:
            status = await self.get_execution_status(execution_id)

            if status["status"] in ["completed", "failed", "aborted"]:
                logger.info(
                    f"✅ Execution {execution_id} finished: {status['status']}"
                )
                return status

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait_s:
                raise TimeoutError(
                    f"Execution {execution_id} did not complete in {max_wait_s}s"
                )

            await asyncio.sleep(poll_interval_s)

    async def stop_execution(
        self,
        execution_id: str,
    ) -> Dict[str, Any]:
        """
        Stop running execution.

        Args:
            execution_id: Execution identifier

        Returns:
            Stop response
        """

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.edge_api_url}/api/v1/executions/{execution_id}/stop",
                    headers=self.headers,
                )

                response.raise_for_status()
                result = response.json()

                logger.info(f"✅ Stopped execution: {execution_id}")

                return result

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to stop execution: {e}")
            raise

    async def get_execution_logs(
        self,
        execution_id: str,
    ) -> List[str]:
        """
        Get execution logs.

        Args:
            execution_id: Execution identifier

        Returns:
            List of log lines
        """

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.edge_api_url}/api/v1/executions/{execution_id}/logs",
                    headers=self.headers,
                )

                response.raise_for_status()
                result = response.json()

                return result.get("logs", [])

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to get execution logs: {e}")
            raise


class MockEdgePlatformRuntimeAdapter(EdgePlatformRuntimeAdapter):
    """
    Mock adapter for local development and testing.

    Simulates Edge Platform execution without external service.
    """

    def __init__(self, registry_adapter):
        # Don't call super().__init__()
        self.registry_adapter = registry_adapter
        self.executions = {}

    async def execute_skill(
        self,
        csa_id: str,
        robot_id: str,
        task_parameters: Dict[str, Any],
        execution_mode: str = "real",
    ) -> str:

        execution_id = f"exec_{len(self.executions)}"

        # Get CSA metadata (will use mock registry)
        try:
            csa_metadata = await self.registry_adapter.get_csa_metadata(csa_id)
        except Exception:
            csa_metadata = {"skill_name": csa_id.split("_")[0]}

        self.executions[execution_id] = {
            "execution_id": execution_id,
            "csa_id": csa_id,
            "robot_id": robot_id,
            "task_parameters": task_parameters,
            "execution_mode": execution_mode,
            "status": "running",
            "logs": [
                f"[MOCK] Starting execution on {robot_id}",
                f"[MOCK] Loading skill: {csa_id}",
                f"[MOCK] Executing with params: {task_parameters}",
            ],
        }

        logger.info(f"🔄 [MOCK] Started execution: {execution_id}")

        # Simulate async completion
        asyncio.create_task(self._simulate_completion(execution_id))

        return execution_id

    async def _simulate_completion(self, execution_id: str):
        """Simulate execution completion after delay"""
        await asyncio.sleep(3)

        if execution_id in self.executions:
            self.executions[execution_id]["status"] = "completed"
            self.executions[execution_id]["logs"].append(
                "[MOCK] Execution completed successfully"
            )

    async def get_execution_status(
        self,
        execution_id: str,
    ) -> Dict[str, Any]:

        if execution_id not in self.executions:
            raise ValueError(f"Execution not found: {execution_id}")

        return self.executions[execution_id]

    async def wait_for_completion(
        self,
        execution_id: str,
        poll_interval_s: float = 0.5,
        max_wait_s: float = 10.0,
    ) -> Dict[str, Any]:

        start_time = asyncio.get_event_loop().time()

        while True:
            status = await self.get_execution_status(execution_id)

            if status["status"] in ["completed", "failed", "aborted"]:
                logger.info(f"🔄 [MOCK] Execution finished: {status['status']}")
                return status

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait_s:
                raise TimeoutError(
                    f"Execution {execution_id} did not complete in {max_wait_s}s"
                )

            await asyncio.sleep(poll_interval_s)

    async def stop_execution(
        self,
        execution_id: str,
    ) -> Dict[str, Any]:

        if execution_id in self.executions:
            self.executions[execution_id]["status"] = "aborted"
            logger.info(f"🔄 [MOCK] Stopped execution: {execution_id}")

        return self.executions.get(execution_id, {})

    async def get_execution_logs(
        self,
        execution_id: str,
    ) -> List[str]:

        if execution_id not in self.executions:
            raise ValueError(f"Execution not found: {execution_id}")

        return self.executions[execution_id].get("logs", [])
