"""
Edge Platform Runtime Adapter

Thin wrapper for runtime execution via Edge Platform's Dynamical API.

Replaces local execution logic in swarm_skill_runtime.
SwarmBridge now only captures & trains; Edge Platform handles execution.
"""

from typing import Dict, List, Optional, Any
import httpx


class EdgePlatformRuntimeAdapter:
    """
    Thin wrapper for Edge Platform runtime execution.

    Responsibilities:
    1. Fetch skill from registry
    2. Invoke Dynamical API for execution
    3. Monitor execution status

    Does NOT:
    - Execute skills locally
    - Manage MoveIt2 directly
    - Handle behavior trees locally

    All runtime logic delegated to Edge Platform.

    Example:
        adapter = EdgePlatformRuntimeAdapter(
            edge_api_url="http://jetson-orin.local:8001"
        )

        # Execute skill
        execution_id = await adapter.execute_skill(
            csa_id="csa_123",
            robot_id="robot_1",
            task_parameters={...},
        )

        # Monitor
        status = await adapter.get_execution_status(execution_id)
    """

    def __init__(
        self,
        edge_api_url: str,
        registry_url: Optional[str] = None,
    ):
        """
        Initialize runtime adapter.

        Args:
            edge_api_url: URL of Edge Platform Dynamical API
            registry_url: Optional registry URL (for fetching CSAs)
        """
        self.edge_api_url = edge_api_url.rstrip("/")
        self.registry_url = registry_url.rstrip("/") if registry_url else None

    async def execute_skill(
        self,
        csa_id: str,
        robot_id: str,
        task_parameters: Optional[Dict[str, Any]] = None,
        fetch_from_registry: bool = True,
    ) -> str:
        """
        Execute skill on Edge Platform.

        Args:
            csa_id: CSA ID to execute
            robot_id: Target robot ID
            task_parameters: Task-specific parameters
            fetch_from_registry: Fetch latest version from registry

        Returns:
            Execution ID for monitoring
        """

        print(f"  Executing skill via Edge Platform...")
        print(f"    CSA ID: {csa_id}")
        print(f"    Robot: {robot_id}")

        # Optionally fetch from registry first
        if fetch_from_registry and self.registry_url:
            await self._ensure_skill_available(csa_id)

        # Invoke Dynamical API
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.edge_api_url}/api/v1/robot/invoke_skill",
                json={
                    "csa_id": csa_id,
                    "robot_id": robot_id,
                    "task_parameters": task_parameters or {},
                },
            )
            response.raise_for_status()
            result = response.json()

            execution_id = result["execution_id"]
            print(f"    Execution started: {execution_id}")

            return execution_id

    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get execution status from Edge Platform.

        Returns:
            Status information
        """

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.edge_api_url}/api/v1/robot/execution/{execution_id}/status"
            )
            response.raise_for_status()
            return response.json()

    async def stop_execution(self, execution_id: str):
        """Stop a running execution"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.edge_api_url}/api/v1/robot/execution/{execution_id}/stop"
            )
            response.raise_for_status()

            print(f"  Execution stopped: {execution_id}")

    async def _ensure_skill_available(self, csa_id: str):
        """Ensure skill is available on Edge Platform"""

        # Check if skill is already on Edge Platform
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{self.edge_api_url}/api/v1/skills/{csa_id}"
                )
                if response.status_code == 200:
                    print(f"    Skill already available on Edge Platform")
                    return
            except:
                pass

        # Not available, push from registry
        print(f"    Pushing skill from registry to Edge Platform...")

        from integrations.edge_platform.bridges.api_bridge import EdgePlatformAPIBridge

        bridge = EdgePlatformAPIBridge(
            sil_registry_url=self.registry_url,
            edge_api_url=self.edge_api_url,
        )

        await bridge.push_csa_to_edge(csa_id=csa_id)

    async def list_available_skills(self) -> List[Dict[str, Any]]:
        """List skills available on Edge Platform"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.edge_api_url}/api/v1/skills"
            )
            response.raise_for_status()
            return response.json()

    async def get_robot_status(self, robot_id: str) -> Dict[str, Any]:
        """Get robot status from Edge Platform"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.edge_api_url}/api/v1/robots/{robot_id}/status"
            )
            response.raise_for_status()
            return response.json()
