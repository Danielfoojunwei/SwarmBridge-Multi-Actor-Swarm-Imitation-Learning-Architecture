"""
Registry Service Adapter

Adapter for CSA Registry REST API.

Provides:
- CSA upload/download
- Version management
- Deployment tracking
"""

import httpx
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import tarfile
import io

logger = logging.getLogger(__name__)


class RegistryAdapter:
    """
    Adapter for CSA Registry Service.

    Example:
        adapter = RegistryAdapter(
            registry_url="http://registry:8000"
        )

        # Upload CSA
        csa_id = await adapter.upload_csa(
            csa_path="./artifacts/handover_v1.tar.gz",
            skill_name="handover",
            version="1.0.0",
        )

        # Download CSA
        await adapter.download_csa(
            csa_id="handover_v1",
            output_path="./downloaded/handover_v1.tar.gz",
        )
    """

    def __init__(
        self,
        registry_url: str,
        timeout_s: float = 60.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize registry adapter.

        Args:
            registry_url: Base URL of registry service
            timeout_s: Request timeout
            api_key: Optional API key for authentication
        """
        self.registry_url = registry_url.rstrip("/")
        self.timeout_s = timeout_s
        self.headers = {}

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def upload_csa(
        self,
        csa_path: str,
        skill_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload CSA to registry.

        Args:
            csa_path: Path to CSA tarball
            skill_name: Skill name
            version: Skill version
            metadata: Additional metadata

        Returns:
            CSA ID
        """

        if not Path(csa_path).exists():
            raise FileNotFoundError(f"CSA not found: {csa_path}")

        files = {"file": open(csa_path, "rb")}
        data = {
            "skill_name": skill_name,
            "version": version,
        }

        if metadata:
            data["metadata"] = str(metadata)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(
                    f"{self.registry_url}/api/v1/csa/upload",
                    files=files,
                    data=data,
                    headers=self.headers,
                )

                response.raise_for_status()
                result = response.json()

                csa_id = result.get("csa_id")
                logger.info(f"✅ Uploaded CSA: {csa_id}")

                return csa_id

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to upload CSA: {e}")
            raise
        finally:
            files["file"].close()

    async def download_csa(
        self,
        csa_id: str,
        output_path: str,
    ) -> str:
        """
        Download CSA from registry.

        Args:
            csa_id: CSA identifier
            output_path: Output file path

        Returns:
            Path to downloaded CSA
        """

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(
                    f"{self.registry_url}/api/v1/csa/{csa_id}/download",
                    headers=self.headers,
                )

                response.raise_for_status()

                # Write to file
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"✅ Downloaded CSA to: {output_path}")

                return output_path

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to download CSA: {e}")
            raise

    async def get_csa_metadata(self, csa_id: str) -> Dict[str, Any]:
        """
        Get CSA metadata.

        Args:
            csa_id: CSA identifier

        Returns:
            CSA metadata
        """

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(
                    f"{self.registry_url}/api/v1/csa/{csa_id}",
                    headers=self.headers,
                )

                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to get CSA metadata: {e}")
            raise

    async def list_csas(
        self,
        skill_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List CSAs in registry.

        Args:
            skill_name: Optional filter by skill name
            limit: Maximum number of results

        Returns:
            List of CSA metadata
        """

        params = {"limit": limit}
        if skill_name:
            params["skill_name"] = skill_name

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(
                    f"{self.registry_url}/api/v1/csa/list",
                    params=params,
                    headers=self.headers,
                )

                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to list CSAs: {e}")
            raise

    async def deploy_csa(
        self,
        csa_id: str,
        deployment_target: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Deploy CSA to target environment.

        Args:
            csa_id: CSA identifier
            deployment_target: Target (edge_platform, swarmbrain, etc.)
            config: Deployment configuration

        Returns:
            Deployment response
        """

        payload = {
            "csa_id": csa_id,
            "deployment_target": deployment_target,
            "config": config or {},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(
                    f"{self.registry_url}/api/v1/csa/{csa_id}/deploy",
                    json=payload,
                    headers=self.headers,
                )

                response.raise_for_status()
                result = response.json()

                logger.info(
                    f"✅ Deployed CSA {csa_id} to {deployment_target}: "
                    f"deployment_id={result.get('deployment_id')}"
                )

                return result

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to deploy CSA: {e}")
            raise

    async def rollback_deployment(
        self,
        deployment_id: str,
    ) -> Dict[str, Any]:
        """
        Rollback deployment.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Rollback response
        """

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(
                    f"{self.registry_url}/api/v1/deployment/{deployment_id}/rollback",
                    headers=self.headers,
                )

                response.raise_for_status()
                result = response.json()

                logger.info(f"✅ Rolled back deployment: {deployment_id}")

                return result

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to rollback deployment: {e}")
            raise


class MockRegistryAdapter(RegistryAdapter):
    """
    Mock adapter for local development and testing.

    Simulates registry without external service.
    """

    def __init__(self):
        # Don't call super().__init__()
        self.csas = {}
        self.deployments = {}

    async def upload_csa(
        self,
        csa_path: str,
        skill_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:

        if not Path(csa_path).exists():
            raise FileNotFoundError(f"CSA not found: {csa_path}")

        csa_id = f"{skill_name}_v{version}"

        self.csas[csa_id] = {
            "csa_id": csa_id,
            "skill_name": skill_name,
            "version": version,
            "path": csa_path,
            "metadata": metadata or {},
        }

        logger.info(f"🔄 [MOCK] Uploaded CSA: {csa_id}")

        return csa_id

    async def download_csa(
        self,
        csa_id: str,
        output_path: str,
    ) -> str:

        if csa_id not in self.csas:
            raise ValueError(f"CSA not found: {csa_id}")

        # Copy file
        import shutil
        source_path = self.csas[csa_id]["path"]
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, output_path)

        logger.info(f"🔄 [MOCK] Downloaded CSA to: {output_path}")

        return output_path

    async def get_csa_metadata(self, csa_id: str) -> Dict[str, Any]:

        if csa_id not in self.csas:
            raise ValueError(f"CSA not found: {csa_id}")

        return self.csas[csa_id]

    async def list_csas(
        self,
        skill_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:

        csas = list(self.csas.values())

        if skill_name:
            csas = [c for c in csas if c["skill_name"] == skill_name]

        return csas[:limit]

    async def deploy_csa(
        self,
        csa_id: str,
        deployment_target: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        deployment_id = f"deploy_{len(self.deployments)}"

        self.deployments[deployment_id] = {
            "deployment_id": deployment_id,
            "csa_id": csa_id,
            "deployment_target": deployment_target,
            "config": config,
            "status": "deployed",
        }

        logger.info(f"🔄 [MOCK] Deployed CSA: {deployment_id}")

        return self.deployments[deployment_id]
