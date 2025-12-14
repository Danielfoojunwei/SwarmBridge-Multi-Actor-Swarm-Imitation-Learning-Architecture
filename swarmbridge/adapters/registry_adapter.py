"""
Registry Adapter

Thin adapter for CSA registry operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import httpx


class RegistryAdapter:
    """
    Adapter for CSA registry service.

    Handles:
    - CSA upload
    - CSA download
    - CSA listing
    - Metadata queries

    Example:
        adapter = RegistryAdapter(registry_url="http://localhost:8000")

        # Upload CSA
        csa_id = await adapter.upload_csa(
            csa_path=Path("artifacts/skill_v1.0.tar.gz"),
            skill_name="cooperative_assembly",
        )

        # Download CSA
        csa_path = await adapter.download_csa(csa_id="csa_123")
    """

    def __init__(self, registry_url: str):
        self.registry_url = registry_url.rstrip("/")

    async def upload_csa(
        self,
        csa_path: Path,
        skill_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload CSA to registry.

        Returns:
            CSA ID
        """

        print(f"  Uploading CSA to registry...")
        print(f"    File: {csa_path}")
        print(f"    Size: {csa_path.stat().st_size / 1024:.1f} KB")

        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(csa_path, "rb") as f:
                files = {"file": (csa_path.name, f, "application/gzip")}
                data = {
                    "skill_name": skill_name,
                    "source": "swarmbridge",
                }

                if metadata:
                    data["metadata"] = str(metadata)

                response = await client.post(
                    f"{self.registry_url}/api/v1/csa/upload",
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                result = response.json()

                csa_id = result["csa_id"]
                print(f"    Uploaded with ID: {csa_id}")

                return csa_id

    async def download_csa(
        self,
        csa_id: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Download CSA from registry.

        Returns:
            Path to downloaded CSA
        """

        print(f"  Downloading CSA: {csa_id}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(
                f"{self.registry_url}/api/v1/csa/{csa_id}/download"
            )
            response.raise_for_status()

            # Save to file
            if output_path is None:
                output_path = Path(f"downloads/{csa_id}.tar.gz")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"    Downloaded to: {output_path}")

            return output_path

    async def get_csa_metadata(self, csa_id: str) -> Dict[str, Any]:
        """Get CSA metadata"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.registry_url}/api/v1/csa/{csa_id}"
            )
            response.raise_for_status()
            return response.json()

    async def list_csas(
        self,
        skill_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List CSAs in registry"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {"limit": limit}
            if skill_name:
                params["skill_name"] = skill_name

            response = await client.get(
                f"{self.registry_url}/api/v1/csa/list",
                params=params,
            )
            response.raise_for_status()
            return response.json()
