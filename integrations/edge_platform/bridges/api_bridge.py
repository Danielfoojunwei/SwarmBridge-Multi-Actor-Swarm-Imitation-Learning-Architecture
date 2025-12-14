"""
API Bridge: Dynamical-SIL Registry ↔ Edge Platform Skills API

Provides bidirectional synchronization between:
- Dynamical-SIL CSA Registry (FastAPI, PostgreSQL)
- Edge Platform Skills API (FastAPI, skill library)

Capabilities:
- Automatic CSA→MoE conversion and upload
- Skill discovery and import from Edge Platform
- Deployment status synchronization
- Version management across both systems
"""

import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
from pydantic import BaseModel, Field


class CSARecord(BaseModel):
    """CSA record from Dynamical-SIL registry"""
    id: str
    skill_name: str
    version: str
    file_path: str
    signature_verified: bool
    uploaded_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EdgeSkillRecord(BaseModel):
    """Skill record from Edge Platform"""
    skill_id: str
    skill_name: str
    version: str
    num_experts: int
    expert_specializations: List[str]
    uploaded_at: str
    privacy_level: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SyncStatus(BaseModel):
    """Synchronization status between systems"""
    csa_id: Optional[str] = None
    edge_skill_id: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    sync_direction: str  # "sil_to_edge", "edge_to_sil", "bidirectional"
    status: str  # "synced", "pending", "failed"
    error_message: Optional[str] = None


class EdgePlatformAPIBridge:
    """
    Bridge between Dynamical-SIL Registry and Edge Platform Skills API.

    Example usage:
        bridge = EdgePlatformAPIBridge(
            sil_registry_url="http://localhost:8000",
            edge_api_url="http://jetson-orin.local:8001"
        )

        # Push CSA to Edge Platform
        await bridge.push_csa_to_edge(csa_id="csa_123")

        # Pull skill from Edge Platform
        await bridge.pull_skill_from_edge(skill_id="skill_456")

        # Bidirectional sync
        await bridge.sync_all()
    """

    def __init__(
        self,
        sil_registry_url: str,
        edge_api_url: str,
        auth_token: Optional[str] = None,
        auto_convert: bool = True,
    ):
        """
        Initialize API bridge.

        Args:
            sil_registry_url: URL of Dynamical-SIL registry (e.g., http://localhost:8000)
            edge_api_url: URL of Edge Platform API (e.g., http://jetson-orin:8001)
            auth_token: Optional authentication token
            auto_convert: Automatically convert CSA↔MoE formats
        """
        self.sil_url = sil_registry_url.rstrip("/")
        self.edge_url = edge_api_url.rstrip("/")
        self.auth_token = auth_token
        self.auto_convert = auto_convert

        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        # Track sync state
        self.sync_map: Dict[str, SyncStatus] = {}

    async def push_csa_to_edge(
        self,
        csa_id: str,
        force_update: bool = False,
    ) -> EdgeSkillRecord:
        """
        Push a CSA from Dynamical-SIL to Edge Platform.

        Steps:
        1. Download CSA from registry
        2. Convert CSA → MoE skill format
        3. Upload to Edge Platform skills API
        4. Update sync status

        Args:
            csa_id: CSA ID in Dynamical-SIL registry
            force_update: Force update even if already synced

        Returns:
            Edge Platform skill record
        """

        print(f"Pushing CSA {csa_id} to Edge Platform...")

        # Check if already synced
        if csa_id in self.sync_map and not force_update:
            sync_status = self.sync_map[csa_id]
            if sync_status.status == "synced":
                print(f"  Already synced (edge_skill_id: {sync_status.edge_skill_id})")
                return await self._get_edge_skill(sync_status.edge_skill_id)

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Step 1: Get CSA metadata
            response = await client.get(
                f"{self.sil_url}/api/v1/csa/{csa_id}",
                headers=self.headers
            )
            response.raise_for_status()
            csa_record = CSARecord(**response.json())

            # Step 2: Download CSA file
            response = await client.get(
                f"{self.sil_url}/api/v1/csa/{csa_id}/download",
                headers=self.headers
            )
            response.raise_for_status()
            csa_data = response.content

            # Save temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp.write(csa_data)
                csa_path = Path(tmp.name)

            try:
                # Step 3: Convert CSA → MoE (if auto_convert enabled)
                if self.auto_convert:
                    from ..adapters.csa_to_moe import CSAToMoEAdapter

                    adapter = CSAToMoEAdapter()
                    moe_path = csa_path.with_suffix(".pt")

                    metadata = adapter.convert_csa_to_moe(
                        csa_path=csa_path,
                        output_path=moe_path,
                    )

                    # Step 4: Upload to Edge Platform
                    with open(moe_path, "rb") as f:
                        files = {"file": (moe_path.name, f, "application/octet-stream")}
                        data = {
                            "skill_name": metadata.skill_name,
                            "version": metadata.version,
                            "source": "dynamical_sil",
                            "source_csa_id": csa_id,
                        }

                        response = await client.post(
                            f"{self.edge_url}/api/v1/skills/upload",
                            headers=self.headers,
                            files=files,
                            data=data,
                        )
                        response.raise_for_status()
                        edge_skill = EdgeSkillRecord(**response.json())

                    # Clean up
                    moe_path.unlink()

                # Step 5: Update sync status
                self.sync_map[csa_id] = SyncStatus(
                    csa_id=csa_id,
                    edge_skill_id=edge_skill.skill_id,
                    last_synced_at=datetime.now(),
                    sync_direction="sil_to_edge",
                    status="synced",
                )

                print(f"  ✓ Pushed to Edge Platform (skill_id: {edge_skill.skill_id})")
                return edge_skill

            finally:
                # Clean up
                csa_path.unlink()

    async def pull_skill_from_edge(
        self,
        skill_id: str,
        force_update: bool = False,
    ) -> CSARecord:
        """
        Pull a skill from Edge Platform into Dynamical-SIL.

        Steps:
        1. Download skill from Edge Platform
        2. Convert MoE skill → CSA format
        3. Upload to Dynamical-SIL registry
        4. Update sync status

        Args:
            skill_id: Skill ID on Edge Platform
            force_update: Force update even if already synced

        Returns:
            CSA record in Dynamical-SIL
        """

        print(f"Pulling skill {skill_id} from Edge Platform...")

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Step 1: Download skill from Edge Platform
            response = await client.get(
                f"{self.edge_url}/api/v1/skills/{skill_id}/download",
                headers=self.headers
            )
            response.raise_for_status()
            skill_data = response.content

            # Save temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp.write(skill_data)
                moe_path = Path(tmp.name)

            try:
                # Step 2: Convert MoE → CSA (if auto_convert enabled)
                if self.auto_convert:
                    from ..adapters.csa_to_moe import CSAToMoEAdapter

                    adapter = CSAToMoEAdapter()
                    csa_path = moe_path.with_suffix(".tar.gz")

                    adapter.convert_moe_to_csa(
                        moe_path=moe_path,
                        output_path=csa_path,
                    )

                    # Step 3: Upload to Dynamical-SIL registry
                    with open(csa_path, "rb") as f:
                        files = {"file": (csa_path.name, f, "application/gzip")}
                        data = {
                            "source": "edge_platform",
                            "source_skill_id": skill_id,
                        }

                        response = await client.post(
                            f"{self.sil_url}/api/v1/csa/upload",
                            headers=self.headers,
                            files=files,
                            data=data,
                        )
                        response.raise_for_status()
                        csa_record = CSARecord(**response.json())

                    # Clean up
                    csa_path.unlink()

                # Step 4: Update sync status
                self.sync_map[csa_record.id] = SyncStatus(
                    csa_id=csa_record.id,
                    edge_skill_id=skill_id,
                    last_synced_at=datetime.now(),
                    sync_direction="edge_to_sil",
                    status="synced",
                )

                print(f"  ✓ Pulled to Dynamical-SIL (csa_id: {csa_record.id})")
                return csa_record

            finally:
                # Clean up
                moe_path.unlink()

    async def sync_all(self) -> Dict[str, int]:
        """
        Perform bidirectional sync of all skills/CSAs.

        Returns:
            Statistics: {pushed: N, pulled: M, failed: K}
        """

        print("Starting bidirectional sync...")

        stats = {"pushed": 0, "pulled": 0, "failed": 0, "skipped": 0}

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get all CSAs from Dynamical-SIL
            response = await client.get(
                f"{self.sil_url}/api/v1/csa/list",
                headers=self.headers
            )
            response.raise_for_status()
            csas = [CSARecord(**item) for item in response.json()]

            # Get all skills from Edge Platform
            response = await client.get(
                f"{self.edge_url}/api/v1/skills",
                headers=self.headers
            )
            response.raise_for_status()
            edge_skills = [EdgeSkillRecord(**item) for item in response.json()]

        # Push new CSAs to Edge Platform
        for csa in csas:
            if csa.id not in self.sync_map:
                try:
                    await self.push_csa_to_edge(csa.id)
                    stats["pushed"] += 1
                except Exception as e:
                    print(f"  ✗ Failed to push {csa.id}: {e}")
                    stats["failed"] += 1
            else:
                stats["skipped"] += 1

        # Pull new skills from Edge Platform
        for skill in edge_skills:
            # Check if already synced
            already_synced = any(
                s.edge_skill_id == skill.skill_id
                for s in self.sync_map.values()
            )

            if not already_synced:
                try:
                    await self.pull_skill_from_edge(skill.skill_id)
                    stats["pulled"] += 1
                except Exception as e:
                    print(f"  ✗ Failed to pull {skill.skill_id}: {e}")
                    stats["failed"] += 1
            else:
                stats["skipped"] += 1

        print(f"\nSync complete: {stats}")
        return stats

    async def _get_edge_skill(self, skill_id: str) -> EdgeSkillRecord:
        """Get skill metadata from Edge Platform"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.edge_url}/api/v1/skills/{skill_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return EdgeSkillRecord(**response.json())

    async def health_check(self) -> Dict[str, bool]:
        """Check health of both systems"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            health = {}

            try:
                response = await client.get(f"{self.sil_url}/health")
                health["sil_registry"] = response.status_code == 200
            except:
                health["sil_registry"] = False

            try:
                response = await client.get(f"{self.edge_url}/health")
                health["edge_platform"] = response.status_code == 200
            except:
                health["edge_platform"] = False

        return health
