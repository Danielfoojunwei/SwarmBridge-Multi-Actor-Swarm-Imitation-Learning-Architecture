"""
SwarmBrain Mission Bridge

Connects Dynamical-SIL and Edge Platform skills to SwarmBrain
mission orchestration system.

Enables:
- Mission creation using CSA/MoE skills
- Robot fleet coordination
- Federated learning orchestration
- Real-time mission monitoring
"""

import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
from pydantic import BaseModel


class MissionStatus(Enum):
    """Mission execution status"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RobotCapabilities:
    """Robot capability declaration"""
    robot_id: str
    capabilities: List[str]
    location: Dict[str, float]  # {x, y, z}
    status: str  # available, busy, offline
    battery_level: float


@dataclass
class MissionSpec:
    """Mission specification"""
    mission_id: str
    skill_name: str
    skill_source: str  # dynamical_sil, edge_platform, swarmbrain
    required_robots: int
    coordination_type: str
    work_order: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    status: MissionStatus = MissionStatus.PENDING


class SwarmBrainMissionBridge:
    """
    Bridge for orchestrating missions using skills from all three systems.

    Example:
        bridge = SwarmBrainMissionBridge(
            swarmbrain_url="http://localhost:8000",
            sil_registry_url="http://localhost:8001",
            edge_api_url="http://jetson-orin:8002",
        )

        # Create mission using CSA from Dynamical-SIL
        mission_id = await bridge.create_mission_from_csa(
            csa_id="csa_cooperative_assembly",
            num_robots=3,
            work_order={...},
        )

        # Monitor mission
        status = await bridge.get_mission_status(mission_id)

        # Complete mission and trigger federated learning
        await bridge.complete_mission_with_learning(mission_id)
    """

    def __init__(
        self,
        swarmbrain_url: str,
        sil_registry_url: Optional[str] = None,
        edge_api_url: Optional[str] = None,
    ):
        self.swarmbrain_url = swarmbrain_url.rstrip("/")
        self.sil_url = sil_registry_url.rstrip("/") if sil_registry_url else None
        self.edge_url = edge_api_url.rstrip("/") if edge_api_url else None

        self.active_missions: Dict[str, MissionSpec] = {}
        self.robot_fleet: Dict[str, RobotCapabilities] = {}

    async def register_robot(
        self,
        robot_id: str,
        capabilities: List[str],
        location: Dict[str, float],
    ) -> bool:
        """Register a robot with SwarmBrain orchestrator"""

        robot_caps = RobotCapabilities(
            robot_id=robot_id,
            capabilities=capabilities,
            location=location,
            status="available",
            battery_level=1.0,
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.swarmbrain_url}/api/v1/robots",
                json={
                    "robot_id": robot_id,
                    "capabilities": capabilities,
                    "location": location,
                },
            )

            if response.status_code == 200:
                self.robot_fleet[robot_id] = robot_caps
                print(f"✓ Registered robot: {robot_id}")
                return True
            else:
                print(f"✗ Failed to register robot: {response.text}")
                return False

    async def create_mission_from_csa(
        self,
        csa_id: str,
        num_robots: int,
        work_order: Dict[str, Any],
        coordination_type: str = "handover",
    ) -> str:
        """
        Create a mission using a CSA from Dynamical-SIL.

        Steps:
        1. Fetch CSA from registry
        2. Convert to SwarmBrain skill format
        3. Create mission specification
        4. Submit to SwarmBrain orchestrator
        """

        print(f"Creating mission from CSA {csa_id}...")

        if not self.sil_url:
            raise ValueError("SIL registry URL not configured")

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Fetch CSA metadata
            response = await client.get(f"{self.sil_url}/api/v1/csa/{csa_id}")
            response.raise_for_status()
            csa_metadata = response.json()

            # Create mission in SwarmBrain
            mission_spec = {
                "skill_name": csa_metadata["skill_name"],
                "skill_source": "dynamical_sil",
                "source_id": csa_id,
                "num_robots": num_robots,
                "coordination_type": coordination_type,
                "work_order": work_order,
            }

            response = await client.post(
                f"{self.swarmbrain_url}/api/v1/missions",
                json=mission_spec,
            )
            response.raise_for_status()
            result = response.json()
            mission_id = result["mission_id"]

            # Track mission
            self.active_missions[mission_id] = MissionSpec(
                mission_id=mission_id,
                skill_name=csa_metadata["skill_name"],
                skill_source="dynamical_sil",
                required_robots=num_robots,
                coordination_type=coordination_type,
                work_order=work_order,
            )

            print(f"✓ Created mission: {mission_id}")
            return mission_id

    async def create_mission_from_edge_skill(
        self,
        skill_id: str,
        num_robots: int,
        work_order: Dict[str, Any],
    ) -> str:
        """Create mission using Edge Platform MoE skill"""

        print(f"Creating mission from Edge skill {skill_id}...")

        if not self.edge_url:
            raise ValueError("Edge Platform URL not configured")

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Fetch skill metadata
            response = await client.get(f"{self.edge_url}/api/v1/skills/{skill_id}")
            response.raise_for_status()
            skill_metadata = response.json()

            # Create mission
            mission_spec = {
                "skill_name": skill_metadata["skill_name"],
                "skill_source": "edge_platform",
                "source_id": skill_id,
                "num_robots": num_robots,
                "work_order": work_order,
            }

            response = await client.post(
                f"{self.swarmbrain_url}/api/v1/missions",
                json=mission_spec,
            )
            response.raise_for_status()
            result = response.json()
            mission_id = result["mission_id"]

            print(f"✓ Created mission: {mission_id}")
            return mission_id

    async def get_mission_status(self, mission_id: str) -> Dict[str, Any]:
        """Get current status of a mission"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.swarmbrain_url}/api/v1/missions/{mission_id}/status"
            )
            response.raise_for_status()
            return response.json()

    async def complete_mission_with_learning(
        self,
        mission_id: str,
        trigger_federated_round: bool = True,
    ):
        """
        Complete a mission and optionally trigger federated learning.

        After successful mission completion:
        1. Collect execution data from robots
        2. Trigger federated learning round (if enabled)
        3. Update skill models across all systems
        """

        print(f"Completing mission {mission_id}...")

        # Get mission status
        status = await self.get_mission_status(mission_id)

        if status["status"] != "completed":
            print(f"⚠ Mission not completed yet: {status['status']}")
            return

        if trigger_federated_round:
            # Trigger federated learning round in SwarmBrain
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.swarmbrain_url}/api/v1/learning/start_round",
                    json={
                        "mission_id": mission_id,
                        "aggregation_strategy": "federated_averaging",
                    },
                )
                response.raise_for_status()
                round_info = response.json()

                print(f"✓ Started federated learning round: {round_info['round_id']}")

        # Update mission status locally
        if mission_id in self.active_missions:
            self.active_missions[mission_id].status = MissionStatus.COMPLETED

    async def list_available_robots(self) -> List[RobotCapabilities]:
        """List all available robots in the fleet"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.swarmbrain_url}/api/v1/robots")
            response.raise_for_status()
            robots_data = response.json()

            return [
                RobotCapabilities(
                    robot_id=r["robot_id"],
                    capabilities=r["capabilities"],
                    location=r["location"],
                    status=r["status"],
                    battery_level=r.get("battery_level", 1.0),
                )
                for r in robots_data
            ]

    async def assign_robots_to_mission(
        self,
        mission_id: str,
        robot_ids: List[str],
        role_assignments: Dict[str, str],
    ):
        """
        Assign specific robots to mission with role assignments.

        Args:
            mission_id: Mission to assign robots to
            robot_ids: List of robot IDs
            role_assignments: Dict mapping robot_id → role_id
        """

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.swarmbrain_url}/api/v1/missions/{mission_id}/assign_robots",
                json={
                    "robot_ids": robot_ids,
                    "role_assignments": role_assignments,
                },
            )
            response.raise_for_status()

            print(f"✓ Assigned {len(robot_ids)} robots to mission {mission_id}")
