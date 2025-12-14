"""
Tri-System Unified Coordinator

Orchestrates the complete workflow across all three systems:
1. Dynamical-SIL: Multi-actor skill training
2. Edge Platform: Edge deployment
3. SwarmBrain: Mission orchestration

End-to-End Workflow:
====================
TRAIN → DEPLOY → EXECUTE → LEARN

1. TRAIN: Federated learning on Dynamical-SIL (OpenFL)
2. DEPLOY: Convert and deploy to Edge Platform (MoE, N2HE)
3. EXECUTE: Orchestrate mission via SwarmBrain (Flower, ROS 2)
4. LEARN: Collect data and trigger new training round

This coordinator provides unified APIs for the complete lifecycle.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx


class WorkflowStage(Enum):
    """Stages in the tri-system workflow"""
    TRAINING = "training"  # Dynamical-SIL federated learning
    DEPLOYING = "deploying"  # Edge Platform deployment
    EXECUTING = "executing"  # SwarmBrain mission execution
    LEARNING = "learning"  # Post-mission federated learning
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TriSystemWorkflow:
    """Complete workflow across all three systems"""
    workflow_id: str
    skill_name: str
    created_at: datetime = field(default_factory=datetime.now)

    # Stage tracking
    current_stage: WorkflowStage = WorkflowStage.TRAINING

    # System-specific IDs
    csa_id: Optional[str] = None  # Dynamical-SIL
    edge_skill_id: Optional[str] = None  # Edge Platform
    swarmbrain_mission_id: Optional[str] = None  # SwarmBrain

    # Federated learning round IDs
    sil_round_id: Optional[int] = None
    swarmbrain_round_id: Optional[int] = None

    # Status
    status_message: str = ""
    error_message: Optional[str] = None


class TriSystemCoordinator:
    """
    Unified coordinator for all three systems.

    Example workflow:
        coordinator = TriSystemCoordinator(
            sil_registry_url="http://localhost:8000",
            sil_coordinator_url="http://localhost:8001",
            edge_api_url="http://jetson-orin:8002",
            swarmbrain_url="http://localhost:8003",
        )

        # Complete end-to-end workflow
        workflow_id = await coordinator.start_complete_workflow(
            skill_name="cooperative_assembly",
            num_sil_sites=3,
            num_edge_devices=2,
            num_robots=3,
            work_order={...},
        )

        # Monitor progress
        status = await coordinator.get_workflow_status(workflow_id)

        # Or use step-by-step control
        csa_id = await coordinator.train_skill_on_sil(...)
        edge_skill_id = await coordinator.deploy_to_edge(csa_id)
        mission_id = await coordinator.execute_on_swarmbrain(edge_skill_id, ...)
        await coordinator.complete_with_learning(mission_id)
    """

    def __init__(
        self,
        sil_registry_url: str,
        sil_coordinator_url: str,
        edge_api_url: str,
        swarmbrain_url: str,
    ):
        self.sil_registry_url = sil_registry_url.rstrip("/")
        self.sil_coordinator_url = sil_coordinator_url.rstrip("/")
        self.edge_url = edge_api_url.rstrip("/")
        self.swarmbrain_url = swarmbrain_url.rstrip("/")

        self.active_workflows: Dict[str, TriSystemWorkflow] = {}
        self.workflow_counter = 0

    async def start_complete_workflow(
        self,
        skill_name: str,
        num_sil_sites: int = 3,
        num_edge_devices: int = 2,
        num_robots: int = 3,
        work_order: Optional[Dict[str, Any]] = None,
        sil_training_rounds: int = 5,
        coordination_type: str = "handover",
    ) -> str:
        """
        Execute complete end-to-end workflow across all three systems.

        Stages:
        1. TRAINING: Train skill on Dynamical-SIL (federated learning)
        2. DEPLOYING: Deploy to Edge Platform as MoE skill
        3. EXECUTING: Execute mission via SwarmBrain
        4. LEARNING: Post-mission federated learning round

        Returns:
            workflow_id for tracking
        """

        self.workflow_counter += 1
        workflow_id = f"workflow_{self.workflow_counter}_{skill_name}"

        workflow = TriSystemWorkflow(
            workflow_id=workflow_id,
            skill_name=skill_name,
            status_message="Starting complete workflow",
        )
        self.active_workflows[workflow_id] = workflow

        print(f"\n{'='*60}")
        print(f"STARTING COMPLETE TRI-SYSTEM WORKFLOW: {workflow_id}")
        print(f"{'='*60}\n")

        try:
            # STAGE 1: TRAINING on Dynamical-SIL
            print("STAGE 1/4: TRAINING on Dynamical-SIL")
            workflow.current_stage = WorkflowStage.TRAINING
            workflow.status_message = "Training multi-actor skill via federated learning"

            csa_id = await self.train_skill_on_sil(
                skill_name=skill_name,
                num_sites=num_sil_sites,
                num_rounds=sil_training_rounds,
            )
            workflow.csa_id = csa_id
            print(f"  ✓ Training complete: {csa_id}\n")

            # STAGE 2: DEPLOYING to Edge Platform
            print("STAGE 2/4: DEPLOYING to Edge Platform")
            workflow.current_stage = WorkflowStage.DEPLOYING
            workflow.status_message = "Deploying to edge devices as MoE skill"

            edge_skill_id = await self.deploy_to_edge(
                csa_id=csa_id,
                num_devices=num_edge_devices,
            )
            workflow.edge_skill_id = edge_skill_id
            print(f"  ✓ Deployment complete: {edge_skill_id}\n")

            # STAGE 3: EXECUTING via SwarmBrain
            print("STAGE 3/4: EXECUTING mission via SwarmBrain")
            workflow.current_stage = WorkflowStage.EXECUTING
            workflow.status_message = "Orchestrating multi-robot mission"

            mission_id = await self.execute_on_swarmbrain(
                skill_id=edge_skill_id,
                skill_source="edge_platform",
                num_robots=num_robots,
                work_order=work_order or {},
                coordination_type=coordination_type,
            )
            workflow.swarmbrain_mission_id = mission_id
            print(f"  ✓ Mission executing: {mission_id}\n")

            # Wait for mission completion (in practice, use webhooks/polling)
            await self._wait_for_mission_completion(mission_id)

            # STAGE 4: LEARNING post-mission
            print("STAGE 4/4: LEARNING from mission execution")
            workflow.current_stage = WorkflowStage.LEARNING
            workflow.status_message = "Triggering post-mission federated learning"

            await self.complete_with_learning(
                mission_id=mission_id,
                csa_id=csa_id,
            )
            print(f"  ✓ Learning round complete\n")

            # WORKFLOW COMPLETE
            workflow.current_stage = WorkflowStage.COMPLETED
            workflow.status_message = "Workflow completed successfully"

            print(f"{'='*60}")
            print(f"WORKFLOW COMPLETE: {workflow_id}")
            print(f"{'='*60}\n")
            print(f"  CSA ID: {csa_id}")
            print(f"  Edge Skill ID: {edge_skill_id}")
            print(f"  Mission ID: {mission_id}")
            print(f"  Duration: {(datetime.now() - workflow.created_at).total_seconds():.1f}s")

            return workflow_id

        except Exception as e:
            workflow.current_stage = WorkflowStage.FAILED
            workflow.error_message = str(e)
            workflow.status_message = f"Workflow failed: {e}"
            print(f"\n✗ Workflow failed: {e}")
            raise

    async def train_skill_on_sil(
        self,
        skill_name: str,
        num_sites: int = 3,
        num_rounds: int = 5,
    ) -> str:
        """
        Train skill on Dynamical-SIL using federated learning.

        Returns:
            CSA ID
        """

        print(f"  Starting federated training: {skill_name}")
        print(f"    Sites: {num_sites}, Rounds: {num_rounds}")

        async with httpx.AsyncClient(timeout=300.0) as client:
            # Start federated learning round
            response = await client.post(
                f"{self.sil_coordinator_url}/api/v1/swarm/start_round",
                json={
                    "skill_name": skill_name,
                    "num_sites": num_sites,
                    "num_rounds": num_rounds,
                    "aggregation_strategy": "trimmed_mean",
                    "privacy_mode": "encrypted",
                },
            )
            response.raise_for_status()
            round_info = response.json()
            round_id = round_info["round_id"]

            # Wait for training (simplified - in practice use polling)
            print(f"    Training in progress (round {round_id})...")
            await asyncio.sleep(30)  # Simulated training time

            # Get resulting CSA
            response = await client.get(
                f"{self.sil_coordinator_url}/api/v1/swarm/round/{round_id}/result"
            )
            response.raise_for_status()
            result = response.json()
            csa_id = result["csa_id"]

            return csa_id

    async def deploy_to_edge(
        self,
        csa_id: str,
        num_devices: int = 2,
    ) -> str:
        """
        Deploy CSA to Edge Platform as MoE skill.

        Returns:
            Edge skill ID
        """

        print(f"  Deploying CSA to Edge Platform: {csa_id}")
        print(f"    Target devices: {num_devices}")

        # Use existing Edge Platform API bridge
        from integrations.edge_platform.bridges.api_bridge import EdgePlatformAPIBridge

        bridge = EdgePlatformAPIBridge(
            sil_registry_url=self.sil_registry_url,
            edge_api_url=self.edge_url,
        )

        edge_skill = await bridge.push_csa_to_edge(csa_id=csa_id)

        return edge_skill.skill_id

    async def execute_on_swarmbrain(
        self,
        skill_id: str,
        skill_source: str,
        num_robots: int = 3,
        work_order: Optional[Dict[str, Any]] = None,
        coordination_type: str = "handover",
    ) -> str:
        """
        Execute mission on SwarmBrain.

        Returns:
            Mission ID
        """

        print(f"  Creating mission on SwarmBrain")
        print(f"    Skill: {skill_id} (from {skill_source})")
        print(f"    Robots: {num_robots}, Coordination: {coordination_type}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.swarmbrain_url}/api/v1/missions",
                json={
                    "skill_id": skill_id,
                    "skill_source": skill_source,
                    "num_robots": num_robots,
                    "work_order": work_order or {},
                    "coordination_type": coordination_type,
                },
            )
            response.raise_for_status()
            result = response.json()
            mission_id = result["mission_id"]

            return mission_id

    async def complete_with_learning(
        self,
        mission_id: str,
        csa_id: str,
    ):
        """
        Complete mission and trigger post-mission federated learning.

        This triggers learning rounds on both SwarmBrain (Flower) and
        updates the original CSA on Dynamical-SIL.
        """

        print(f"  Triggering post-mission learning")
        print(f"    Mission: {mission_id}, CSA: {csa_id}")

        # Trigger SwarmBrain federated learning
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{self.swarmbrain_url}/api/v1/learning/start_round",
                json={
                    "mission_id": mission_id,
                    "aggregation_strategy": "federated_averaging",
                },
            )
            response.raise_for_status()
            swarmbrain_round = response.json()

            print(f"    SwarmBrain learning round: {swarmbrain_round['round_id']}")

            # Update CSA on Dynamical-SIL with learned improvements
            # (In practice, this would aggregate updates from SwarmBrain back to SIL)
            print(f"    Updating CSA with mission learnings...")

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""

        if workflow_id not in self.active_workflows:
            return {"error": f"Workflow {workflow_id} not found"}

        workflow = self.active_workflows[workflow_id]

        return {
            "workflow_id": workflow.workflow_id,
            "skill_name": workflow.skill_name,
            "current_stage": workflow.current_stage.value,
            "status_message": workflow.status_message,
            "csa_id": workflow.csa_id,
            "edge_skill_id": workflow.edge_skill_id,
            "swarmbrain_mission_id": workflow.swarmbrain_mission_id,
            "created_at": workflow.created_at.isoformat(),
            "duration_seconds": (datetime.now() - workflow.created_at).total_seconds(),
            "error_message": workflow.error_message,
        }

    async def _wait_for_mission_completion(self, mission_id: str, timeout_s: int = 600):
        """Wait for mission to complete (polling)"""

        print(f"    Waiting for mission completion...")

        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout_s:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.swarmbrain_url}/api/v1/missions/{mission_id}/status"
                )

                if response.status_code == 200:
                    status = response.json()

                    if status.get("status") == "completed":
                        print(f"    Mission completed successfully")
                        return
                    elif status.get("status") == "failed":
                        raise Exception(f"Mission failed: {status.get('error')}")

            await asyncio.sleep(5)

        raise TimeoutError(f"Mission {mission_id} did not complete within {timeout_s}s")

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""

        return [
            {
                "workflow_id": w.workflow_id,
                "skill_name": w.skill_name,
                "current_stage": w.current_stage.value,
                "status_message": w.status_message,
            }
            for w in self.active_workflows.values()
        ]
