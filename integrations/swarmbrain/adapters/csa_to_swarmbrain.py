"""
CSA to SwarmBrain Skill Adapter

Converts Cooperative Skill Artefacts (CSA) from Dynamical-SIL into
SwarmBrain skill format for mission orchestration.

Integration Strategy:
- CSA policy adapters → SwarmBrain role-conditioned policies
- CSA coordination encoder → Coordination primitives
- CSA roles → Robot role assignments
- Multi-actor coordination → SwarmBrain task graphs
"""

import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np


@dataclass
class SwarmBrainSkillMetadata:
    """Metadata for SwarmBrain skills"""
    skill_name: str
    version: str
    roles: List[str]  # leader, follower, observer
    coordination_type: str  # handover, mutex, barrier, rendezvous
    min_robots: int
    max_robots: int
    task_graph: Dict[str, Any]
    capabilities_required: List[str]
    source_csa_id: Optional[str] = None


@dataclass
class RoleConditionedPolicy:
    """Role-conditioned policy for SwarmBrain"""
    role_id: str
    policy_weights: Dict[str, torch.Tensor]
    observation_space: Dict[str, int]
    action_space: Dict[str, int]
    control_frequency_hz: int = 100


class CSAToSwarmBrainAdapter:
    """
    Converts CSA packages to SwarmBrain skill format.

    SwarmBrain skills consist of:
    1. Role-conditioned policies (reflex layer)
    2. Coordination primitives (coordination layer)
    3. Task graphs for mission planning
    4. ROS 2 action interfaces

    Example:
        adapter = CSAToSwarmBrainAdapter()
        skill_metadata = adapter.convert_csa_to_swarmbrain(
            csa_path="artifacts/cooperative_assembly_v1.0.tar.gz",
            output_dir="swarmbrain_skills/cooperative_assembly",
            coordination_type="handover",
        )
    """

    def __init__(self, ros2_workspace: Optional[Path] = None):
        self.ros2_workspace = ros2_workspace or Path("ros2_ws")

    def convert_csa_to_swarmbrain(
        self,
        csa_path: Path,
        output_dir: Path,
        coordination_type: str = "handover",
    ) -> SwarmBrainSkillMetadata:
        """
        Convert CSA to SwarmBrain skill format.

        Args:
            csa_path: Path to CSA tarball
            output_dir: Directory to save SwarmBrain skill
            coordination_type: Type of coordination (handover, mutex, barrier, rendezvous)

        Returns:
            SwarmBrain skill metadata
        """

        print(f"Converting CSA {csa_path} to SwarmBrain skill...")

        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Extract CSA
            with tarfile.open(csa_path, "r:gz") as tar:
                tar.extractall(tmpdir_path)

            # Load manifest
            with open(tmpdir_path / "manifest.json") as f:
                manifest = json.load(f)

            # Extract roles and create role-conditioned policies
            roles = manifest.get("roles", [])
            policies = []

            for role in roles:
                role_id = role["role_id"]
                adapter_path = tmpdir_path / f"roles/{role_id}/policy_adapter.pt"

                if adapter_path.exists():
                    policy = self._create_role_policy(
                        adapter_path=adapter_path,
                        role_config=role,
                    )
                    policies.append(policy)

                    # Save policy for SwarmBrain
                    policy_output = output_dir / f"policies/{role_id}_policy.pt"
                    policy_output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(policy.policy_weights, policy_output)

            # Create task graph from CSA coordination
            task_graph = self._create_task_graph(
                manifest=manifest,
                coordination_type=coordination_type,
            )

            # Create coordination primitives configuration
            coord_config = self._create_coordination_config(
                roles=[p.role_id for p in policies],
                coordination_type=coordination_type,
            )

            # Save coordination config
            with open(output_dir / "coordination.json", "w") as f:
                json.dump(coord_config, f, indent=2)

            # Save task graph
            with open(output_dir / "task_graph.json", "w") as f:
                json.dump(task_graph, f, indent=2)

            # Create skill metadata
            metadata = SwarmBrainSkillMetadata(
                skill_name=manifest["skill_name"],
                version=manifest["version"],
                roles=[p.role_id for p in policies],
                coordination_type=coordination_type,
                min_robots=len(policies),
                max_robots=len(policies) * 2,  # Allow scaling
                task_graph=task_graph,
                capabilities_required=self._extract_capabilities(manifest),
                source_csa_id=manifest.get("csa_id"),
            )

            # Save metadata
            with open(output_dir / "skill_metadata.json", "w") as f:
                json.dump({
                    "skill_name": metadata.skill_name,
                    "version": metadata.version,
                    "roles": metadata.roles,
                    "coordination_type": metadata.coordination_type,
                    "min_robots": metadata.min_robots,
                    "max_robots": metadata.max_robots,
                    "task_graph": metadata.task_graph,
                    "capabilities_required": metadata.capabilities_required,
                    "source_csa_id": metadata.source_csa_id,
                    "format": "swarmbrain_v1",
                }, f, indent=2)

        print(f"✓ SwarmBrain skill created at {output_dir}")
        return metadata

    def _create_role_policy(
        self,
        adapter_path: Path,
        role_config: Dict[str, Any],
    ) -> RoleConditionedPolicy:
        """Create role-conditioned policy from CSA adapter"""

        weights = torch.load(adapter_path, map_location="cpu")

        return RoleConditionedPolicy(
            role_id=role_config["role_id"],
            policy_weights=weights,
            observation_space={
                "joint_positions": role_config.get("observation_dim", 15),
                "task_features": 64,
            },
            action_space={
                "joint_velocities": role_config.get("action_dim", 7),
            },
            control_frequency_hz=100,
        )

    def _create_task_graph(
        self,
        manifest: Dict[str, Any],
        coordination_type: str,
    ) -> Dict[str, Any]:
        """
        Create task graph for SwarmBrain orchestrator.

        Task graph defines:
        - Task nodes and dependencies
        - Role assignments
        - Synchronization points
        """

        skill_name = manifest["skill_name"]
        roles = manifest.get("roles", [])

        # Create task nodes based on skill structure
        if coordination_type == "handover":
            return {
                "tasks": [
                    {
                        "task_id": "approach",
                        "assigned_roles": [roles[0]["role_id"]],
                        "dependencies": [],
                        "coordination": None,
                    },
                    {
                        "task_id": "handover_sync",
                        "assigned_roles": [r["role_id"] for r in roles],
                        "dependencies": ["approach"],
                        "coordination": {"type": "rendezvous", "timeout_s": 5.0},
                    },
                    {
                        "task_id": "transfer",
                        "assigned_roles": [roles[1]["role_id"]] if len(roles) > 1 else [],
                        "dependencies": ["handover_sync"],
                        "coordination": None,
                    },
                ],
                "success_criteria": "all_tasks_completed",
            }
        elif coordination_type == "mutex":
            return {
                "tasks": [
                    {
                        "task_id": "resource_access",
                        "assigned_roles": [r["role_id"] for r in roles],
                        "dependencies": [],
                        "coordination": {"type": "mutex", "resource_id": "shared_workspace"},
                    },
                ],
                "success_criteria": "all_tasks_completed",
            }
        elif coordination_type == "barrier":
            return {
                "tasks": [
                    {
                        "task_id": "parallel_work",
                        "assigned_roles": [r["role_id"] for r in roles],
                        "dependencies": [],
                        "coordination": None,
                    },
                    {
                        "task_id": "sync_point",
                        "assigned_roles": [r["role_id"] for r in roles],
                        "dependencies": ["parallel_work"],
                        "coordination": {"type": "barrier", "timeout_s": 10.0},
                    },
                    {
                        "task_id": "combined_action",
                        "assigned_roles": [r["role_id"] for r in roles],
                        "dependencies": ["sync_point"],
                        "coordination": None,
                    },
                ],
                "success_criteria": "all_tasks_completed",
            }
        else:  # rendezvous
            return {
                "tasks": [
                    {
                        "task_id": "independent_prep",
                        "assigned_roles": [r["role_id"] for r in roles],
                        "dependencies": [],
                        "coordination": None,
                    },
                    {
                        "task_id": "rendezvous_point",
                        "assigned_roles": [r["role_id"] for r in roles],
                        "dependencies": ["independent_prep"],
                        "coordination": {"type": "rendezvous", "location": "meeting_point"},
                    },
                    {
                        "task_id": "cooperative_task",
                        "assigned_roles": [r["role_id"] for r in roles],
                        "dependencies": ["rendezvous_point"],
                        "coordination": None,
                    },
                ],
                "success_criteria": "all_tasks_completed",
            }

    def _create_coordination_config(
        self,
        roles: List[str],
        coordination_type: str,
    ) -> Dict[str, Any]:
        """Create coordination primitives configuration"""

        return {
            "coordination_type": coordination_type,
            "roles": roles,
            "parameters": {
                "timeout_s": 10.0,
                "retry_attempts": 3,
                "collision_avoidance": True,
                "formation_maintenance": True,
            },
            "ros2_topics": {
                "coordination_state": f"/swarm/coordination/{coordination_type}/state",
                "role_assignments": "/swarm/roles",
                "task_status": "/swarm/tasks/status",
            },
        }

    def _extract_capabilities(self, manifest: Dict[str, Any]) -> List[str]:
        """Extract required capabilities from CSA manifest"""

        capabilities = set()

        for role in manifest.get("roles", []):
            capabilities.update(role.get("capabilities", []))

        return list(capabilities)

    def create_ros2_launch_file(
        self,
        skill_dir: Path,
        metadata: SwarmBrainSkillMetadata,
        output_path: Path,
    ):
        """
        Create ROS 2 launch file for SwarmBrain skill.

        This launch file starts:
        - Role-conditioned policy nodes
        - Coordination primitive handlers
        - Task execution monitors
        """

        launch_content = f"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    nodes = []

    # Launch policy nodes for each role
    roles = {metadata.roles}

    for role_id in roles:
        node = Node(
            package='robot_control',
            executable='policy_executor',
            name=f'policy_{{role_id}}',
            parameters=[{{
                'skill_name': '{metadata.skill_name}',
                'role_id': role_id,
                'policy_path': str(Path('{skill_dir}') / f'policies/{{role_id}}_policy.pt'),
                'control_frequency_hz': 100,
            }}],
        )
        nodes.append(node)

    # Launch coordination primitive handler
    coord_node = Node(
        package='robot_control',
        executable='coordination_handler',
        name='coordination_handler',
        parameters=[{{
            'coordination_type': '{metadata.coordination_type}',
            'config_path': str(Path('{skill_dir}') / 'coordination.json'),
        }}],
    )
    nodes.append(coord_node)

    # Launch task graph executor
    task_node = Node(
        package='orchestrator',
        executable='task_executor',
        name='task_executor',
        parameters=[{{
            'task_graph_path': str(Path('{skill_dir}') / 'task_graph.json'),
            'skill_name': '{metadata.skill_name}',
        }}],
    )
    nodes.append(task_node)

    return LaunchDescription(nodes)
"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(launch_content)

        print(f"✓ Created ROS 2 launch file: {output_path}")
