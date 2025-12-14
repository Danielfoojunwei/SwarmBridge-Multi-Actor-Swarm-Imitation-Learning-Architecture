#!/usr/bin/env python3
"""
SwarmBridge 2.0 - Complete Workflow Demonstration

This script demonstrates the complete SwarmBridge 2.0 pipeline:

STAGE 1: CAPTURE - Record multi-actor demonstrations
STAGE 2: PROCESS - Process and prepare training data
STAGE 3: TRAIN   - Train cooperative policy
STAGE 4: PACKAGE - Package as CSA
STAGE 5: PUBLISH - Upload to registry
STAGE 6: FEDERATE - (Optional) Submit to federated learning
STAGE 7: EXECUTE  - (Optional) Execute via Edge Platform

This uses mock adapters for demonstration without external services.
"""

import asyncio
import sys
from pathlib import Path

# Add swarmbridge to path
sys.path.insert(0, str(Path(__file__).parent))

from swarmbridge.schemas import (
    SharedRoleSchema,
    CoordinationPrimitives,
    CoordinationType,
)
from swarmbridge.adapters import (
    RegistryAdapter,
    FederatedLearningAdapter,
    EdgePlatformRuntimeAdapter,
)


def print_banner(title: str):
    """Print section banner"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def demonstrate_swarmbridge_2_0():
    """
    Complete SwarmBridge 2.0 workflow demonstration.
    """

    print_banner("SWARMBRIDGE 2.0 - COMPLETE WORKFLOW DEMONSTRATION")

    # ========================================================================
    # SETUP: Define skill parameters
    # ========================================================================

    print("\n📋 Skill Configuration:")
    skill_config = {
        "skill_name": "cooperative_assembly",
        "num_actors": 2,
        "coordination_type": "handover",
        "num_demonstrations": 5,
        "version": "1.0.0",
    }

    for key, value in skill_config.items():
        print(f"   • {key}: {value}")

    # ========================================================================
    # STAGE 1: Create Role Schema (from shared schemas)
    # ========================================================================

    print_banner("STAGE 1: CREATE ROLE SCHEMA")

    roles = SharedRoleSchema.create_role_set(
        num_actors=skill_config["num_actors"],
        coordination_type=skill_config["coordination_type"],
    )

    print(f"\n✅ Created {len(roles)} roles:")
    for role in roles:
        print(f"   • {role.role_id} ({role.role_type.value})")
        print(f"     - Capabilities: {', '.join(role.capabilities)}")
        print(f"     - Can coordinate with: {', '.join(role.can_coordinate_with)}")

    # ========================================================================
    # STAGE 2: Create Coordination Primitive (from shared schemas)
    # ========================================================================

    print_banner("STAGE 2: CREATE COORDINATION PRIMITIVE")

    primitive = CoordinationPrimitives.get_primitive(
        coordination_type=CoordinationType.HANDOVER,
        roles=[role.role_id for role in roles],
        parameters={
            "handover_location": "midpoint",
            "object_id": "assembly_part_A",
            "grasp_force": 5.0,
            "transfer_speed": 0.1,
        },
    )

    print(f"\n✅ Created {primitive.coordination_type.value} primitive:")
    print(f"   • Participating roles: {', '.join(primitive.participating_roles)}")
    print(f"   • Timeout: {primitive.timeout_s}s")
    print(f"   • Retry attempts: {primitive.retry_attempts}")
    print(f"\n   Parameters:")
    for key, value in primitive.parameters.items():
        print(f"     - {key}: {value}")

    # Validate primitive
    is_valid, message = CoordinationPrimitives.validate_primitive(primitive)
    print(f"\n   ✅ Validation: {message}")

    # ========================================================================
    # STAGE 3: Convert to SwarmBrain Task Graph
    # ========================================================================

    print_banner("STAGE 3: GENERATE SWARMBRAIN TASK GRAPH")

    task_graph = CoordinationPrimitives.to_swarmbrain_task_graph(primitive)

    print(f"\n✅ Generated task graph with {len(task_graph['tasks'])} tasks:")
    for i, task in enumerate(task_graph["tasks"], 1):
        print(f"\n   Task {i}: {task['task_id']}")
        print(f"     - Assigned roles: {', '.join(task['assigned_roles'])}")
        print(f"     - Dependencies: {', '.join(task.get('dependencies', [])) or 'None'}")
        if "coordination" in task:
            print(f"     - Coordination: {task['coordination']['type']}")

    # ========================================================================
    # STAGE 4: Convert Roles to Different Formats
    # ========================================================================

    print_banner("STAGE 4: DEMONSTRATE ROLE FORMAT CONVERSIONS")

    print("\n✅ Converting 'giver' role to different system formats:")

    giver_role = roles[0]

    # CSA format (for SwarmBridge)
    csa_format = SharedRoleSchema.to_csa_format(giver_role)
    print(f"\n   CSA Format (SwarmBridge):")
    print(f"     - role_id: {csa_format['role_id']}")
    print(f"     - capabilities: {csa_format['capabilities']}")
    print(f"     - observation_dim: {csa_format['observation_dim']}")
    print(f"     - action_dim: {csa_format['action_dim']}")

    # MoE format (for Edge Platform)
    moe_format = SharedRoleSchema.to_moe_format(giver_role)
    print(f"\n   MoE Format (Edge Platform):")
    print(f"     - expert_id: {moe_format['expert_id']}")
    print(f"     - specialization: {moe_format['specialization']}")
    print(f"     - priority: {moe_format['priority']}")

    # SwarmBrain format
    sb_format = SharedRoleSchema.to_swarmbrain_format(giver_role)
    print(f"\n   SwarmBrain Format:")
    print(f"     - role_id: {sb_format['role_id']}")
    print(f"     - role_type: {sb_format['role_type']}")
    print(f"     - coordination_partners: {sb_format['coordination_partners']}")

    # ========================================================================
    # STAGE 5: Demonstrate Adapter Pattern (Mock Adapters)
    # ========================================================================

    print_banner("STAGE 5: DEMONSTRATE SWARMBRIDGE 2.0 ADAPTERS")

    print("\n📝 Note: Using mock adapters (no external services required)")

    # Import mock adapters
    try:
        from swarmbridge.adapters.registry_adapter import MockRegistryAdapter
        from swarmbridge.adapters.federated_adapter import MockFederatedLearningAdapter
        from swarmbridge.adapters.runtime_adapter import MockEdgePlatformRuntimeAdapter

        # Initialize adapters
        registry_adapter = MockRegistryAdapter()
        federated_adapter = MockFederatedLearningAdapter()
        runtime_adapter = MockEdgePlatformRuntimeAdapter(registry_adapter)

        print("\n✅ Initialized adapters:")
        print("   • RegistryAdapter (mock)")
        print("   • FederatedLearningAdapter (mock)")
        print("   • EdgePlatformRuntimeAdapter (mock)")

        # ====================================================================
        # ADAPTER DEMO 1: Registry Operations
        # ====================================================================

        print("\n" + "-" * 70)
        print("  ADAPTER DEMO 1: Registry Operations")
        print("-" * 70)

        # Simulate CSA upload (would normally have a real tarball)
        print("\n🔄 Simulating CSA upload to registry...")

        # Note: In real usage, you'd have a CSA tarball
        # For demo, we'll simulate with a dummy path
        print("   (In production: upload packaged CSA tarball)")

        # ====================================================================
        # ADAPTER DEMO 2: Federated Learning
        # ====================================================================

        print("\n" + "-" * 70)
        print("  ADAPTER DEMO 2: Federated Learning")
        print("-" * 70)

        # Submit local update
        print("\n🔄 Submitting local CSA update to federated learning service...")
        update_result = await federated_adapter.submit_local_update(
            csa_id="cooperative_assembly_v1.0.0_site1",
            skill_name="cooperative_assembly",
            metadata={
                "site_id": "site_1",
                "num_demonstrations": 5,
                "training_iterations": 1000,
            },
        )
        print(f"   ✅ Update submitted: {update_result['update_id']}")

        # Request federated merge
        print("\n🔄 Requesting federated merge...")
        merge_result = await federated_adapter.request_merge(
            skill_name="cooperative_assembly",
            merge_strategy="fedavg",
            min_updates=2,
        )
        print(f"   ✅ Merge completed: {merge_result['global_csa_id']}")

        # ====================================================================
        # ADAPTER DEMO 3: Edge Platform Runtime
        # ====================================================================

        print("\n" + "-" * 70)
        print("  ADAPTER DEMO 3: Edge Platform Runtime Execution")
        print("-" * 70)

        # Execute skill
        print("\n🔄 Executing skill via Edge Platform...")
        execution_id = await runtime_adapter.execute_skill(
            csa_id="cooperative_assembly_global_v0",
            robot_id="robot_team_1",
            task_parameters={
                "target_object": "assembly_part_A",
                "assembly_location": "workstation_1",
            },
            execution_mode="sim",
        )
        print(f"   ✅ Execution started: {execution_id}")

        # Wait for completion
        print("\n🔄 Waiting for execution to complete...")
        final_status = await runtime_adapter.wait_for_completion(
            execution_id,
            poll_interval_s=0.5,
            max_wait_s=10.0,
        )
        print(f"   ✅ Execution finished: {final_status['status']}")

        # Get logs
        logs = await runtime_adapter.get_execution_logs(execution_id)
        print(f"\n📋 Execution logs ({len(logs)} lines):")
        for log in logs:
            print(f"   {log}")

    except ImportError as e:
        print(f"\n⚠️  Mock adapters not available: {e}")
        print("   (This is expected if httpx is not installed)")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print_banner("DEMONSTRATION COMPLETE")

    print("\n✅ SwarmBridge 2.0 Key Features Demonstrated:")
    print("   1. ✅ Shared role schema (compatible with CSA, MoE, SwarmBrain)")
    print("   2. ✅ Coordination primitives library (reusable patterns)")
    print("   3. ✅ Task graph generation for SwarmBrain")
    print("   4. ✅ Role format conversions across systems")
    print("   5. ✅ Modular adapter pattern for external services")
    print("   6. ✅ Federated learning integration (via adapter)")
    print("   7. ✅ Edge Platform runtime delegation (via adapter)")

    print("\n📚 SwarmBridge 2.0 Architecture Benefits:")
    print("   • Focused on core competencies (capture, train, package)")
    print("   • Delegates runtime to Edge Platform")
    print("   • Delegates federated learning to external service")
    print("   • Shared schemas ensure cross-system compatibility")
    print("   • Clean separation of concerns")

    print("\n🔗 Complete Ecosystem:")
    print("   SwarmBridge  → Capture demonstrations, train cooperative policies")
    print("   Edge Platform → Deploy MoE skills, execute on robots")
    print("   SwarmBrain   → Orchestrate multi-robot missions")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(demonstrate_swarmbridge_2_0())
