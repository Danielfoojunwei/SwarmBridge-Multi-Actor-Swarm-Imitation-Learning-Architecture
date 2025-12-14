# SwarmBridge 2.0: Refactored Architecture

## Overview

SwarmBridge has been refactored to **focus on its core competencies** while delegating runtime and orchestration to specialized external systems.

### **Core Focus (What SwarmBridge Does)**

✅ **Multi-Actor Demonstration Capture** (ROS 2)
✅ **Cooperative Imitation Learning** (Training)
✅ **Skill Artifact Packaging** (CSA)
✅ **Registry Publishing**

### **Delegated to External Systems (What SwarmBridge Doesn't Do)**

❌ Runtime Execution → **Edge Platform** (Dynamical API)
❌ Federated Learning Orchestration → **Federated Learning Service**
❌ Mission Orchestration → **SwarmBrain**

---

## Refactored Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   SWARMBRIDGE 2.0                          │
│              (Capture, Train, Package)                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │          END-TO-END PIPELINE                        │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │                                                     │  │
│  │  CAPTURE ──► PROCESS ──► TRAIN ──► PACKAGE ──► PUBLISH│  │
│  │                                                     │  │
│  │  ROS2     │  Extract  │  Coop   │  CSA    │  Registry│  │
│  │  Demos    │  Obs/Act  │  IL     │  Build  │  Upload │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │          ADAPTERS (Thin Wrappers)                   │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │                                                     │  │
│  │  • FederatedLearningAdapter                        │  │
│  │    → External federated service (not OpenFL)       │  │
│  │                                                     │  │
│  │  • RegistryAdapter                                 │  │
│  │    → CSA upload/download                           │  │
│  │                                                     │  │
│  │  • EdgePlatformRuntimeAdapter                      │  │
│  │    → Execution via Dynamical API (not local)       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │          SHARED SCHEMAS                             │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │                                                     │  │
│  │  • SharedRoleSchema                                │  │
│  │    → Unified role definitions                      │  │
│  │                                                     │  │
│  │  • CoordinationPrimitives                          │  │
│  │    → Library of coordination patterns              │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
             │                 │                 │
             ▼                 ▼                 ▼
    Edge Platform      Federated Service    SwarmBrain
    (Runtime)          (FL Orchestration)   (Mission Orchestration)
```

---

## Key Changes

### **1. Modular Pipeline**

**Before:** Scattered capture and training scripts
**After:** Unified `SwarmBridgePipeline` class

```python
from swarmbridge import SwarmBridgePipeline

pipeline = SwarmBridgePipeline(
    registry_url="http://localhost:8000",
    federated_service_url="http://localhost:8001",
)

# Complete end-to-end
csa_id = await pipeline.run_complete_pipeline(
    skill_name="cooperative_assembly",
    num_demonstrations=3,
    num_actors=2,
    coordination_type="handover",
)

# Output:
# STAGE 1/5: CAPTURE multi-actor demonstrations
#   ✓ Captured 3 demonstrations
# STAGE 2/5: PROCESS demonstrations
#   ✓ Processed 3 trajectories
# STAGE 3/5: TRAIN cooperative imitation learning
#   ✓ Training complete
# STAGE 4/5: PACKAGE as CSA artifact
#   ✓ CSA packaged
# STAGE 5/5: PUBLISH to registry
#   ✓ Published: csa_assembly_v1.0
```

### **2. Federated Learning Adapter**

**Before:** Direct OpenFL dependencies in core modules
**After:** Adapter pattern to external service

```python
from swarmbridge.adapters import FederatedLearningAdapter

adapter = FederatedLearningAdapter(
    service_url="http://localhost:8001"
)

# Submit local update (service handles OpenFL/Flower)
await adapter.submit_local_update(
    csa_id="csa_123",
    skill_name="cooperative_assembly",
)

# Request merge
merged_csa_id = await adapter.request_merge(
    skill_name="cooperative_assembly",
)

# Unlearning
await adapter.request_unlearning(
    csa_id="csa_123",
    method="influence_removal",
)
```

**Benefits:**
- ✅ No OpenFL in SwarmBridge core
- ✅ FL framework-agnostic (service can use OpenFL, Flower, etc.)
- ✅ Unlearning support via service
- ✅ Easier to test and maintain

### **3. Thin Runtime Wrapper**

**Before:** Local BT execution, MoveIt2 integration
**After:** Delegate to Edge Platform's Dynamical API

```python
from swarmbridge.adapters import EdgePlatformRuntimeAdapter

runtime = EdgePlatformRuntimeAdapter(
    edge_api_url="http://jetson-orin.local:8001",
    registry_url="http://localhost:8000",
)

# Fetch from registry and execute on Edge Platform
execution_id = await runtime.execute_skill(
    csa_id="csa_123",
    robot_id="robot_1",
    task_parameters={"object_id": "cube_red"},
)

# Monitor
status = await runtime.get_execution_status(execution_id)
print(f"Status: {status['state']}")  # running, completed, failed
```

**What's Removed:**
- ❌ Local behavior tree execution
- ❌ Local MoveIt2 integration
- ❌ Local skill runtime logic

**What's Kept:**
- ✅ Fetch skills from registry
- ✅ Invoke Dynamical API
- ✅ Monitor execution

### **4. Shared Schemas**

**Before:** Different role/coordination formats per system
**After:** Unified schemas for compatibility

```python
from swarmbridge.schemas import SharedRoleSchema, CoordinationPrimitives, CoordinationType

# Create role set
schema = SharedRoleSchema()
roles = schema.create_role_set(
    num_actors=2,
    coordination_type="handover",
)

# roles = [
#     RoleDefinition(role_id="giver", role_type=RoleType.LEADER, ...),
#     RoleDefinition(role_id="receiver", role_type=RoleType.FOLLOWER, ...)
# ]

# Convert to system-specific formats
csa_format = schema.to_csa_format(roles[0])
moe_format = schema.to_moe_format(roles[0])
swarmbrain_format = schema.to_swarmbrain_format(roles[0])

# Coordination primitives
primitives = CoordinationPrimitives()
handover = primitives.get_primitive(
    CoordinationType.HANDOVER,
    roles=["giver", "receiver"],
)

# Convert to task graph
task_graph = primitives.to_swarmbrain_task_graph(handover)
```

**Benefits:**
- ✅ Single source of truth for roles
- ✅ Automatic conversion to CSA/MoE/SwarmBrain formats
- ✅ Ensures compatibility across systems
- ✅ Standard coordination primitives library

---

## File Structure

```
swarmbridge/
├── __init__.py                 # Main package
├── pipeline/
│   ├── __init__.py            # SwarmBridgePipeline
│   ├── capture.py             # ROS 2 capture wrapper
│   └── processing.py          # Demonstration processing
├── adapters/
│   ├── __init__.py
│   ├── federated_adapter.py   # Federated learning service
│   ├── registry_adapter.py    # Registry operations
│   └── runtime_adapter.py     # Edge Platform execution
└── schemas/
    ├── __init__.py
    ├── role_schema.py          # Shared role definitions
    └── coordination_primitives.py  # Coordination library
```

---

## Usage Examples

### **Example 1: Complete Pipeline**

```python
from swarmbridge import SwarmBridgePipeline

# Configure pipeline
pipeline = SwarmBridgePipeline(
    registry_url="http://localhost:8000",
    federated_service_url="http://localhost:8001",
)

# Run complete workflow
csa_id = await pipeline.run_complete_pipeline(
    skill_name="cooperative_pick_place",
    num_demonstrations=3,
    num_actors=2,
    roles=["left_arm", "right_arm"],
    coordination_type="collaborative_manipulation",
    enable_federated_learning=True,
)

print(f"CSA created and published: {csa_id}")
```

### **Example 2: Step-by-Step Pipeline**

```python
# More granular control

# 1. Capture
demonstrations = await pipeline.capture_demonstrations(
    skill_name="cooperative_assembly",
    num_demonstrations=3,
    num_actors=2,
)

# 2. Process
processed_data = await pipeline.process_demonstrations(
    demonstrations=demonstrations,
    num_actors=2,
)

# 3. Train
trained_model = await pipeline.train_cooperative_policy(
    dataset=processed_data,
    skill_name="cooperative_assembly",
    roles=["leader", "follower"],
    coordination_type="handover",
)

# 4. Package
csa_path = await pipeline.package_csa(
    skill_name="cooperative_assembly",
    model=trained_model,
    roles=["leader", "follower"],
    coordination_type="handover",
)

# 5. Publish
csa_id = await pipeline.publish_to_registry(
    csa_path=csa_path,
    skill_name="cooperative_assembly",
)
```

### **Example 3: Federated Learning**

```python
from swarmbridge.adapters import FederatedLearningAdapter

adapter = FederatedLearningAdapter(
    service_url="http://localhost:8001",
    site_id="swarmbridge_site_1",
)

# Submit local update
await adapter.submit_local_update(
    csa_id="csa_123",
    skill_name="cooperative_assembly",
    privacy_mode="encrypted",
    epsilon=1.0,
    delta=1e-5,
)

# Check round status
round_status = await adapter.get_round_status(round_id=42)
print(f"Round {round_status.round_id}: {round_status.status}")

# Request merge
if round_status.status == "completed":
    merged_csa_id = await adapter.request_merge(
        skill_name="cooperative_assembly",
    )
    print(f"Merged CSA: {merged_csa_id}")
```

### **Example 4: Runtime Execution via Edge Platform**

```python
from swarmbridge.adapters import EdgePlatformRuntimeAdapter

runtime = EdgePlatformRuntimeAdapter(
    edge_api_url="http://jetson-orin.local:8001",
    registry_url="http://localhost:8000",
)

# Execute skill (fetches from registry if needed)
execution_id = await runtime.execute_skill(
    csa_id="csa_cooperative_assembly",
    robot_id="robot_1",
    task_parameters={
        "object_id": "cube_red",
        "target_location": {"x": 0.5, "y": 0.5, "z": 0.2},
    },
    fetch_from_registry=True,
)

# Monitor execution
import asyncio
while True:
    status = await runtime.get_execution_status(execution_id)

    if status["state"] == "completed":
        print("✓ Skill executed successfully")
        break
    elif status["state"] == "failed":
        print(f"✗ Execution failed: {status['error']}")
        break

    await asyncio.sleep(1.0)
```

### **Example 5: Using Shared Schemas**

```python
from swarmbridge.schemas import SharedRoleSchema, CoordinationPrimitives, CoordinationType

# Define roles
schema = SharedRoleSchema()
roles = schema.create_role_set(
    num_actors=3,
    coordination_type="formation",
)

# roles = [
#     RoleDefinition(role_id="actor_0", role_type=RoleType.LEADER, ...),
#     RoleDefinition(role_id="actor_1", role_type=RoleType.FOLLOWER, ...),
#     RoleDefinition(role_id="actor_2", role_type=RoleType.FOLLOWER, ...),
# ]

# Use in training
pipeline = SwarmBridgePipeline()
csa_id = await pipeline.run_complete_pipeline(
    skill_name="formation_flight",
    num_demonstrations=3,
    num_actors=3,
    roles=[role.role_id for role in roles],
    coordination_type="formation",
)

# Coordination primitive
primitives = CoordinationPrimitives()
formation = primitives.get_primitive(
    CoordinationType.FORMATION,
    roles=["actor_0", "actor_1", "actor_2"],
    parameters={
        "formation_type": "triangle",
        "spacing_m": 1.0,
    },
)

# Convert to SwarmBrain task graph
task_graph = primitives.to_swarmbrain_task_graph(formation)
```

---

## Migration Guide

### **For Existing Code Using OpenFL Directly**

**Before:**
```python
from openfl.component.aggregator import Aggregator

aggregator = Aggregator(...)
aggregator.run()
```

**After:**
```python
from swarmbridge.adapters import FederatedLearningAdapter

adapter = FederatedLearningAdapter(service_url="http://localhost:8001")
await adapter.submit_local_update(csa_id="csa_123", skill_name="skill")
```

### **For Existing Runtime Code**

**Before:**
```python
from swarm_skill_runtime import SkillExecutor

executor = SkillExecutor()
executor.execute_local(csa_path="...")
```

**After:**
```python
from swarmbridge.adapters import EdgePlatformRuntimeAdapter

runtime = EdgePlatformRuntimeAdapter(edge_api_url="http://jetson:8001")
await runtime.execute_skill(csa_id="csa_123", robot_id="robot_1")
```

### **For Role Definitions**

**Before:**
```python
# Different formats per system
csa_role = {"role_id": "leader", "capabilities": [...]}
moe_expert = {"expert_id": "leader", "specialization": "leader"}
```

**After:**
```python
from swarmbridge.schemas import SharedRoleSchema, RoleType, RoleDefinition

role = RoleDefinition(
    role_id="leader",
    role_type=RoleType.LEADER,
    capabilities=["grasp", "handover"],
)

# Convert to any format
csa_format = SharedRoleSchema.to_csa_format(role)
moe_format = SharedRoleSchema.to_moe_format(role)
swarmbrain_format = SharedRoleSchema.to_swarmbrain_format(role)
```

---

## Benefits of Refactoring

✅ **Clearer Responsibilities**: SwarmBridge focuses on capture & training
✅ **Modularity**: Easy to test and maintain individual components
✅ **Framework-Agnostic**: Federated service can use any FL framework
✅ **Thin Runtime**: Execution delegated to Edge Platform's optimized runtime
✅ **Shared Schemas**: Ensures compatibility across all three systems
✅ **Easier Integration**: Clear adapter interfaces for external systems
✅ **Unlearning Support**: Built-in via federated adapter
✅ **Production-Ready**: Well-structured, tested, documented

---

## Configuration

```yaml
# swarmbridge_config.yaml

pipeline:
  ros2_workspace: ros2_ws
  demonstration_topic: /swarm/demonstrations
  training_epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  csa_output_dir: artifacts

registry:
  url: http://localhost:8000

federated_service:
  url: http://localhost:8001
  site_id: swarmbridge_site_1

edge_platform:
  url: http://jetson-orin.local:8001
```

---

## Testing

```bash
# Test complete pipeline
pytest tests/swarmbridge/test_pipeline.py -v

# Test adapters
pytest tests/swarmbridge/test_adapters.py -v

# Test shared schemas
pytest tests/swarmbridge/test_schemas.py -v
```

---

## Architecture Comparison

| Component | Before | After |
|-----------|--------|-------|
| **Pipeline** | Scattered scripts | Unified `SwarmBridgePipeline` |
| **Federated Learning** | Direct OpenFL | Adapter to external service |
| **Runtime** | Local BT/MoveIt2 | Thin wrapper to Edge Platform |
| **Role Schema** | Per-system formats | Shared schema with converters |
| **Coordination** | Hardcoded | Primitives library |
| **Dependencies** | OpenFL, MoveIt2, BT | Minimal (adapters only) |
| **Testing** | Difficult | Easy (mocked services) |
| **Deployment** | Monolithic | Modular services |

---

## System Integration

```
SwarmBridge 2.0
    │
    ├──► Captures demonstrations (ROS 2)
    ├──► Trains cooperative policies (PyTorch)
    ├──► Packages CSAs (artifact format)
    ├──► Publishes to registry
    │
    └──► Delegates:
         │
         ├──► Federated Learning → Federated Service
         ├──► Runtime Execution → Edge Platform (Dynamical API)
         └──► Mission Orchestration → SwarmBrain
```

---

## Next Steps

1. **Deploy Federated Service**: Set up external federated learning service
2. **Configure Edge Platform**: Ensure Dynamical API is accessible
3. **Migrate Existing Code**: Update to use adapters instead of direct calls
4. **Test Integration**: End-to-end testing with all three systems
5. **Monitor Performance**: Ensure pipeline throughput meets requirements

---

## References

- **Edge Platform Integration**: `docs/EDGE_PLATFORM_INTEGRATION.md`
- **Tri-System Integration**: `docs/TRI_SYSTEM_INTEGRATION.md`
- **SwarmBrain Integration**: `docs/TRI_SYSTEM_INTEGRATION.md#swarmbrain`

---

## Status

✅ **Refactoring Complete** (SwarmBridge 2.0)

**Focus**: Capture, Train, Package
**Delegation**: Runtime, Federated Orchestration, Mission Planning
**Benefits**: Modular, Maintainable, Production-Ready
