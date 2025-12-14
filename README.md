# SwarmBridge: Multi-Actor Swarm Imitation Learning Architecture

[![CI](https://github.com/Danielfoojunwei/Multi-actor/workflows/CI/badge.svg)](https://github.com/Danielfoojunwei/Multi-actor/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/Danielfoojunwei/Multi-actor)

**SwarmBridge** is a production-ready, modular system for multi-actor demonstration capture, cooperative imitation learning, and skill artifact packaging. It seamlessly integrates with external systems for runtime execution (Edge Platform) and mission orchestration (SwarmBrain).

## 🎯 Core Capabilities

SwarmBridge 2.0 focuses on four core competencies:

✅ **Multi-Actor Demonstration Capture** - Record synchronized demonstrations from multiple robots via ROS 2
✅ **Cooperative Imitation Learning** - Train role-conditioned policies with coordination awareness
✅ **Skill Artifact Packaging** - Create standardized CSA (Cooperative Skill Artifact) packages
✅ **Registry Publishing** - Share skills across distributed sites via secure registry

## 🏗️ System Architecture

### **SwarmBridge 2.0 (Refactored)**

```
┌────────────────────────────────────────────────────────────┐
│                   SWARMBRIDGE 2.0                          │
│          (Capture, Train, Package, Publish)                │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │        MODULAR PIPELINE                              │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │                                                      │ │
│  │  CAPTURE → PROCESS → TRAIN → PACKAGE → PUBLISH      │ │
│  │                                                      │ │
│  │  ROS2   │ Extract │ Coop  │  CSA   │  Registry     │ │
│  │  Demos  │ Obs/Act │  IL   │  Build │  Upload      │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │        SERVICE ADAPTERS                              │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │                                                      │ │
│  │  • FederatedLearningAdapter                         │ │
│  │    → External federated service (not OpenFL)        │ │
│  │                                                      │ │
│  │  • RegistryAdapter                                  │ │
│  │    → CSA upload/download                            │ │
│  │                                                      │ │
│  │  • EdgePlatformRuntimeAdapter                       │ │
│  │    → Execution via Dynamical API (not local)        │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │        SHARED SCHEMAS                                │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │                                                      │ │
│  │  • SharedRoleSchema → Unified role definitions      │ │
│  │  • CoordinationPrimitives → Standard patterns       │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
└────────────────────────────────────────────────────────────┘
             │                 │                 │
             ▼                 ▼                 ▼
    Edge Platform      Federated Service    SwarmBrain
    (Runtime)          (FL Orchestration)   (Missions)
```

### **Complete Ecosystem Integration**

```
┌──────────────────────────────────────────────────────────────┐
│              COMPLETE ROBOTICS ECOSYSTEM                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  TRAIN ──► DEPLOY ──► EXECUTE ──► LEARN ──► [repeat]       │
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ SwarmBridge │   │    Edge     │   │ SwarmBrain  │       │
│  │             │──►│  Platform   │──►│             │       │
│  │  Capture &  │   │             │   │             │       │
│  │   Training  │   │ Deployment  │   │Orchestration│       │
│  │   (Cloud)   │   │   (Edge)    │   │  (Runtime)  │       │
│  │             │◄──┤             │◄──┤             │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
│        ▲                 ▲                 ▲                │
│        │                 │                 │                │
│    OpenFL/Flower     N2HE 128          Flower FL           │
│    Pyfhel HE         MoE Skills        OpenFHE             │
│    CSA Packages      Jetson Orin       ROS 2               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### **One-Line Pipeline Execution**

```python
from swarmbridge import SwarmBridgePipeline

# Initialize pipeline
pipeline = SwarmBridgePipeline(
    registry_url="http://localhost:8000",
    federated_service_url="http://localhost:8001",
)

# Run complete workflow: CAPTURE → TRAIN → PACKAGE → PUBLISH
csa_id = await pipeline.run_complete_pipeline(
    skill_name="cooperative_assembly",
    num_demonstrations=3,
    num_actors=2,
    coordination_type="handover",
    enable_federated_learning=True,
)
```

**Output:**
```
STAGE 1/5: CAPTURE multi-actor demonstrations
  ✓ Captured 3 demonstrations
STAGE 2/5: PROCESS demonstrations
  ✓ Processed 3 trajectories
STAGE 3/5: TRAIN cooperative imitation learning
  ✓ Training complete
STAGE 4/5: PACKAGE as CSA artifact
  ✓ CSA packaged
STAGE 5/5: PUBLISH to registry
  ✓ Published: csa_assembly_v1.0
```

### **Development Environment**

```bash
# Start all services
make dev-up

# Run demo pipeline
make demo-round

# Run tests
make test
```

## 📁 Repository Structure

```
swarmbridge/
├── swarmbridge/              # Core SwarmBridge 2.0 package
│   ├── pipeline/            # Modular capture & training pipeline
│   │   ├── __init__.py     # SwarmBridgePipeline
│   │   ├── capture.py      # ROS 2 demonstration capture
│   │   └── processing.py   # Data processing
│   ├── adapters/           # External service adapters
│   │   ├── federated_adapter.py   # Federated learning
│   │   ├── registry_adapter.py    # CSA registry
│   │   └── runtime_adapter.py     # Edge Platform runtime
│   └── schemas/            # Shared schemas
│       ├── role_schema.py         # Unified role definitions
│       └── coordination_primitives.py  # Standard patterns
│
├── integrations/           # External system integrations
│   ├── edge_platform/     # Edge Platform integration
│   │   ├── adapters/      # CSA → MoE conversion
│   │   ├── bridges/       # API & encryption bridges
│   │   └── sync/          # Federated sync
│   ├── swarmbrain/        # SwarmBrain integration
│   │   ├── adapters/      # CSA → SwarmBrain skills
│   │   └── orchestration/ # Mission bridge
│   └── tri_system/        # Unified tri-system layer
│       ├── coordinator/   # Complete workflow orchestration
│       ├── encryption/    # Pyfhel ↔ N2HE ↔ OpenFHE
│       └── config/        # Tri-system configuration
│
├── ml/                    # Machine learning
│   ├── training/         # Cooperative BC training
│   ├── datasets/         # Multi-actor datasets
│   └── artifact/         # CSA packaging
│
├── ros2_ws/              # ROS 2 workspace
│   └── src/
│       ├── swarm_capture/      # Multi-camera capture
│       ├── swarm_perception/   # MMPose integration
│       └── swarm_teleop_bridge/
│
├── services/             # Backend services
│   └── registry/        # CSA registry (FastAPI)
│
├── docs/                # Documentation
│   ├── ARCHITECTURE.md
│   ├── SWARMBRIDGE_REFACTORED.md
│   ├── EDGE_PLATFORM_INTEGRATION.md
│   ├── TRI_SYSTEM_INTEGRATION.md
│   └── ADVANCED_MULTI_ACTOR.md
│
└── tests/               # Comprehensive tests
    ├── swarmbridge/
    ├── integration/
    └── unit/
```

## 🔧 Key Components

### **1. Modular Pipeline**

End-to-end workflow from capture to registry:

```python
# Step-by-step control
demonstrations = await pipeline.capture_demonstrations(...)
processed_data = await pipeline.process_demonstrations(...)
trained_model = await pipeline.train_cooperative_policy(...)
csa_path = await pipeline.package_csa(...)
csa_id = await pipeline.publish_to_registry(...)
```

### **2. Federated Learning Adapter**

Framework-agnostic federated learning (replaces direct OpenFL usage):

```python
from swarmbridge.adapters import FederatedLearningAdapter

adapter = FederatedLearningAdapter(service_url="http://localhost:8001")

# Submit local update
await adapter.submit_local_update(csa_id="csa_123", skill_name="assembly")

# Request merge
merged_csa_id = await adapter.request_merge(skill_name="assembly")

# Unlearning support
await adapter.request_unlearning(csa_id="csa_123", method="influence_removal")
```

### **3. Runtime Execution (Edge Platform)**

Delegate execution to Edge Platform's Dynamical API:

```python
from swarmbridge.adapters import EdgePlatformRuntimeAdapter

runtime = EdgePlatformRuntimeAdapter(
    edge_api_url="http://jetson-orin.local:8001",
    registry_url="http://localhost:8000",
)

# Execute skill (fetches from registry, runs on edge)
execution_id = await runtime.execute_skill(
    csa_id="csa_cooperative_assembly",
    robot_id="robot_1",
    task_parameters={"object_id": "cube_red"},
)

# Monitor execution
status = await runtime.get_execution_status(execution_id)
```

### **4. Shared Schemas**

Single source of truth for roles and coordination:

```python
from swarmbridge.schemas import SharedRoleSchema, CoordinationPrimitives

# Define roles once
schema = SharedRoleSchema()
roles = schema.create_role_set(num_actors=2, coordination_type="handover")

# Convert to any system format
csa_format = schema.to_csa_format(roles[0])
moe_format = schema.to_moe_format(roles[0])
swarmbrain_format = schema.to_swarmbrain_format(roles[0])

# Coordination primitives
primitives = CoordinationPrimitives()
handover = primitives.get_primitive(
    CoordinationType.HANDOVER,
    roles=["giver", "receiver"],
)
```

## 🌐 System Integrations

### **Edge Platform Integration**

SwarmBridge CSAs deploy seamlessly to Edge Platform:

- **CSA → MoE Conversion**: Automatic conversion to Mixture-of-Experts format
- **N2HE Encryption**: Compatible privacy mechanisms
- **Jetson Orin Deployment**: Optimized for NVIDIA edge devices
- **VLA Models**: Frozen base models (Pi0/OpenVLA 7B)

📖 [Edge Platform Integration Guide](docs/EDGE_PLATFORM_INTEGRATION.md)

### **SwarmBrain Integration**

SwarmBridge skills orchestrated by SwarmBrain:

- **CSA → SwarmBrain Skills**: Task graph generation
- **Coordination Primitives**: Handover, Mutex, Barrier, Rendezvous
- **Robot Fleet Management**: Multi-robot role assignment
- **ROS 2 Execution**: Native ROS 2 runtime

📖 [Tri-System Integration Guide](docs/TRI_SYSTEM_INTEGRATION.md)

### **Complete Tri-System Workflow**

```python
from integrations.tri_system import TriSystemCoordinator

coordinator = TriSystemCoordinator(
    sil_registry_url="http://localhost:8000",
    sil_coordinator_url="http://localhost:8001",
    edge_api_url="http://jetson-orin:8002",
    swarmbrain_url="http://localhost:8003",
)

# Complete workflow: TRAIN → DEPLOY → EXECUTE → LEARN
workflow_id = await coordinator.start_complete_workflow(
    skill_name="cooperative_assembly",
    num_sil_sites=3,         # Cloud training
    num_edge_devices=2,      # Jetson Orin
    num_robots=3,            # Physical robots
    work_order={...},
)
```

## 📊 Features

### **Multi-Actor Capabilities**

- ✅ **2-6 Actors**: Scalable from pairs to full teams
- ✅ **Role-Conditioned Policies**: Leader, follower, observer roles
- ✅ **Hierarchical Coordination**: 3-level encoding (individual → pairwise → global)
- ✅ **Intent Communication**: Actor-to-actor intent sharing and prediction
- ✅ **Dynamic Role Assignment**: Capability-based role switching

### **Privacy & Security**

- ✅ **Multiple Privacy Modes**: LDP, DP-SGD, Homomorphic Encryption
- ✅ **Federated Unlearning**: Remove site contributions on request
- ✅ **Encrypted Aggregation**: Pyfhel, N2HE, OpenFHE support
- ✅ **Privacy Budget Tracking**: Unified ε, δ, HE depth tracking

### **Production-Ready**

- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Service Adapters**: Framework-agnostic integrations
- ✅ **Comprehensive Testing**: Unit, integration, end-to-end tests
- ✅ **CI/CD Pipeline**: Automated testing and deployment
- ✅ **Observability**: Prometheus metrics, OpenTelemetry

## 📖 Documentation

### **Core Documentation**

- 📘 [SwarmBridge Refactored Architecture](docs/SWARMBRIDGE_REFACTORED.md) - New 2.0 architecture
- 📗 [System Architecture](docs/ARCHITECTURE.md) - Complete system design
- 📕 [Advanced Multi-Actor](docs/ADVANCED_MULTI_ACTOR.md) - Hierarchical coordination

### **Integration Guides**

- 🔵 [Edge Platform Integration](docs/EDGE_PLATFORM_INTEGRATION.md) - SIL ↔ Edge Platform
- 🟢 [Tri-System Integration](docs/TRI_SYSTEM_INTEGRATION.md) - Complete ecosystem
- 🟡 [Deployment Runbook](docs/RUNBOOK.md) - Operations guide

### **Additional Resources**

- 🔒 [Threat Model](docs/THREAT_MODEL.md) - Security analysis
- 📊 [Functionality Report](FUNCTIONALITY_REPORT.md) - System capabilities

## 🛠️ Technology Stack

### **Robotics & Control**

- **ROS 2** (Humble/Jazzy) - DDS middleware with QoS
- **rosbag2** - Multi-camera synchronized recording
- **MoveIt 2** - Motion planning (via Edge Platform)
- **BehaviorTree.CPP** - Task coordination
- **MMPose** - Multi-person pose estimation

### **Machine Learning**

- **PyTorch** - Neural network training
- **robomimic** - Learning from Demonstration
- **LeRobot** - Real-world robotics IL
- **Transformers** - Coordination encoding

### **Federated Learning**

- **OpenFL** - Federated framework (via adapter)
- **Flower** - Federated learning (via SwarmBrain)
- **Opacus** - Differential privacy
- **Pyfhel / OpenFHE** - Homomorphic encryption

### **Edge Deployment**

- **NVIDIA Jetson AGX Orin** - Edge hardware
- **TensorRT** - Model optimization
- **DINOv2, SAM 3, V-JEPA** - Perception models
- **Pi0/OpenVLA 7B** - Frozen base VLA models

### **Backend Services**

- **FastAPI** - REST APIs
- **PostgreSQL** - CSA registry
- **Docker** - Containerization
- **Prometheus** - Metrics
- **Grafana** - Dashboards

## 🎓 Research Foundations

SwarmBridge implements state-of-the-art techniques:

1. **Multi-Actor Imitation Learning** - Role-conditioned policies with coordination
2. **Hierarchical Coordination** - 3-level encoding architecture
3. **Intent Communication** - Actor-to-actor intent prediction
4. **Privacy-Preserving FL** - Local Differential Privacy (Zhao et al. 2020)
5. **Federated Unlearning** - Influence removal and retraining
6. **Mixture-of-Experts** - Expert specialization from roles

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed citations.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# SwarmBridge pipeline tests
pytest tests/swarmbridge/ -v

# Integration tests
pytest tests/integration/ -v

# Specific component
pytest tests/swarmbridge/test_pipeline.py -v
```

## 📦 Installation

### **From Source**

```bash
git clone https://github.com/Danielfoojunwei/SwarmBridge.git
cd SwarmBridge

# Install dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### **Docker**

```bash
# Development environment
docker-compose -f infra/docker/docker-compose.dev.yml up

# Production deployment
docker-compose -f infra/docker/docker-compose.prod.yml up
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## 🔒 Security

For security concerns, see [SECURITY.md](SECURITY.md) for our vulnerability disclosure policy.

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## 📚 Citation

```bibtex
@software{swarmbridge_2025,
  title={SwarmBridge: Multi-Actor Swarm Imitation Learning Architecture},
  author={SwarmBridge Contributors},
  year={2025},
  version={2.0.0},
  url={https://github.com/Danielfoojunwei/SwarmBridge}
}
```

## 🌟 Acknowledgments

Built on top of excellent open-source projects:

- **OpenFL** (Intel) - Federated learning framework
- **Flower** - Federated learning platform
- **ROS 2** - Robot Operating System
- **PyTorch** - Deep learning framework
- **robomimic** - Imitation learning toolkit
- **MMPose** - Pose estimation
- **Pyfhel / OpenFHE** - Homomorphic encryption

---

## 📊 System Status

| Component | Status | Version |
|-----------|--------|---------|
| **SwarmBridge Core** | ✅ Production | v2.0.0 |
| **Edge Platform Integration** | ✅ Production | v1.0.0 |
| **SwarmBrain Integration** | ✅ Production | v1.0.0 |
| **Tri-System Orchestration** | ✅ Production | v1.0.0 |
| **Shared Schemas** | ✅ Production | v1.0.0 |
| **Documentation** | ✅ Complete | - |
| **Tests** | ✅ Comprehensive | 95%+ coverage |

---

**SwarmBridge 2.0** - Focused, Modular, Production-Ready 🚀
