# Dynamical-SIL Architecture

## Overview

Dynamical-SIL is a privacy-preserving multi-actor swarm imitation learning system that enables cooperative humanoid skills to be learned from limited demonstrations (2-3) and shared securely across distributed sites.

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Dynamical-SIL System                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Site A    │  │   Site B    │  │   Site C    │
│             │  │             │  │             │
│ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │
│ │ Capture │ │  │ │ Capture │ │  │ │ Capture │ │
│ │ ONVIF   │ │  │ │ ONVIF   │ │  │ │ ONVIF   │ │
│ │ RTSP    │ │  │ │ RTSP    │ │  │ │ RTSP    │ │
│ │ ROS 2   │ │  │ │ ROS 2   │ │  │ │ ROS 2   │ │
│ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │
│      ↓      │  │      ↓      │  │      ↓      │
│ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │
│ │ Percept │ │  │ │ Percept │ │  │ │ Percept │ │
│ │ MMPose  │ │  │ │ MMPose  │ │  │ │ MMPose  │ │
│ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │
│      ↓      │  │      ↓      │  │      ↓      │
│ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │
│ │Training │ │  │ │Training │ │  │ │Training │ │
│ │robomimic│ │  │ │robomimic│ │  │ │robomimic│ │
│ │LeRobot  │ │  │ │LeRobot  │ │  │ │LeRobot  │ │
│ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │
│      ↓      │  │      ↓      │  │      ↓      │
│  CSA Delta  │  │  CSA Delta  │  │  CSA Delta  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ↓
              ┌──────────────────┐
              │ OpenFL Swarm     │
              │ Coordinator      │
              │                  │
              │ Privacy Modes:   │
              │  • LDP           │
              │  • DP-SGD        │
              │  • HE/FHE        │
              │                  │
              │ Robust Agg:      │
              │  • Trimmed Mean  │
              │  • Krum          │
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │  CSA Registry    │
              │  (FastAPI + PG)  │
              │                  │
              │ • Versioning     │
              │ • Signing        │
              │ • Rollback       │
              │ • Provenance     │
              └────────┬─────────┘
                       ↓
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Runtime A   │ │ Runtime B   │ │ Runtime C   │
│             │ │             │ │             │
│ BT Executor │ │ BT Executor │ │ BT Executor │
│ MoveIt2     │ │ MoveIt2     │ │ MoveIt2     │
│ Safety Mon. │ │ Safety Mon. │ │ Safety Mon. │
└─────────────┘ └─────────────┘ └─────────────┘
```

## Component Details

### 1. Multi-Actor Capture Pipeline

**Purpose**: Synchronize multi-camera streams and record cooperative demonstrations.

**Technologies**:
- **ONVIF** (python-onvif-zeep): Camera discovery and control
- **RTSP** (GStreamer): Stream ingest
- **ROS 2** (rosbag2): Recording and playback
- **Time Sync**: chrony/PTP for sub-millisecond synchronization

**Data Flow**:
1. Discover ONVIF cameras on network
2. Configure synchronized RTSP streams
3. Ingest via GStreamer pipelines
4. Publish to ROS 2 topics with hardware timestamps
5. Record to rosbag2 with QoS guarantees

**References**:
- [ROS 2 QoS Policies](https://docs.ros.org/en/rolling/Concepts/About-Quality-of-Service-Settings.html)
- [rosbag2 Documentation](https://github.com/ros2/rosbag2)

### 2. Cooperative Skill Representation (CSA)

**Purpose**: Package cooperative skills as portable, versioned, testable artifacts.

**CSA Structure**:
```
csa_v1.0.0.tar.gz
├── manifest.json              # Metadata
├── roles/
│   ├── leader_adapter.pt      # Role-specific policy adapters
│   ├── follower_adapter.pt
│   └── spotter_adapter.pt
├── coordination_encoder.pt     # Shared coordination latent encoder
├── phase_machine.xml          # BehaviorTree.CPP state machine
├── safety_envelope.json       # Constraints (velocity, force, workspace)
├── tests/
│   └── test_suite.json        # Deterministic offline tests
└── checksums.sha256           # Integrity verification
```

**Key Features**:
- **Role-Conditioned Policies**: Small adapters (LoRA, IA3, linear) instead of full foundation weights
- **Coordination Latent**: Shared z_t from transformer/RNN encoder for synchronization
- **Phase Machine**: BehaviorTree.CPP for task logic (approach → grasp → lift → transfer → place → retreat)
- **Safety Envelope**: Runtime constraints (max velocity, force, workspace bounds, min separation distance)
- **Immutable + Versioned**: Semantic versioning with cryptographic signing

**Implementation**: See `ml/artifact/schema.py`

### 3. Local Imitation Learning

**Purpose**: Train role-conditioned cooperative policies from 2-3 demonstrations.

**Architecture**:
```
Input: Multi-actor observations [B, N, T, D]
       N = number of actors
       T = sequence length
       D = observation dim

CoordinationEncoder(obs) → z_t [B, coord_dim]
   ↓
SharedEncoder(obs_i, z_t) → shared_features [B, hidden_dim]
   ↓
RoleAdapter_i(shared_features) → action_i [B, action_dim]
```

**Training**:
- **Loss**: BC loss + coordination consistency + action regularization
- **Optimizer**: AdamW with cosine annealing
- **Data**: robomimic HDF5 or LeRobot dataset format
- **Augmentation**: Temporal jittering, noise injection

**Technologies**:
- [robomimic](https://robomimic.github.io/) - LfD framework
- [LeRobot](https://github.com/huggingface/lerobot) - Real-world robotics IL toolkit

**Implementation**: See `ml/training/train_cooperative_bc.py`

### 4. Privacy-Preserving Swarm Learning

**Purpose**: Share skill artifacts privately across sites without exposing raw demonstrations.

#### Privacy Modes

**a) Local Differential Privacy (LDP)**

Based on: Zhao et al. (2020) "Local Differential Privacy-based Federated Learning for Internet of Things"

- **Mechanism**: Add calibrated Laplace/Gaussian noise at each site before sending updates
- **Guarantee**: ε-LDP (edge-first privacy)
- **Formula**: `noise ~ Laplace(0, Δf/ε)` where Δf is L1 sensitivity
- **Use Case**: IoT/edge deployment where coordinator is untrusted

**Implementation**: `swarm/privacy/ldp.py`

**b) Differential Privacy SGD (DP-SGD)**

Based on: Opacus library (PyTorch DP training)

- **Mechanism**: Per-sample gradient clipping + Gaussian noise
- **Guarantee**: (ε, δ)-DP with RDP accounting
- **Formula**: Clip gradients to C, add noise ~ N(0, σ²C²I)
- **Use Case**: Trusted coordinator, formal privacy guarantees

**Implementation**: `swarm/privacy/dp_sgd.py`

**c) Homomorphic Encryption (HE/FHE)**

Based on: NTU DR FHE cloud-edge architecture, Pyfhel/OpenFHE

- **Mechanism**: Encrypt CSA adapters, aggregate in encrypted space, decrypt only at coordinator
- **Guarantee**: Computational security (ciphertext indistinguishability)
- **Schemes**: BFV (integers), CKKS (approximate reals)
- **Use Case**: Validate-only mode (encrypted scoring before acceptance)

**Implementation**: `swarm/privacy/he_wrapper.py`

#### Robust Aggregation

**Purpose**: Defend against poisoning attacks from malicious sites.

**Strategies**:
1. **Trimmed Mean**: Remove top/bottom k% outliers, average remaining
2. **Median**: Element-wise median (most robust, slow)
3. **Krum**: Select update with smallest sum of distances to m nearest neighbors
4. **Coordinate-wise Median**: Median per parameter (faster than full median)

**Implementation**: `swarm/openfl/aggregator.py`

**References**:
- [OpenFL Documentation](https://openfl.readthedocs.io/)
- Blanchard et al. (2017) "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"

### 5. Federated Unlearning

**Purpose**: Remove a site's contribution from CSA (GDPR "right to be forgotten").

**Approach**:
1. **Provenance Tracking**: Record which sites contributed to each round
2. **Unlearning Methods**:
   - **Retraining**: Re-aggregate all rounds excluding target site (exact)
   - **Influence Removal**: Gradient-based approximate removal (fast)
3. **Certification**: Verify removal through weight change + test suite
4. **Versioning**: Produce new CSA version with provenance update

**Implementation**: `swarm/unlearning/unlearner.py`

**References**:
- DTC Federated Unlearning publications
- Bourtoule et al. (2021) "Machine Unlearning"

### 6. CSA Registry Service

**Purpose**: Centralized versioning, deployment tracking, and rollback.

**API Endpoints**:
- `POST /api/v1/csa/upload` - Upload new CSA
- `GET /api/v1/csa/list` - List CSAs (filterable)
- `GET /api/v1/csa/{id}/download` - Download CSA artifact
- `POST /api/v1/deployment/deploy` - Deploy CSA to site
- `POST /api/v1/deployment/rollback` - Rollback to previous version
- `GET /api/v1/deployment/history/{site}` - Deployment history

**Database Schema**:
- `csa_artifacts`: Metadata, file path, signature verification, privacy mode
- `deployments`: Site ID, CSA ID, timestamp, status (deployed/rolled_back/failed)

**Technologies**:
- FastAPI (async Python web framework)
- PostgreSQL (relational database)
- SQLAlchemy ORM

**Implementation**: `services/registry/main.py`

### 7. Runtime Execution

**Purpose**: Deploy and execute cooperative skills with safety monitoring.

**Components**:
- **BehaviorTree.CPP**: State machine execution (phase transitions)
- **MoveIt2**: Motion planning and collision avoidance
- **Safety Monitor**: Real-time constraint checking (velocity, force, separation distance)
- **Abort Logic**: Emergency stop on safety violations

**Execution Flow**:
1. Load CSA from registry
2. Initialize role-conditioned policies
3. Start BehaviorTree phase machine
4. Loop:
   - Observe multi-actor states
   - Encode coordination latent z_t
   - Compute role-conditioned actions
   - Check safety envelope
   - Execute actions via MoveIt2
   - Monitor for abort conditions

**Technologies**:
- [BehaviorTree.CPP](https://www.behaviortree.dev/) - Behavior trees
- [Groot](https://www.behaviortree.dev/groot/) - Visual BT editor
- [MoveIt 2](https://moveit.ros.org/) - Manipulation planning

**Implementation**: `ros2_ws/src/swarm_skill_runtime/`

## Data Flow

### Training Flow

```
Demonstrations (rosbag2)
    ↓
Extract multi-actor trajectories
    ↓
Dataset (robomimic HDF5 / LeRobot)
    ↓
Train CoordinationEncoder + RoleConditionedPolicy
    ↓
Package CSA (adapters + encoder + BT + safety + tests)
    ↓
Sign CSA
    ↓
Upload to Registry
```

### Swarm Flow

```
Site A: Train local CSA_A
Site B: Train local CSA_B
Site C: Train local CSA_C
    ↓
Apply privacy (LDP/DP-SGD/HE)
    ↓
Send to Coordinator
    ↓
Robust aggregation (trimmed mean/Krum)
    ↓
Merged CSA_merged
    ↓
Upload to Registry (new version)
    ↓
Sites pull CSA_merged
    ↓
Deploy to runtime
```

### Unlearning Flow

```
GDPR request: Remove Site B
    ↓
Identify rounds where Site B participated
    ↓
Unlearn (retraining or influence removal)
    ↓
Generate new CSA_unlearned
    ↓
Certify removal (weight change + tests)
    ↓
Upload to Registry (new version)
    ↓
Update provenance (remove Site B)
    ↓
Deploy CSA_unlearned
```

## Technology Stack Summary

### Robotics Runtime
- **ROS 2 Humble/Jazzy**: DDS middleware, QoS policies
- **rosbag2**: Recording/replay
- **MoveIt 2**: Motion planning
- **BehaviorTree.CPP**: Task coordination
- **Groot**: BT visual editor

### Sensing & Perception
- **python-onvif-zeep**: Camera control
- **GStreamer RTSP**: Stream ingest
- **MMPose**: Pose estimation
- **MMDetection**: Object detection (optional)
- **CVAT**: Annotation

### Imitation Learning
- **robomimic**: LfD framework
- **LeRobot**: Robotics IL toolkit
- **PyTorch**: Deep learning
- **Gymnasium**: RL environments

### Federated Learning
- **OpenFL**: FL framework (primary)
- **FATE**: Secure FL (HE/MPC, secondary)
- **Opacus**: PyTorch DP training
- **CrypTen**: MPC backend
- **Pyfhel** / **OpenFHE**: Homomorphic encryption

### Backend Services
- **FastAPI**: REST API
- **PostgreSQL**: Database
- **SQLAlchemy**: ORM

### Observability
- **OpenTelemetry**: Distributed tracing
- **Prometheus**: Metrics
- **Grafana**: Dashboards

### DevOps
- **Docker** / **Docker Compose**: Containerization
- **Kubernetes** (optional): Orchestration
- **GitHub Actions**: CI/CD

## Security & Privacy Summary

### Threat Model

See [THREAT_MODEL.md](THREAT_MODEL.md) for detailed analysis.

**Adversaries**:
1. **Honest-but-curious coordinator**: Wants to infer raw demonstrations
2. **Malicious sites**: Send poisoned updates to degrade model
3. **Network eavesdropper**: Intercept CSA transfers
4. **Insider**: Access to registry, attempt to tamper with CSAs

**Mitigations**:
1. **Privacy modes** (LDP/DP-SGD/HE) protect against coordinator
2. **Robust aggregation** (trimmed mean/Krum) defend against poisoning
3. **TLS** for network transport
4. **Cryptographic signing** + **checksums** for CSA integrity
5. **RBAC** + **audit logs** for registry access

### Privacy Accounting

- **LDP**: ε-LDP per site (additive across rounds)
- **DP-SGD**: (ε, δ)-DP with RDP accountant
- **HE**: Computational security (no degradation)

## Scalability & Performance

### Bottlenecks
1. **HE aggregation**: Slow for large models (use sparse updates or HE validation-only)
2. **Coordination encoder**: Sequence length scales O(T²) for transformers
3. **Robust aggregation**: Median is O(nk log k) where n=sites, k=params

### Optimizations
1. **Model compression**: Use small adapters (LoRA rank 8-16)
2. **Sparse updates**: Only send changed parameters
3. **Async rounds**: Don't wait for stragglers (staleness tolerance)
4. **Hierarchical aggregation**: Aggregate in clusters, then global

## Future Extensions

1. **Jetson Orin target**: CUDA/TensorRT optimized runtime
2. **Factory network constraints**: NAT traversal, bandwidth adaptation
3. **Foundation model integration**: Pre-trained vision-language-action models
4. **Hardware-in-the-loop simulation**: Isaac Sim / Gazebo integration
5. **Multi-modal demonstrations**: Vision, tactile, audio
