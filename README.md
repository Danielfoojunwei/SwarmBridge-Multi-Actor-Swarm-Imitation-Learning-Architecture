# Dynamical-SIL: Privacy-Preserving Multi-Actor Swarm Imitation Learning

[![CI](https://github.com/Danielfoojunwei/Multi-actor/workflows/CI/badge.svg)](https://github.com/Danielfoojunwei/Multi-actor/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Dynamical-SIL** is a production-grade, privacy-preserving multi-actor swarm imitation learning system that enables cooperative humanoid skills to be learned from 2–3 human demonstrations and shared privately across distributed sites with reliability guarantees.

## Key Features

- 🤖 **Multi-Actor Imitation Learning**: Role-conditioned policies with coordination state machines
- 🔒 **Privacy-Preserving Collaboration**: LDP-FL, DP-SGD, and FHE-enabled secure compute
- 🌐 **Federated Swarm Learning**: OpenFL-based private skill artifact sharing
- 🔄 **Federated Unlearning**: First-class "remove site contribution" capability
- 🏭 **Production-Ready**: Reproducible builds, CI/CD, observability, security hardening

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Dynamical-SIL System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Site A     │  │   Site B     │  │   Site C     │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ ONVIF Cameras│  │ ONVIF Cameras│  │ ONVIF Cameras│          │
│  │ RTSP Ingest  │  │ RTSP Ingest  │  │ RTSP Ingest  │          │
│  │ ROS 2 Capture│  │ ROS 2 Capture│  │ ROS 2 Capture│          │
│  │ MMPose       │  │ MMPose       │  │ MMPose       │          │
│  │ robomimic    │  │ robomimic    │  │ robomimic    │          │
│  │ Local Train  │  │ Local Train  │  │ Local Train  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │  OpenFL Swarm   │                            │
│                  │  Coordinator    │                            │
│                  │  - LDP Mode     │                            │
│                  │  - DP-SGD Mode  │                            │
│                  │  - HE/FHE Mode  │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │ CSA Registry    │                            │
│                  │ (FastAPI+PG)    │                            │
│                  │ - Versioning    │                            │
│                  │ - Signing       │                            │
│                  │ - Rollback      │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         │                 │                 │                  │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐         │
│  │ Runtime A    │  │ Runtime B    │  │ Runtime C    │         │
│  │ BT Execution │  │ BT Execution │  │ BT Execution │         │
│  │ MoveIt2      │  │ MoveIt2      │  │ MoveIt2      │         │
│  │ Safety Mon.  │  │ Safety Mon.  │  │ Safety Mon.  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU (optional, for training)
- ROS 2 Humble/Jazzy (for native builds)

### One-Command Development Environment

```bash
make dev-up
```

This brings up:
- CSA Registry service (port 8080)
- OpenFL Coordinator (port 8081)
- PostgreSQL database
- Prometheus metrics
- Sample ROS 2 graph

### Run Complete Demo Pipeline

```bash
make demo-round
```

This executes:
1. Replay sample rosbag2 capture
2. Train cooperative BC policy
3. Package CSA artifact
4. Run OpenFL swarm merge (2 simulated sites)
5. Deploy merged CSA to runtime

## Repository Structure

```
dynamical-sil/
├── docs/               # Architecture, threat model, runbooks
├── infra/              # Docker, K8s, Helm, Terraform
├── ros2_ws/            # ROS 2 workspace
│   └── src/
│       ├── swarm_capture/      # ONVIF + RTSP + rosbag2
│       ├── swarm_perception/   # MMPose integration
│       ├── swarm_skill_runtime/# BT execution + MoveIt2
│       └── swarm_teleop_bridge/# Teleop adapters
├── ml/                 # ML training & evaluation
│   ├── datasets/       # Schema + converters
│   ├── training/       # robomimic/LeRobot trainers
│   ├── evaluation/     # Offline metrics
│   └── artifact/       # CSA packaging
├── swarm/              # Federated learning infrastructure
│   ├── openfl/         # OpenFL workspace
│   ├── privacy/        # LDP, DP-SGD, HE/FHE
│   └── unlearning/     # Federated unlearning
├── services/           # Backend services
│   ├── registry/       # CSA registry (FastAPI)
│   └── telemetry/      # OpenTelemetry + Prometheus
├── ci/                 # GitHub Actions
└── tests/              # Unit + integration tests
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and component mapping
- [Threat Model](docs/THREAT_MODEL.md) - Adversary models and privacy modes
- [Deployment Runbook](docs/RUNBOOK.md) - Operations, incident response, rollback
- [API Reference](docs/API.md) - Service APIs and ROS 2 interfaces
- [Privacy Modes](docs/PRIVACY.md) - LDP, DP-SGD, HE/FHE configurations

## Technology Stack

### Robotics Runtime
- [ROS 2](https://docs.ros.org/en/rolling/) - DDS-based middleware with QoS
- [rosbag2](https://github.com/ros2/rosbag2) - Recording and replay
- [MoveIt 2](https://moveit.ros.org/) - Manipulation planning
- [BehaviorTree.CPP](https://www.behaviortree.dev/) - Task coordination
- [Groot](https://www.behaviortree.dev/groot/) - BT visual tooling

### Multi-Actor Sensing
- [python-onvif-zeep](https://github.com/FalkTannhaeuser/python-onvif-zeep) - Camera discovery/control
- [GStreamer RTSP](https://gstreamer.freedesktop.org/) - Stream ingest
- [MMPose](https://github.com/open-mmlab/mmpose) - Pose estimation
- [CVAT](https://github.com/opencv/cvat) - Annotation tooling

### Imitation Learning
- [robomimic](https://robomimic.github.io/) - LfD framework
- [LeRobot](https://github.com/huggingface/lerobot) - Real-world robotics IL toolkit

### Federated Learning & Privacy
- [OpenFL](https://github.com/securefederatedai/openfl) - Federated framework (primary)
- [FATE](https://fate.readthedocs.io/) - HE/MPC protocols (secondary)
- [Opacus](https://opacus.ai/) - PyTorch differential privacy
- [CrypTen](https://crypten.ai/) - MPC backend
- [Pyfhel](https://github.com/ibarrond/Pyfhel) - Python HE
- [OpenFHE](https://www.openfhe.org/) - Production FHE (C++)

## Research Foundations

This system implements privacy-preserving cooperative imitation learning based on:

1. **Local Differential Privacy FL** (edge-first): Zhao et al. (2020)
2. **FHE Cloud-Edge Architecture**: NTU Digital Research (DR)
3. **Federated Unlearning**: DTC Publications + FU paper
4. **Multi-Actor Imitation Learning**: Role-conditioned policies with coordination

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed citations and mappings.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{dynamical_sil_2025,
  title={Dynamical-SIL: Privacy-Preserving Multi-Actor Swarm Imitation Learning},
  author={Dynamical-SIL Contributors},
  year={2025},
  url={https://github.com/Danielfoojunwei/Multi-actor}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Security

For security concerns, see [SECURITY.md](SECURITY.md) for our vulnerability disclosure policy.
