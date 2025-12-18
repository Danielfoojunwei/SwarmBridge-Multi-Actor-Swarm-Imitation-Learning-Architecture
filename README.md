# SwarmBridge: Multi-Actor Swarm IL Extension for Dynamical v0.3.3

[![CI](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture/workflows/CI/badge.svg)](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0--dynamical-green.svg)](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture)
[![Dynamical](https://img.shields.io/badge/extends-Dynamical%20v0.3.3-blue)](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Learning-Platform)

**SwarmBridge** is the **multi-actor swarm extension layer** for the Dynamical skill-centric edge platform. It turns multi-robot demonstrations into **MoE skill experts** that plug directly into Dynamical's VLA + MoE Skills Layer, enabling federated learning of cooperative behaviors across distributed robot fleets.

## 🎯 Relationship to Dynamical

SwarmBridge sits **on top of Dynamical's skill-centric architecture** and extends it to multi-actor scenarios:

```
┌──────────────────────────────────────────────────────────────┐
│                    DYNAMICAL v0.3.3                          │
│         (Skill-Centric Edge IL Platform)                     │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Perception: MOAI Compression (512-dim embeddings)     │ │
│  │  Foundation: Pi0/OpenVLA 7B (frozen)                   │ │
│  │  Skills: MoE Layer (10-50M experts per skill)          │ │
│  │  Privacy: N2HE 128-bit encryption                      │ │
│  │  Platform: NVIDIA Jetson AGX Orin                      │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                            ▲
                            │ Consumes MoE Skill Experts
                            │
┌──────────────────────────────────────────────────────────────┐
│                     SWARMBRIDGE                              │
│         (Multi-Actor Swarm IL Extension)                     │
│                                                              │
│  INPUT from Dynamical:                                       │
│  • MOAI-compressed embeddings (512-dim)                     │
│  • Skill IDs from Dynamical skill catalog                   │
│  • N2HE encryption scheme                                   │
│                                                              │
│  PROCESSING:                                                 │
│  • Multi-actor ROS 2 demonstration capture                  │
│  • Cooperative BC with coordination encoder                 │
│  • Role-conditioned policy training                         │
│  • Cross-role consistency regularization                    │
│  • Federated learning via Flower (encrypted)                │
│  • Local-global distillation                                │
│                                                              │
│  OUTPUT to Dynamical:                                        │
│  • Per-role MoE skill experts (ONNX/PyTorch)               │
│  • CooperativeSkillArtifact manifests                       │
│  • Compatible with Dynamical MoE layer format               │
│  • Versioned skill artifacts (round_id, site_id)            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼ Distributed Sites
┌──────────────────────────────────────────────────────────────┐
│                    SWARMBRAIN                                │
│         (Federated Orchestration & Aggregation)              │
│                                                              │
│  • Flower server for multi-site FL                          │
│  • OpenFHE secure aggregation                               │
│  • Global skill prior management                            │
│  • Mission orchestration (ROS 2)                            │
└──────────────────────────────────────────────────────────────┘
```

### Key Principle

**SwarmBridge never replaces Dynamical's VLA + MoE layer** - it **feeds it new cooperative skills**. Each robot still runs Dynamical's edge stack; SwarmBridge extends it with multi-actor coordination.

## 🏗️ Core Capabilities

✅ **Multi-Actor Demonstration Capture** - ROS 2 synchronized recording from swarms  
✅ **Cooperative BC Training** - Coordination encoder + role-conditioned policies  
✅ **Dynamical MoE Export** - Outputs skill experts in Dynamical-compatible format  
✅ **Federated Swarm IL** - Flower-based encrypted multi-site learning  
✅ **Local-Global Distillation** - Aligns local swarm policies with global priors  
✅ **Privacy-Preserving** - N2HE/Pyfhel encryption + Opacus DP  

## 📊 Technology Stack Alignment

| Component | SwarmBridge | Dynamical v0.3.3 | Status |
|-----------|-------------|------------------|--------|
| **Embeddings** | MOAI 512-dim | MOAI 512-dim | ✅ Aligned |
| **Encryption** | Pyfhel/N2HE | N2HE 128-bit | ✅ Compatible |
| **IL Framework** | PyTorch + robomimic | LeRobot | ✅ Compatible |
| **Federated Learning** | Flower | FedAvg protocol | ✅ Aligned |
| **Skill Format** | CSA → MoE experts | MoE experts | ✅ Compatible |
| **ROS** | ROS 2 Humble | - | ✅ Multi-robot |
| **Foundation Model** | Uses Dynamical VLA | Pi0/OpenVLA 7B | ✅ Shared |

## 🚀 Quick Start

### Prerequisites

SwarmBridge requires **Dynamical v0.3.3** to be deployed on each robot in your swarm.

```bash
# Install Dynamical first (on each robot)
# See: https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Learning-Platform

# Install SwarmBridge (cloud/edge coordinator)
pip install -r requirements.txt
sudo apt install ros-humble-desktop ros-humble-rosbag2-py
source /opt/ros/humble/setup.bash
```

### Basic Workflow

```python
from swarmbridge.integrations.dynamical import (
    build_cooperative_skill_artifacts,
    register_skills_with_dynamical
)
from swarmbridge.training.cooperative_bc_trainer import (
    CooperativeBCModel, CooperativeBCTrainer, TrainingConfig
)

# 1. Configure multi-actor roles aligned with Dynamical skills
role_configs = [
    {
        "role_id": "giver",
        "skill_id": "handover_box",  # From Dynamical skill catalog
        "uses_moai": True,
        "moai_version": "0.3.3",
        "embedding_dim": 512,
        "observation_dim": 512,  # MOAI embeddings
        "action_dim": 7,
    },
    {
        "role_id": "receiver",
        "skill_id": "handover_box",
        "uses_moai": True,
        "moai_version": "0.3.3",
        "embedding_dim": 512,
        "observation_dim": 512,
        "action_dim": 7,
    },
]

# 2. Train cooperative BC model
config = TrainingConfig(
    coordination_encoder_type="transformer",
    use_moai_embeddings=True,
    use_local_global_distillation=True,
    global_prior_uri="dynamical://skills/handover_box/global_v3",
)

model = CooperativeBCModel(
    num_actors=2,
    role_configs=role_configs,
    config=config,
)

trainer = CooperativeBCTrainer(model, config)
history = trainer.train(train_loader, val_loader)

# 3. Export as Dynamical-compatible MoE skill experts
artifacts = build_cooperative_skill_artifacts(
    model=model,
    role_configs=role_configs,
    skill_id="handover_box",
    moai_config={"version": "0.3.3", "embedding_dim": 512},
    output_dir="./artifacts/handover_v4/",
)

# 4. Register with Dynamical skill registry
register_skills_with_dynamical(
    artifact_manifest=artifacts["manifest"],
    registry_uri="dynamical://skills",
    encryption_scheme="n2he",
)

# 5. Deploy to Dynamical robots
# Each robot's Dynamical instance will load the appropriate role expert
# into its MoE layer and execute as part of normal skill invocation
```

## 📦 CooperativeSkillArtifact Schema

SwarmBridge exports skills in a format that **directly plugs into Dynamical's MoE layer**:

```python
from swarmbridge.schemas.cooperative_skill_artifact import CooperativeSkillArtifact

artifact = CooperativeSkillArtifact(
    skill_id="handover_box",           # From Dynamical skill catalog
    role_id="giver",                   # Multi-actor role
    input_embedding_type="moai_512",   # Matches Dynamical perception
    expert_checkpoint_uri="s3://skills/handover_v4/giver.onnx",
    encryption_scheme="n2he",          # Compatible with Dynamical
    version="4.0",
    site_id="warehouse_A",
    round_id=15,
    coordination_primitive="handover",
    compatible_roles=["receiver"],
    metadata={
        "moai_version": "0.3.3",
        "dynamical_version": "0.3.3",
        "training_episodes": 500,
        "federated_sites": 3,
    }
)
```

## 🔄 Adaptive Loops in SwarmBridge

SwarmBridge extends Dynamical's local-global distillation to multi-actor scenarios:

### 1. **Multi-Actor Novelty Sampling**

```python
from swarmbridge.adaptive.novelty_sampler import SwarmNoveltySampler

sampler = SwarmNoveltySampler(
    detect_patterns=["near_collision", "tricky_handover", "coordination_failure"]
)

# Prioritize rare coordination patterns in training
prioritized_episodes = sampler.sample(
    all_episodes,
    prioritize_rare=True,
    coordination_aware=True,
)
```

### 2. **Cross-Role Consistency Regularization**

```python
# In cooperative BC trainer
loss = bc_loss + lambda_consistency * cross_role_consistency_loss(
    giver_actions, receiver_actions, coordination_latent
)
```

### 3. **Local-Global Distillation (Multi-Actor)**

```python
# Local training objective
local_loss = (
    bc_loss_on_local_demos
    + kl_divergence(local_policy, global_cooperative_prior)
    + cross_role_consistency_loss
)

# Flower federated round
global_update = federated_average(
    [site1_update, site2_update, site3_update],
    encryption="n2he"
)
```

## 🔐 Privacy & Security (Aligned with Dynamical)

SwarmBridge uses the **same encryption schemes as Dynamical**:

```python
# N2HE encryption (Dynamical standard)
from swarmbridge.privacy.n2he_bridge import encrypt_for_dynamical

encrypted_update = encrypt_for_dynamical(
    model_weights,
    encryption_scheme="n2he_128bit",
    key_id="dynamical_fleet_key_v3"
)

# Differential Privacy (Opacus)
privacy_engine.make_private(
    model, optimizer, train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)
```

## 📚 Documentation

- **[Relationship to Dynamical](docs/RELATIONSHIP_TO_DYNAMICAL.md)** - Detailed integration guide
- **[CooperativeSkillArtifact Spec](docs/CSA_SPECIFICATION.md)** - Skill export format
- **[MOAI Integration](docs/MOAI_INTEGRATION.md)** - Using Dynamical embeddings
- **[Local-Global Distillation](docs/LOCAL_GLOBAL_DISTILLATION.md)** - Multi-actor FL
- **[Production Implementation](docs/PRODUCTION_IMPLEMENTATION.md)** - Tech stack details

## 🧪 End-to-End Example

See `examples/handover_with_dynamical/` for a complete workflow:

1. **Capture** multi-actor handover demo with 2 robots running Dynamical
2. **Train** cooperative BC using MOAI embeddings from Dynamical
3. **Export** per-role skill experts in Dynamical MoE format
4. **Register** skills in Dynamical cloud registry
5. **Execute** cooperative handover via Dynamical's skill API on humanoids

## 🎯 Key Differentiators vs Standalone IL

| Feature | Generic Multi-Robot IL | SwarmBridge + Dynamical |
|---------|----------------------|------------------------|
| **Perception** | Raw observations | MOAI 512-dim (Dynamical) ✅ |
| **Output Format** | Generic policies | Dynamical MoE experts ✅ |
| **Deployment** | Custom per robot | Dynamical edge stack ✅ |
| **Privacy** | Optional | N2HE (Dynamical standard) ✅ |
| **Federated Learning** | Generic FL | Aligned with Dynamical FL ✅ |
| **Skill Catalog** | None | Shared with Dynamical ✅ |
| **Cross-Role Consistency** | Ad-hoc | Explicit regularization ✅ |
| **Adaptive Sampling** | None | Novelty-based (Dynamical-style) ✅ |

## 🏆 Production Metrics

- **MOAI Embedding Support**: ✅ 512-dim (Dynamical v0.3.3)
- **Skill Export Format**: ✅ Dynamical MoE compatible
- **Encryption**: ✅ N2HE 128-bit + Pyfhel
- **Federated Learning**: ✅ Flower with encrypted aggregation
- **Tests**: ✅ 15+ unit tests + integration tests
- **Documentation**: ✅ 5 comprehensive guides

## 📖 Citation

```bibtex
@software{swarmbridge2024,
  title = {SwarmBridge: Multi-Actor Swarm IL Extension for Dynamical},
  author = {Foo, Daniel Jun Wei},
  year = {2024},
  version = {2.0.0-dynamical},
  note = {Extends Dynamical v0.3.3 skill-centric edge platform},
  url = {https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture}
}
```

## 🤝 Related Projects

- **[Dynamical v0.3.3](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Learning-Platform)** - Skill-centric edge IL platform (required)
- **[SwarmBrain](https://github.com/Danielfoojunwei/SwarmBraim-Swarm-Imitation-Learning-Orchestration-Architecture)** - Federated orchestration layer
- **[Edge Platform](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform)** - Edge deployment (same as Dynamical)

---

**Built as an extension to Dynamical v0.3.3 • Enables multi-actor swarm IL with skill-centric deployment**

*Last Updated: 2025-12-14 | Version: 2.0.0-dynamical*
