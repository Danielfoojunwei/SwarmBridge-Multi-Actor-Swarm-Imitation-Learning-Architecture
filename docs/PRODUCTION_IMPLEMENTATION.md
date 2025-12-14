# SwarmBridge Production Implementation Guide

This document describes the production-ready implementations that replace all mock code, aligned with Edge Platform and SwarmBrain technologies.

## 🎯 Overview

SwarmBridge now uses production open-source libraries that align with the broader Dynamical ecosystem:

| Component | Technology | Alignment |
|-----------|-----------|-----------|
| **Federated Learning** | Flower | SwarmBrain |
| **Deep Learning** | PyTorch + robomimic | Edge Platform |
| **ROS Integration** | ROS 2 Humble | SwarmBrain |
| **Encryption** | Pyfhel (HE) | SwarmBrain (OpenFHE) / Edge Platform (N2HE) |
| **APIs** | FastAPI + httpx | Edge Platform |
| **Message Queue** | RabbitMQ + Celery | SwarmBrain |
| **Monitoring** | Prometheus + Grafana | SwarmBrain |

## 📦 Production Components

### 1. Federated Learning Adapter (Flower-based)

**File**: `swarmbridge/adapters/federated_adapter_flower.py`

**Features**:
- Production Flower `NumPyClient` implementation
- Integrates with SwarmBrain's Flower server
- Supports encrypted model updates via Pyfhel
- Differential privacy with DP-SGD (Opacus)
- Real federated training with configurable rounds

**Usage**:
```python
from swarmbridge.adapters.federated_adapter_flower import FederatedLearningAdapter
import torch.nn as nn

# Initialize adapter
adapter = FederatedLearningAdapter(
    server_address="swarmbrain.local:8080",
    use_encryption=True,
)

# Submit to federated training
await adapter.submit_to_federated_training(
    model=cooperative_bc_model,
    train_loader=train_data,
    skill_name="handover",
    num_rounds=10,
)
```

**Key Classes**:
- `SwarmBridgeFlowerClient(NumPyClient)`: Flower client for federated training
- `FederatedLearningAdapter`: High-level adapter interface
- `FlowerConfig`: Configuration dataclass

### 2. Cooperative BC Trainer (PyTorch)

**File**: `swarmbridge/training/cooperative_bc_trainer.py`

**Features**:
- Multi-actor behavior cloning with role-conditioned policies
- Coordination encoder (Transformer/RNN/MLP)
- Production PyTorch training loop
- Gradient clipping, checkpointing, LR scheduling
- Validation and early stopping

**Architecture**:
```
Input: Multi-Actor Observations [batch, num_actors, obs_dim]
         ↓
    Coordination Encoder (Transformer/RNN/MLP)
         ↓
    Coordination Latent [batch, latent_dim]
         ↓
    ┌─────────┬─────────┬─────────┐
    │ Policy  │ Policy  │ Policy  │ (Role-Conditioned)
    │ Role 0  │ Role 1  │ Role N  │
    └─────────┴─────────┴─────────┘
         ↓
    Predicted Actions [batch, num_actors, action_dim]
```

**Usage**:
```python
from swarmbridge.training.cooperative_bc_trainer import (
    CooperativeBCModel,
    CooperativeBCTrainer,
    TrainingConfig,
)

# Create model
model = CooperativeBCModel(
    num_actors=2,
    role_configs=[
        {"role_id": "giver", "observation_dim": 15, "action_dim": 7},
        {"role_id": "receiver", "observation_dim": 15, "action_dim": 7},
    ],
    config=TrainingConfig(
        coordination_encoder_type="transformer",
        num_epochs=100,
    ),
)

# Train
trainer = CooperativeBCTrainer(model, config)
history = trainer.train(train_loader, val_loader, checkpoint_dir)
```

**Key Classes**:
- `CoordinationEncoder`: Encodes multi-actor interactions
- `RoleConditionedPolicy`: Per-role action prediction
- `CooperativeBCModel`: Complete multi-actor model
- `CooperativeBCTrainer`: Production training loop

### 3. ROS 2 Multi-Actor Capture

**File**: `swarmbridge/capture/ros2_multiactor_capture.py`

**Features**:
- Real ROS 2 Humble integration
- Synchronized multi-robot state capture
- rosbag2 recording
- Joint states, end-effector poses, F/T sensors, grippers
- Camera feeds (RGB-D)

**Usage**:
```python
from swarmbridge.capture.ros2_multiactor_capture import (
    capture_multiactor_demonstration,
    RobotConfig,
)

# Configure robots
robot_configs = [
    RobotConfig(robot_id="robot_0", role_id="giver"),
    RobotConfig(robot_id="robot_1", role_id="receiver"),
]

# Capture demonstration
demo_file = capture_multiactor_demonstration(
    robot_configs=robot_configs,
    skill_name="handover",
    duration_s=30.0,
    control_freq_hz=100.0,
)
```

**Key Classes**:
- `MultiActorCaptureNode(Node)`: ROS 2 node for capture
- `RobotConfig`: Per-robot topic configuration
- `Demonstration`: Timestamped multi-actor state

## 🔧 Installation

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. ROS 2 Humble (Ubuntu 22.04)

```bash
# Install ROS 2 Humble
sudo apt install ros-humble-desktop
sudo apt install ros-humble-rosbag2-py

# Source ROS 2
source /opt/ros/humble/setup.bash
```

### 3. GPU Support (Optional)

```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Development Tools (Optional)

```bash
pre-commit install
```

## 🚀 End-to-End Workflow

### Complete Production Pipeline

```python
import asyncio
from swarmbridge import SwarmBridgePipeline
from swarmbridge.schemas import SharedRoleSchema, CoordinationPrimitives, CoordinationType
from swarmbridge.adapters.federated_adapter_flower import FederatedLearningAdapter
from swarmbridge.training.cooperative_bc_trainer import TrainingConfig
from swarmbridge.capture.ros2_multiactor_capture import RobotConfig

async def production_workflow():
    # 1. CAPTURE: Multi-actor demonstrations via ROS 2
    robot_configs = [
        RobotConfig(robot_id="robot_0", role_id="giver"),
        RobotConfig(robot_id="robot_1", role_id="receiver"),
    ]
    
    demo_file = capture_multiactor_demonstration(
        robot_configs=robot_configs,
        skill_name="handover",
        duration_s=30.0,
    )
    
    # 2. PROCESS: Load and prepare data
    # (Implementation in pipeline)
    
    # 3. TRAIN: Cooperative BC with PyTorch
    config = TrainingConfig(
        coordination_encoder_type="transformer",
        num_epochs=100,
        batch_size=32,
    )
    
    # Create model and train
    # (See cooperative_bc_trainer.py)
    
    # 4. FEDERATE: Submit to Flower server (SwarmBrain)
    fl_adapter = FederatedLearningAdapter(
        server_address="swarmbrain.local:8080",
        use_encryption=True,
    )
    
    await fl_adapter.submit_to_federated_training(
        model=trained_model,
        train_loader=train_data,
        skill_name="handover",
        num_rounds=10,
    )
    
    # 5. PACKAGE: Create CSA
    # (CSA packager - existing implementation)
    
    # 6. DEPLOY: To Edge Platform
    # (Edge Platform adapter - existing implementation)

if __name__ == "__main__":
    asyncio.run(production_workflow())
```

## 🔐 Privacy & Security

### Homomorphic Encryption (Pyfhel)

```python
from Pyfhel import Pyfhel

# Create HE context
HE = Pyfhel()
HE.contextGen(scheme='ckks', n=2**14, scale=2**30, qi_sizes=[60,30,30,30,60])
HE.keyGen()
HE.relinKeyGen()
HE.rotateKeyGen()

# Encrypt model weights
encrypted_weights = [HE.encrypt(w.flatten().numpy()) for w in model.parameters()]

# Use in Flower client
client = SwarmBridgeFlowerClient(
    model=model,
    train_loader=train_data,
    encryption_context=HE,
)
```

### Differential Privacy (Opacus)

```python
from opacus import PrivacyEngine

# Wrap model and optimizer
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)
```

## 📊 Monitoring (Prometheus + Grafana)

Aligned with SwarmBrain's monitoring stack:

```python
from prometheus_client import start_http_server, Counter, Histogram

# Metrics
training_rounds = Counter('training_rounds_total', 'Total FL rounds')
training_loss = Histogram('training_loss', 'Training loss distribution')

# Start metrics server
start_http_server(8000)
```

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=swarmbridge --cov-report=html

# Specific module
pytest tests/test_cooperative_bc.py -v
```

### Integration Test

```python
import pytest
from swarmbridge.training.cooperative_bc_trainer import *

def test_cooperative_bc_training():
    # Create dummy data
    model = CooperativeBCModel(
        num_actors=2,
        role_configs=[...],
        config=TrainingConfig(num_epochs=2),
    )
    
    trainer = CooperativeBCTrainer(model, config)
    history = trainer.train(train_loader, val_loader)
    
    assert len(history["train_loss"]) == 2
    assert history["train_loss"][-1] < history["train_loss"][0]
```

## 📝 Migration from Mock to Production

| Component | Before (Mock) | After (Production) |
|-----------|--------------|-------------------|
| **FL Adapter** | MockFederatedLearningAdapter | Flower-based FederatedLearningAdapter |
| **Training** | Placeholder pipeline | PyTorch + robomimic trainer |
| **Capture** | Synthetic data | ROS 2 Humble integration |
| **Encryption** | No-op placeholders | Pyfhel HE operations |
| **Registry** | Mock storage | FastAPI + SQLAlchemy |

## 🔄 Ecosystem Integration

### SwarmBrain Integration

```python
# SwarmBrain uses Flower server
# SwarmBridge connects as Flower client

# SwarmBridge (this repo)
adapter = FederatedLearningAdapter(
    server_address="swarmbrain.local:8080"
)

# SwarmBrain (external)
# Runs: flwr-server --server-address 0.0.0.0:8080
```

### Edge Platform Integration

```python
# Edge Platform provides Dynamical API
# SwarmBridge delegates runtime execution

from swarmbridge.adapters.runtime_adapter import EdgePlatformRuntimeAdapter

runtime = EdgePlatformRuntimeAdapter(
    edge_api_url="http://edge-platform:8080",
    registry_adapter=registry,
)

# Execute trained skill on Edge Platform
execution_id = await runtime.execute_skill(
    csa_id="handover_v1",
    robot_id="robot_1",
    task_parameters={"object": "cube"},
)
```

## 🎓 Key Takeaways

✅ **Production Ready**: All core components use real open-source libraries  
✅ **Ecosystem Aligned**: Technologies match Edge Platform and SwarmBrain  
✅ **Privacy Preserving**: Pyfhel HE + Opacus DP integration  
✅ **ROS 2 Native**: Real robot integration via ROS 2 Humble  
✅ **Federated Learning**: Flower client for distributed training  
✅ **Modular Architecture**: Clean separation of concerns  

## 📚 References

- [Flower Documentation](https://flower.dev/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ROS 2 Humble](https://docs.ros.org/en/humble/)
- [Pyfhel Documentation](https://pyfhel.readthedocs.io/)
- [robomimic](https://robomimic.github.io/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

**Last Updated**: 2025-12-14  
**Version**: 2.0.0 (Production)
