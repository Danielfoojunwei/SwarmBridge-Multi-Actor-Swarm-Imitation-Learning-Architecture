# Dynamical-SIL ↔ Edge Platform Integration

## Overview

This integration layer seamlessly connects **Dynamical-SIL** (multi-actor swarm imitation learning) with the **Dynamical Edge Platform** (skill-centric VLA models on edge devices), enabling unified federated learning across cloud-based swarm coordination and edge-deployed robotics.

### **System Pairing**

| System | Primary Focus | Key Technologies |
|--------|--------------|------------------|
| **Dynamical-SIL** | Multi-actor swarm learning with privacy preservation | CSA packaging, OpenFL, ROS 2, multi-actor coordination |
| **Edge Platform** | Skill-centric VLA models on NVIDIA Jetson | MoE skills, N2HE encryption, DINOv2/SAM/V-JEPA, frozen base models |

### **Integration Benefits**

✅ **Unified Skill Repository**: CSAs and MoE skills automatically synchronized
✅ **Cross-System Federated Learning**: Coordinated training rounds across cloud and edge
✅ **Privacy-Preserving Aggregation**: Compatible encryption (Pyfhel ↔ N2HE)
✅ **Bidirectional Model Transfer**: Seamless conversion between CSA and MoE formats
✅ **Multi-Site Coordination**: Cloud swarm sites + edge devices in single federation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   INTEGRATION LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │  CSA-to-MoE      │◄───────►│  MoE-to-CSA      │            │
│  │  Adapter         │         │  Adapter         │            │
│  └──────────────────┘         └──────────────────┘            │
│           │                            │                       │
│           ▼                            ▼                       │
│  ┌─────────────────────────────────────────────────┐          │
│  │         API Bridge (Sync Controller)            │          │
│  │  • Registry API ↔ Skills API                   │          │
│  │  • Deployment status synchronization            │          │
│  │  • Version management                           │          │
│  └─────────────────────────────────────────────────┘          │
│           │                            │                       │
│           ▼                            ▼                       │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │  Encryption      │         │  Federated Sync  │            │
│  │  Bridge          │◄───────►│  Service         │            │
│  │  (Pyfhel↔N2HE)   │         │  (Coordinator)   │            │
│  └──────────────────┘         └──────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         ▲                                    ▲
         │                                    │
┌────────┴───────────┐            ┌──────────┴────────────┐
│  DYNAMICAL-SIL     │            │  EDGE PLATFORM        │
│                    │            │                       │
│  • CSA Registry    │            │  • Skills Library     │
│  • OpenFL Swarm    │            │  • MoE Experts        │
│  • Multi-Actor IL  │            │  • N2HE Encryption    │
│  • Pyfhel HE       │            │  • VLA Models         │
│  • ROS 2 Runtime   │            │  • Jetson Orin        │
└────────────────────┘            └───────────────────────┘
```

---

## Components

### 1. **CSA-to-MoE Adapter** (`adapters/csa_to_moe.py`)

Converts Cooperative Skill Artefacts to Mixture-of-Experts skill format.

**Conversion Strategy:**
- CSA policy adapters (per role) → MoE expert networks
- CSA coordination encoder → Skill router context
- Multi-actor roles → Expert specialization

**Example:**

```python
from integrations.edge_platform.adapters.csa_to_moe import CSAToMoEAdapter

adapter = CSAToMoEAdapter()

# Convert CSA → MoE
metadata = adapter.convert_csa_to_moe(
    csa_path="artifacts/cooperative_assembly_v1.0.tar.gz",
    output_path="skills/cooperative_assembly_moe.pt",
    router_hidden_dim=256,
)

print(f"Created MoE skill with {metadata.num_experts} experts")
# Output: Created MoE skill with 3 experts

# Reverse: MoE → CSA
adapter.convert_moe_to_csa(
    moe_path="skills/cooperative_assembly_moe.pt",
    output_path="artifacts/cooperative_assembly_from_edge.tar.gz",
    num_actors=3,
)
```

**MoE Skill Format:**

```python
{
    "metadata": {
        "skill_name": "cooperative_assembly",
        "version": "1.0",
        "num_experts": 3,
        "expert_specializations": ["leader", "follower", "observer"],
        "source_csa_id": "csa_abc123",
    },
    "experts": [
        {
            "expert_id": "leader",
            "specialization": "leader",
            "weights": {...},  # PyTorch state dict
        },
        # ... more experts
    ],
    "router": {...},  # Router network state dict
}
```

---

### 2. **API Bridge** (`bridges/api_bridge.py`)

Synchronizes artifacts between the Dynamical-SIL registry and Edge Platform skills API.

**Key Features:**
- Automatic CSA→MoE conversion and upload
- Skill discovery and import from Edge Platform
- Deployment status synchronization
- Bidirectional sync with conflict resolution

**Example:**

```python
from integrations.edge_platform.bridges.api_bridge import EdgePlatformAPIBridge

bridge = EdgePlatformAPIBridge(
    sil_registry_url="http://localhost:8000",
    edge_api_url="http://jetson-orin.local:8001",
    auth_token="your_token_here",  # Optional
)

# Push CSA to Edge Platform
edge_skill = await bridge.push_csa_to_edge(csa_id="csa_123")
print(f"Deployed as skill: {edge_skill.skill_id}")

# Pull skill from Edge Platform
csa_record = await bridge.pull_skill_from_edge(skill_id="skill_456")
print(f"Imported as CSA: {csa_record.id}")

# Bidirectional sync all
stats = await bridge.sync_all()
print(f"Synced: {stats['pushed']} pushed, {stats['pulled']} pulled")

# Health check
health = await bridge.health_check()
print(f"SIL: {health['sil_registry']}, Edge: {health['edge_platform']}")
```

**API Endpoints Mapped:**

| Dynamical-SIL | Edge Platform | Integration |
|---------------|---------------|-------------|
| `POST /api/v1/csa/upload` | `POST /api/v1/skills/upload` | Auto-convert during upload |
| `GET /api/v1/csa/list` | `GET /api/v1/skills` | Sync discovery |
| `GET /api/v1/csa/{id}/download` | `GET /api/v1/skills/{id}/download` | Format conversion |
| `POST /api/v1/deployment/deploy` | `POST /api/v1/robot/invoke_skill` | Deployment coordination |

---

### 3. **Encryption Bridge** (`bridges/encryption_bridge.py`)

Bridges homomorphic encryption systems between Pyfhel (SIL) and N2HE (Edge).

**Capabilities:**
- Encrypted weight transfer between systems
- Homomorphic aggregation across encryption schemes
- Privacy budget tracking (ε, δ, HE depth)

**Example:**

```python
from integrations.edge_platform.bridges.encryption_bridge import (
    EncryptionBridge,
    PrivacyBudgetTracker,
)

bridge = EncryptionBridge()

# Encrypt weights for Edge Platform (N2HE format)
weights = torch.randn(1000, 1000)
encrypted = bridge.encrypt_for_edge(weights, key_id="key_123")
print(f"Encrypted: {encrypted.encryption_context.scheme}")  # N2HE

# Decrypt from Edge Platform
decrypted_weights = bridge.decrypt_from_edge(encrypted)

# Aggregate encrypted gradients from both systems
encrypted_list = [
    bridge.encrypt_for_edge(grad1),  # From Edge device 1
    bridge.encrypt_for_edge(grad2),  # From Edge device 2
    bridge.encrypt_for_sil(grad3, scheme="CKKS"),  # From SIL site
]

aggregated = bridge.aggregate_encrypted(encrypted_list, strategy="mean")

# Track privacy budgets
tracker = PrivacyBudgetTracker()
tracker.add_dp_operation(epsilon=1.0, delta=1e-5, operation="gradient_release", system="dynamical_sil")
tracker.add_he_operation(depth=2, operation="aggregation", system="edge_platform")

status = tracker.get_budget_status()
print(f"ε used: {status['differential_privacy']['epsilon_total']}")
print(f"HE depth: {status['homomorphic_encryption']['depth_used']}")
```

**Privacy Budget Limits:**

- **Differential Privacy**: ε ≤ 10.0, δ ≤ 1e-5 (default)
- **Homomorphic Encryption**: Max depth = 10 (typical for CKKS/BFV)

---

### 4. **Federated Sync Service** (`sync/federated_sync.py`)

Orchestrates coordinated federated learning rounds across both systems.

**Architecture:**
- SIL sites train multi-actor CSAs (OpenFL)
- Edge devices train MoE skills (N2HE encryption)
- Sync service coordinates rounds and aggregates globally

**Example:**

```python
from integrations.edge_platform.sync.federated_sync import (
    FederatedSyncService,
    SyncMode,
)

sync = FederatedSyncService(
    sil_coordinator_url="http://localhost:8001",
    edge_api_url="http://jetson-orin:8001",
)

# Start coordinated training round
round_id = await sync.start_federated_round(
    skill_name="cooperative_pick_place",
    num_sil_sites=3,
    num_edge_devices=2,
    privacy_mode="encrypted",
)

print(f"Round {round_id} started")

# Monitor progress
status = await sync.get_round_status(round_id)
print(f"Duration: {status['duration_seconds']}s")
print(f"SIL status: {status['sil_status']}")
print(f"Edge status: {status['edge_status']}")

# Complete round and aggregate
results = await sync.complete_round(
    round_id=round_id,
    aggregation_weights={"sil": 0.6, "edge": 0.4},  # Weight contributions
)

print(f"Aggregated CSA: {results['csa_id']}")
print(f"Aggregated Skill: {results['skill_id']}")

# View all rounds
rounds = sync.get_all_rounds()
for r in rounds:
    print(f"Round {r['round_id']}: {r['status']}")
```

**Federated Round Workflow:**

```
┌─────────────────────────────────────────────────────┐
│ 1. START ROUND                                      │
│    • Configure both systems                         │
│    • Initialize privacy budgets                     │
│    • Distribute base model                          │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐      ┌──────────────┐
│ SIL TRAINING  │      │ EDGE TRAINING│
│ • 3 sites     │      │ • 2 devices  │
│ • Multi-actor │      │ • MoE skills │
│ • OpenFL      │      │ • N2HE       │
└───────┬───────┘      └──────┬───────┘
        │                     │
        └──────────┬──────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│ 2. AGGREGATE                                        │
│    • Collect encrypted updates                      │
│    • Cross-system aggregation                       │
│    • Weighted combination (60% SIL, 40% Edge)       │
└──────────────────┬──────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│ 3. DISTRIBUTE                                       │
│    • Send aggregated CSA to SIL sites              │
│    • Send aggregated skill to Edge devices          │
│    • Update privacy budgets                         │
└─────────────────────────────────────────────────────┘
```

---

### 5. **Configuration Manager** (`config/integration_config.py`)

Unified configuration for the integration layer.

**Example:**

```python
from integrations.edge_platform.config.integration_config import IntegrationConfig

# Create default config
config = IntegrationConfig()

# Customize
config.endpoints.sil_registry = "http://192.168.1.10:8000"
config.endpoints.edge_platform = "http://jetson-orin-1.local:8001"
config.federated.num_sil_sites = 5
config.federated.num_edge_devices = 3
config.federated.sil_weight = 0.7
config.federated.edge_weight = 0.3
config.encryption.enable_cross_system_encryption = True

# Save configuration
config.to_yaml(Path("config/integration.yaml"))

# Load configuration
loaded = IntegrationConfig.from_yaml(Path("config/integration.yaml"))
```

**Configuration Schema:**

```yaml
endpoints:
  sil_registry: http://localhost:8000
  sil_coordinator: http://localhost:8001
  edge_platform: http://jetson-orin.local:8001
  edge_device_discovery: http://jetson-orin.local:8001/api/devices

encryption:
  sil_scheme: CKKS  # BFV, CKKS
  edge_scheme: N2HE
  security_bits: 128
  poly_modulus_degree: 8192
  enable_cross_system_encryption: true

federated:
  num_sil_sites: 3
  num_edge_devices: 2
  rounds_per_sync: 5
  aggregation_strategy: weighted_average  # median, trimmed_mean
  sil_weight: 0.5
  edge_weight: 0.5
  privacy_mode: encrypted  # encrypted, differential_privacy, hybrid

sync:
  auto_sync: true
  sync_interval_seconds: 300  # 5 minutes
  max_retries: 3
  retry_delay_seconds: 10
  enable_bidirectional: true

auth_token: null  # Optional authentication
```

---

### 6. **Data Converters** (`converters/data_converters.py`)

Convert data formats between systems.

**Example:**

```python
from integrations.edge_platform.converters.data_converters import (
    ObservationConverter,
    ActionConverter,
    MetadataConverter,
)

# Observation conversion
sil_obs = {
    "multi_actor_observations": np.random.randn(3, 10, 15),  # [N, T, D]
}
edge_obs = ObservationConverter.sil_to_edge(sil_obs)
# → {"visual": Tensor, "proprioception": Tensor, "language_instruction": ""}

# Action conversion
edge_action = torch.randn(7)  # Single-actor action
sil_actions = ActionConverter.edge_to_sil(edge_action, num_actors=3)
# → {"multi_actor_actions": Array[3, 7]}

# Metadata conversion
csa_meta = {"skill_name": "assembly", "roles": [...], ...}
skill_meta = MetadataConverter.csa_metadata_to_skill_metadata(csa_meta)
# → {"skill_name": "assembly", "num_experts": N, ...}
```

---

## Usage Scenarios

### Scenario 1: Deploy CSA to Edge Devices

```python
from integrations.edge_platform.bridges.api_bridge import EdgePlatformAPIBridge
from integrations.edge_platform.config.integration_config import IntegrationConfig

# Load config
config = IntegrationConfig.from_yaml("config/integration.yaml")

# Initialize bridge
bridge = EdgePlatformAPIBridge(
    sil_registry_url=config.endpoints.sil_registry,
    edge_api_url=config.endpoints.edge_platform,
    auth_token=config.auth_token,
)

# Push CSA to all edge devices
csa_id = "csa_cooperative_assembly_v2"
edge_skill = await bridge.push_csa_to_edge(csa_id)

print(f"✓ Deployed to Edge Platform: {edge_skill.skill_id}")
print(f"  Experts: {edge_skill.expert_specializations}")
print(f"  Devices: Available via /api/v1/robot/invoke_skill")
```

### Scenario 2: Import Edge-Trained Skill into Swarm

```python
# Pull skill trained on edge devices
skill_id = "skill_new_manipulation"
csa_record = await bridge.pull_skill_from_edge(skill_id)

print(f"✓ Imported to Dynamical-SIL: {csa_record.id}")
print(f"  Now available for multi-actor swarm training")
print(f"  Skill: {csa_record.skill_name} v{csa_record.version}")
```

### Scenario 3: Coordinated Federated Round

```python
from integrations.edge_platform.sync.federated_sync import FederatedSyncService

sync = FederatedSyncService(
    sil_coordinator_url=config.endpoints.sil_coordinator,
    edge_api_url=config.endpoints.edge_platform,
)

# Start training across 3 cloud sites + 2 edge devices
round_id = await sync.start_federated_round(
    skill_name="cooperative_pick_place",
    num_sil_sites=3,
    num_edge_devices=2,
    privacy_mode="encrypted",
)

# Wait for training (in practice, poll or use webhooks)
await asyncio.sleep(600)  # 10 minutes

# Aggregate and distribute
results = await sync.complete_round(
    round_id=round_id,
    aggregation_weights={"sil": 0.6, "edge": 0.4},
)

print(f"✓ Federated round complete")
print(f"  Aggregated CSA: {results['csa_id']}")
print(f"  Aggregated Skill: {results['skill_id']}")
```

### Scenario 4: Privacy-Preserving Cross-System Aggregation

```python
from integrations.edge_platform.bridges.encryption_bridge import (
    EncryptionBridge,
    PrivacyBudgetTracker,
)

bridge = EncryptionBridge()
tracker = PrivacyBudgetTracker()

# Collect encrypted gradients from both systems
sil_gradients = [...]  # From OpenFL sites
edge_gradients = [...]  # From Edge devices

# Encrypt for cross-system aggregation
encrypted_sil = [bridge.encrypt_for_sil(g, scheme="CKKS") for g in sil_gradients]
encrypted_edge = [bridge.encrypt_for_edge(g) for g in edge_gradients]

# Aggregate homomorphically
all_encrypted = encrypted_sil + encrypted_edge
aggregated = bridge.aggregate_encrypted(all_encrypted, strategy="mean")

# Track privacy budget
tracker.add_he_operation(depth=2, operation="cross_system_agg", system="integration")

if tracker.is_budget_exceeded():
    print("⚠ Privacy budget exceeded!")
else:
    print(f"✓ Privacy budget OK (HE depth: {tracker.he_depth_used}/10)")
```

---

## Testing

Run integration tests:

```bash
# Unit tests (no services required)
pytest tests/integration/test_edge_platform_integration.py -v

# Integration tests (requires both systems running)
pytest tests/integration/test_edge_platform_integration.py -v -m integration

# Specific test
pytest tests/integration/test_edge_platform_integration.py::TestAPIBridge::test_health_check -v
```

---

## Integration Checklist

Before deploying the integration:

- [ ] **Both systems running**
  - [ ] Dynamical-SIL registry API (`http://localhost:8000`)
  - [ ] Dynamical-SIL swarm coordinator (`http://localhost:8001`)
  - [ ] Edge Platform API (`http://jetson-orin:8001`)

- [ ] **Configuration**
  - [ ] `config/integration.yaml` created with correct endpoints
  - [ ] Authentication tokens configured (if required)
  - [ ] Privacy budgets set appropriately

- [ ] **Network connectivity**
  - [ ] SIL registry reachable from edge devices
  - [ ] Edge API reachable from SIL services
  - [ ] Firewall rules allow bidirectional traffic

- [ ] **Storage**
  - [ ] Sufficient disk space for CSA/skill artifacts
  - [ ] Shared storage or S3 bucket for large models (optional)

- [ ] **Testing**
  - [ ] Health checks passing (`bridge.health_check()`)
  - [ ] Simple CSA push/pull working
  - [ ] Privacy mechanisms functional

---

## Troubleshooting

### Issue: `ConnectionRefusedError` when syncing

**Cause:** Services not running or incorrect URLs

**Solution:**
```python
# Check health
bridge = EdgePlatformAPIBridge(...)
health = await bridge.health_check()
print(health)

# If False, verify:
# 1. Services are running
# 2. URLs are correct
# 3. Firewalls allow connections
```

### Issue: CSA conversion fails

**Cause:** Incompatible CSA format or missing fields

**Solution:**
```python
# Validate CSA before conversion
from ml.artifact.validator import CSAValidator

validator = CSAValidator()
is_valid, errors = validator.validate_package("artifacts/my_csa.tar.gz")

if not is_valid:
    print(f"Validation errors: {errors}")
```

### Issue: Privacy budget exceeded

**Cause:** Too many federated rounds or operations

**Solution:**
```python
# Check budget before operations
tracker = PrivacyBudgetTracker()
# ... (after operations)

status = tracker.get_budget_status()
print(f"ε: {status['differential_privacy']['epsilon_total']}/10.0")

# If exceeded, reset or adjust limits
tracker.epsilon_total = 0.0  # Reset (use with caution!)
```

### Issue: Aggregation weights don't sum to 1.0

**Cause:** Configuration error

**Solution:**
```python
# Weights must sum to 1.0
config.federated.sil_weight = 0.6
config.federated.edge_weight = 0.4
assert config.federated.sil_weight + config.federated.edge_weight == 1.0
```

---

## Performance Considerations

| Operation | Typical Duration | Notes |
|-----------|------------------|-------|
| CSA → MoE conversion | 2-5 seconds | Depends on model size |
| MoE → CSA conversion | 2-5 seconds | - |
| Upload CSA to Edge (10MB) | 5-10 seconds | Depends on network |
| Download skill from Edge | 5-10 seconds | - |
| Bidirectional sync (10 artifacts) | 1-2 minutes | Parallel uploads |
| Encrypted aggregation (3 sites) | 10-30 seconds | HE overhead |
| Federated round (full) | 10-30 minutes | Training time dominant |

**Optimization Tips:**
- Enable `auto_sync=True` for background synchronization
- Use `aggregation_weights` to balance cloud vs edge contributions
- Cache converted artifacts to avoid repeated conversions
- Use compression for large model transfers

---

## Security Notes

🔒 **Authentication:** Always use `auth_token` in production
🔒 **TLS/HTTPS:** Enable TLS for all API communications
🔒 **Privacy Budgets:** Monitor ε, δ budgets closely
🔒 **Encryption Keys:** Rotate HE keys periodically
🔒 **Network Security:** Use VPN or private networks for federated sites

---

## References

- **Dynamical-SIL Documentation:** `docs/ARCHITECTURE.md`, `docs/ADVANCED_MULTI_ACTOR.md`
- **Edge Platform Documentation:** [GitHub - Dynamical Edge Platform](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform)
- **OpenFL:** https://github.com/intel/openfl
- **Pyfhel (HE):** https://github.com/ibarrond/Pyfhel
- **Federated Learning:** McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)

---

## Contact & Support

For integration issues or questions:
- Open an issue on this repository
- Refer to `docs/ARCHITECTURE.md` for system internals
- Check `tests/integration/test_edge_platform_integration.py` for examples

**Status:** ✅ Production-ready (v1.0.0)
