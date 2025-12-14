# Integrations

This directory contains integration layers for connecting Dynamical-SIL with external systems.

## Edge Platform Integration

**Location:** `integrations/edge_platform/`

Comprehensive integration with the [Dynamical Edge Platform](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform) for seamless federated learning across cloud-based multi-actor swarm learning and edge-deployed VLA models.

### Quick Start

```python
from integrations.edge_platform import (
    CSAToMoEAdapter,
    EdgePlatformAPIBridge,
    EncryptionBridge,
    FederatedSyncService,
    IntegrationConfig,
)

# Initialize integration
config = IntegrationConfig.from_yaml("config/integration.yaml")

bridge = EdgePlatformAPIBridge(
    sil_registry_url=config.endpoints.sil_registry,
    edge_api_url=config.endpoints.edge_platform,
)

# Push CSA to Edge Platform
await bridge.push_csa_to_edge(csa_id="csa_123")

# Coordinated federated round
sync = FederatedSyncService(
    sil_coordinator_url=config.endpoints.sil_coordinator,
    edge_api_url=config.endpoints.edge_platform,
)

round_id = await sync.start_federated_round(
    skill_name="cooperative_assembly",
    num_sil_sites=3,
    num_edge_devices=2,
)
```

### Documentation

See **[docs/EDGE_PLATFORM_INTEGRATION.md](../docs/EDGE_PLATFORM_INTEGRATION.md)** for complete documentation including:
- Architecture overview
- Component details
- Usage scenarios
- API reference
- Troubleshooting

### Components

- **`adapters/`** - CSA ↔ MoE skill format conversion
- **`bridges/`** - API and encryption bridges between systems
- **`sync/`** - Federated synchronization service
- **`config/`** - Configuration management
- **`converters/`** - Data format converters

### Tests

```bash
pytest tests/integration/test_edge_platform_integration.py -v
```

### Status

✅ Production-ready (v1.0.0)
