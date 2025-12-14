# Deployment & Operations Runbook

## Quick Start

### Development Environment

```bash
# Clone repository
git clone https://github.com/Danielfoojunwei/Multi-actor.git
cd Multi-actor

# Install dependencies
pip install -e ".[dev]"

# Start services
make dev-up

# Run demo pipeline
make demo-round
```

### Production Deployment

See [Production Deployment](#production-deployment) section below.

## Architecture Overview

```
Registry (port 8080) ← Sites upload CSAs
    ↓
Swarm Coordinator (port 8081) ← Sites participate in rounds
    ↓
Runtime Nodes ← Sites deploy CSAs
```

## Development Environment

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- (Optional) ROS 2 Humble/Jazzy for native development
- (Optional) NVIDIA GPU for training

### Initial Setup

```bash
# 1. Install Python dependencies
pip install -e ".[dev]"

# 2. Install pre-commit hooks
pre-commit install

# 3. Start infrastructure
make dev-up
```

This starts:
- PostgreSQL (port 5432)
- CSA Registry (port 8080)
- Swarm Coordinator (port 8081)
- Prometheus (port 9090)
- Grafana (port 3000, admin/admin)

### Running Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# Full suite with coverage
make test
```

### Code Quality

```bash
# Lint
make lint

# Format
black ml/ swarm/ services/
ruff check --fix ml/ swarm/ services/
```

## Production Deployment

### Infrastructure Requirements

**Per Site**:
- 1x Compute server (GPU recommended for training)
  - 16+ CPU cores
  - 32GB+ RAM
  - 1x NVIDIA GPU (optional, for training)
  - 500GB+ storage
- 2-4x ONVIF cameras (1080p@30fps minimum)
- 1-3x Robot manipulators (e.g., UR5e, Franka Panda)
- Network: Gigabit Ethernet, <10ms latency between cameras

**Central**:
- 1x Registry server (4 cores, 8GB RAM, 1TB storage)
- 1x Swarm coordinator (8 cores, 16GB RAM)
- 1x PostgreSQL database (8 cores, 16GB RAM, 500GB SSD)

### Deployment Steps

#### 1. Prepare Infrastructure

```bash
# On each server, install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin

# Clone repository
git clone https://github.com/Danielfoojunwei/Multi-actor.git
cd Multi-actor
```

#### 2. Configure Environment

Create `.env` file:

```bash
# Registry
DATABASE_URL=postgresql://user:password@postgres:5432/csa_registry
SECRET_KEY=<generate with: openssl rand -hex 32>
ARTIFACT_STORAGE_PATH=/data/artifacts

# Swarm
REGISTRY_URL=https://registry.example.com
PRIVACY_MODE=ldp  # ldp, dp_sgd, he
PRIVACY_EPSILON=1.0
AGGREGATION_STRATEGY=trimmed_mean

# Signing keys (generate with: make generate-keys)
SIGNING_KEY_PATH=/secrets/signing_key.pem
SIGNING_PUBLIC_KEY_PATH=/secrets/signing_key.pub
```

#### 3. Generate Signing Keys

```bash
# On registry server
python3 -c "
from ml.artifact import CSASigner
from pathlib import Path

signer = CSASigner()
signer.save_private_key(Path('signing_key.pem'))
signer.save_public_key(Path('signing_key.pub'))
print('✓ Generated signing keys')
"

# Distribute public key to all sites
# Keep private key secure (use Vault/KMS in production)
```

#### 4. Start Services

**Central (Registry + Coordinator)**:

```bash
# Start with production compose file
docker-compose -f infra/docker/docker-compose.prod.yml up -d

# Verify services
curl http://localhost:8080/health  # Registry
curl http://localhost:8081/health  # Coordinator
```

**Site (Capture + Training + Runtime)**:

```bash
# Build images
make build-images

# Start ROS 2 runtime
docker run -d \
  --name ros2-runtime \
  --network host \
  --privileged \
  -v /dev:/dev \
  -e ROS_DOMAIN_ID=42 \
  dynamical-sil/ros2-runtime:latest

# Start ML training container (when needed)
docker run -d \
  --name ml-training \
  --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  dynamical-sil/ml-training:latest
```

## Operational Procedures

### Capture Demonstrations

```bash
# 1. Discover cameras
ros2 run swarm_capture discover_cameras

# 2. Start synchronized recording
ros2 launch swarm_capture multi_camera_record.launch.py \
  camera_ips:="['192.168.1.101', '192.168.1.102']" \
  output_bag:=/data/demo_001.db3

# 3. Perform demonstration (2-3 complete task executions)

# 4. Stop recording (Ctrl+C)

# 5. Verify rosbag
ros2 bag info /data/demo_001.db3
```

### Train Local Skill

```bash
# 1. Export demonstrations to training dataset
python3 ml/datasets/export_from_rosbag.py \
  --bag /data/demo_001.db3 \
  --output /data/cooperative_task.hdf5 \
  --format robomimic

# 2. Train cooperative BC policy
python3 ml/training/train_cooperative_bc.py \
  --config configs/training/cooperative_bc.yaml \
  --output-dir outputs/local_training

# 3. Package CSA
python3 ml/artifact/package_csa.py \
  --model-dir outputs/local_training \
  --output artifacts/cooperative_task_v1.0.0.tar.gz \
  --skill-name "cooperative_assembly" \
  --version "1.0.0"

# 4. Sign CSA
python3 ml/artifact/sign_csa.py \
  --artifact artifacts/cooperative_task_v1.0.0.tar.gz \
  --private-key signing_key.pem \
  --signer-id "site_a"
```

### Upload to Registry

```bash
curl -X POST http://registry.example.com/api/v1/csa/upload \
  -F "file=@artifacts/cooperative_task_v1.0.0.tar.gz" \
  -F "signature=@artifacts/cooperative_task_v1.0.0.tar.gz.sig" \
  -F "uploaded_by=site_a"

# Response:
# {
#   "id": 42,
#   "skill_name": "cooperative_assembly",
#   "version": "1.0.0",
#   "signature_verified": true
# }
```

### Participate in Swarm Round

```bash
# 1. Register with coordinator
python3 swarm/openfl/client.py register \
  --coordinator-url https://coordinator.example.com \
  --site-id site_a

# 2. Submit local CSA delta
python3 swarm/openfl/client.py submit \
  --round-id round_001 \
  --csa-delta artifacts/cooperative_task_v1.0.0.tar.gz

# 3. Wait for aggregation (coordinator notifies when complete)

# 4. Download merged CSA
python3 swarm/openfl/client.py download \
  --round-id round_001 \
  --output artifacts/merged_v1.1.0.tar.gz
```

### Deploy CSA to Runtime

```bash
# 1. Download from registry
curl -O http://registry.example.com/api/v1/csa/42/download

# 2. Verify signature
python3 -c "
from ml.artifact import CSAVerifier
from pathlib import Path

verifier = CSAVerifier()
valid, msg = verifier.verify_artifact(
    Path('cooperative_task_v1.1.0.tar.gz')
)
print(msg)
assert valid, 'Signature verification failed!'
"

# 3. Deploy to runtime
ros2 run swarm_skill_runtime deploy_csa \
  --csa-path artifacts/cooperative_task_v1.1.0.tar.gz \
  --skill-id cooperative_assembly

# 4. Execute skill
ros2 run swarm_skill_runtime execute_skill \
  --skill-id cooperative_assembly \
  --role leader  # or follower, spotter
```

### Rollback to Previous Version

```bash
# 1. Identify target version
curl http://registry.example.com/api/v1/csa/list?skill_name=cooperative_assembly

# 2. Trigger rollback
curl -X POST http://registry.example.com/api/v1/deployment/rollback \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": "site_a",
    "target_csa_id": 40
  }'

# 3. Reload runtime
ros2 service call /skill_runtime/reload \
  std_srvs/srv/Trigger
```

### Federated Unlearning

```bash
# 1. Submit unlearning request
python3 swarm/unlearning/request_unlearning.py \
  --site-id site_b \
  --reason "gdpr_request" \
  --current-csa-id 42

# 2. Unlearner processes request (generates new CSA version)

# 3. Download unlearned CSA
curl -O http://registry.example.com/api/v1/csa/43/download

# 4. Deploy unlearned version
# (same as "Deploy CSA to Runtime" above)
```

## Monitoring & Observability

### Prometheus Metrics

**Registry**:
- `csa_upload_total` - Total CSA uploads
- `csa_download_total` - Total downloads
- `deployment_total{status}` - Deployments by status
- `signature_verification_total{result}` - Signature checks

**Swarm Coordinator**:
- `swarm_round_duration_seconds` - Round duration
- `swarm_participants_count` - Participants per round
- `swarm_aggregation_duration_seconds` - Aggregation time
- `privacy_budget_spent{site_id}` - Cumulative epsilon

**Runtime**:
- `skill_execution_total{skill_id,status}` - Executions
- `safety_violations_total{type}` - Safety violations
- `phase_transition_duration_seconds{from,to}` - BT transitions

### Grafana Dashboards

Access: http://grafana.example.com:3000 (admin/admin)

**Dashboards**:
1. **System Overview**: All services health
2. **Swarm Rounds**: Participation, aggregation, privacy budgets
3. **CSA Registry**: Upload/download rates, storage
4. **Robot Runtime**: Execution success rate, safety events
5. **Privacy Accounting**: Cumulative epsilon per site

### Logs

**Centralized Logging** (optional, requires ELK stack):

```bash
# View registry logs
docker logs dynamical-sil-registry

# View coordinator logs
docker logs dynamical-sil-swarm

# Search for errors
docker logs dynamical-sil-registry 2>&1 | grep ERROR

# Follow logs
docker logs -f dynamical-sil-registry
```

**Structured Logging Format**:

```json
{
  "timestamp": "2025-01-19T10:30:45Z",
  "level": "INFO",
  "service": "registry",
  "correlation_id": "round_001",
  "message": "CSA uploaded",
  "csa_id": 42,
  "skill_name": "cooperative_assembly",
  "version": "1.0.0"
}
```

## Incident Response

### Incident: Poisoned CSA Detected

**Symptoms**:
- Skill execution success rate drops
- Safety violations increase
- Robot behavior anomalous

**Response**:

```bash
# 1. IMMEDIATE: Rollback to last known good
curl -X POST http://registry.example.com/api/v1/deployment/rollback \
  -d '{"site_id": "ALL", "target_csa_id": <last_good_id>}'

# 2. Quarantine suspected CSA
# (mark as inactive in database)

# 3. Investigate provenance
python3 swarm/unlearning/unlearner.py audit \
  --csa-id <suspected_id>

# 4. Notify affected sites
# (send alert via ops channel)

# 5. Re-aggregate without malicious site
python3 swarm/openfl/run_round.py \
  --exclude-site <malicious_site_id>

# 6. Post-mortem: Update aggregation strategy
# (e.g., lower trim_ratio, switch to Krum)
```

### Incident: Privacy Breach Suspected

**Symptoms**:
- Unusual network traffic from coordinator
- Logs show unexpected data access
- Privacy budget exceeded without rounds

**Response**:

```bash
# 1. IMMEDIATE: Halt all swarm rounds
docker stop dynamical-sil-swarm

# 2. Audit logs
grep -i "privacy\|gradient\|update" /var/log/swarm/*.log

# 3. Check privacy accounting
python3 swarm/privacy/audit_privacy.py --all-sites

# 4. Investigate coordinator
# (forensics, network capture)

# 5. Rotate keys if compromised
python3 scripts/rotate_keys.py

# 6. Notify affected sites + regulators (if GDPR)

# 7. Resume with stricter privacy mode
# (e.g., switch from LDP to HE)
```

### Incident: Registry Database Compromised

**Symptoms**:
- Unauthorized CSA modifications
- Provenance records altered
- Unknown deployments

**Response**:

```bash
# 1. IMMEDIATE: Take registry offline
docker stop dynamical-sil-registry

# 2. Restore from backup
pg_restore -d csa_registry /backups/latest.dump

# 3. Re-verify all CSA signatures
python3 scripts/verify_all_csas.py

# 4. Audit database access logs
# (identify unauthorized queries)

# 5. Rotate database credentials
# Update .env with new DATABASE_URL

# 6. Re-deploy registry with hardened config
# (tighter RBAC, audit logging)

# 7. Notify sites to re-verify deployed CSAs
```

### Incident: Safety Violation

**Symptoms**:
- Robot force/torque limits exceeded
- Collision detected
- Workspace bounds violated

**Response**:

```bash
# 1. IMMEDIATE: Emergency stop
ros2 topic pub /emergency_stop std_msgs/Bool "data: true"

# 2. Analyze logs
ros2 bag play --topics /joint_states /safety_monitor \
  <incident_bag>.db3

# 3. Identify root cause
# (skill execution error, sensor fault, CSA bug)

# 4. If CSA bug:
#    a. Quarantine CSA version
#    b. Update safety envelope
#    c. Re-run test suite

# 5. Update safety envelope in CSA manifest
python3 ml/artifact/update_safety_envelope.py \
  --csa <path> \
  --max-force <new_limit>

# 6. Deploy updated CSA

# 7. Post-mortem: Update test suite
```

## Backup & Recovery

### Database Backups

```bash
# Automated daily backups (add to cron)
0 2 * * * docker exec dynamical-sil-postgres \
  pg_dump -U dynamical csa_registry | \
  gzip > /backups/csa_registry_$(date +\%Y\%m\%d).sql.gz

# Restore from backup
gunzip < /backups/csa_registry_20250119.sql.gz | \
  docker exec -i dynamical-sil-postgres \
  psql -U dynamical csa_registry
```

### Artifact Backups

```bash
# Sync artifacts to S3 (daily)
aws s3 sync /app/artifacts s3://dynamical-sil-artifacts/ \
  --exclude "*.tmp"

# Restore from S3
aws s3 sync s3://dynamical-sil-artifacts/ /app/artifacts
```

### Configuration Backups

```bash
# Backup
tar czf config_backup_$(date +%Y%m%d).tar.gz \
  .env \
  configs/ \
  infra/docker/docker-compose.prod.yml

# Restore
tar xzf config_backup_20250119.tar.gz
```

## Performance Tuning

### Training Performance

```bash
# Enable mixed precision (2x speedup on GPU)
export CUDA_VISIBLE_DEVICES=0,1  # Multi-GPU
python3 ml/training/train_cooperative_bc.py \
  --mixed-precision \
  --batch-size 128  # Increase batch size

# Use multiple data workers
python3 ml/training/train_cooperative_bc.py \
  --num-workers 8
```

### Swarm Aggregation Performance

```bash
# Use sparse updates (10x bandwidth reduction)
python3 swarm/openfl/coordinator.py \
  --sparse-updates \
  --sparsity-threshold 0.01

# Async rounds (don't wait for stragglers)
python3 swarm/openfl/coordinator.py \
  --async-rounds \
  --staleness-tolerance 2
```

### Database Performance

```sql
-- Index commonly queried columns
CREATE INDEX idx_csa_skill_version ON csa_artifacts(skill_name, version);
CREATE INDEX idx_deployments_site_time ON deployments(site_id, deployed_at DESC);

-- Vacuum regularly
VACUUM ANALYZE csa_artifacts;
VACUUM ANALYZE deployments;
```

## Troubleshooting

### Problem: CSA Upload Fails with "Checksum Mismatch"

**Cause**: File corrupted during transfer

**Solution**:
```bash
# Re-download artifact
curl -O --retry 3 <csa_url>

# Verify checksum manually
sha256sum cooperative_task_v1.0.0.tar.gz
# Compare with expected checksum in manifest

# Re-upload
curl -X POST ... -F "file=@cooperative_task_v1.0.0.tar.gz"
```

### Problem: Swarm Round Times Out

**Cause**: Site offline or network latency

**Solution**:
```bash
# Check site connectivity
ping site_b.example.com

# Check swarm coordinator logs
docker logs dynamical-sil-swarm | grep "timeout\|site_b"

# Increase timeout
# Edit swarm_config.yaml:
# timeout_seconds: 7200  # 2 hours

# Or exclude slow site
python3 swarm/openfl/run_round.py --exclude-site site_b
```

### Problem: Privacy Budget Exceeded

**Cause**: Too many rounds with low epsilon

**Solution**:
```bash
# Check current budget
python3 swarm/privacy/audit_privacy.py --site-id site_a

# Increase epsilon (less privacy, more budget)
# Edit swarm_config.yaml:
# epsilon: 2.0  # was 1.0

# Or pause participation until budget resets
# (depends on privacy policy)
```

## Contacts

- **Security Issues**: security@example.com
- **Operations**: ops@example.com
- **On-Call**: +1-555-0100 (24/7)
- **GitHub Issues**: https://github.com/Danielfoojunwei/Multi-actor/issues
