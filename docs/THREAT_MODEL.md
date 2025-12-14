# Threat Model

## Overview

This document analyzes security threats to the Dynamical-SIL system and describes mitigations.

## Assets

### Critical Assets
1. **Raw demonstration data**: Multi-actor video + sensor streams (privacy-sensitive)
2. **Trained CSA artifacts**: Cooperative skill policies (IP/competitive advantage)
3. **Registry database**: Provenance, deployment history (integrity-critical)
4. **Cryptographic keys**: Signing keys, HE keys (confidentiality-critical)

### Valuable Assets
5. **Training infrastructure**: Compute resources (availability)
6. **Runtime robots**: Physical safety (safety-critical)

## Adversaries

### A1: Honest-but-Curious Coordinator

**Capabilities**:
- Receives CSA deltas from all sites
- Controls aggregation process
- Has network-level visibility

**Goals**:
- Infer raw demonstration data from updates
- Identify which site contributed specific skills

**Mitigations**:
- ✅ **Local Differential Privacy (LDP)**: Add noise before sending updates
- ✅ **DP-SGD**: Clip gradients + add noise with formal privacy guarantees
- ✅ **Homomorphic Encryption**: Aggregate in encrypted space
- ✅ **Sparse updates**: Only send changed parameters (reduce information leakage)

**Residual Risk**: LOW (with privacy modes enabled)

### A2: Malicious Site (Poisoning Attack)

**Capabilities**:
- Participate in swarm rounds
- Send arbitrarily crafted CSA deltas
- Collude with other malicious sites (up to k/n sites)

**Goals**:
- Degrade merged CSA performance (sabotage)
- Inject backdoors (trigger-based misbehavior)
- Cause safety violations

**Attack Vectors**:
1. **Gradient poisoning**: Send large-magnitude updates to skew aggregation
2. **Backdoor injection**: Embed trigger patterns that cause specific failures
3. **Data poisoning**: Train on adversarial demonstrations

**Mitigations**:
- ✅ **Robust aggregation**: Trimmed mean (Byzantine-resilient up to k/n outliers)
- ✅ **Krum**: Select most representative update (tolerates f < n/2 - 1 malicious)
- ✅ **Norm clipping**: Bound update magnitude
- ✅ **CSA test suite**: Deterministic offline validation before acceptance
- ✅ **Safety envelope**: Runtime constraints reject unsafe actions
- ⚠️ **Anomaly detection**: Monitor update distributions (optional, not implemented)

**Residual Risk**: MEDIUM (robust aggregation provides defense, but sophisticated attacks possible)

### A3: Network Eavesdropper

**Capabilities**:
- Passive monitoring of network traffic
- Cannot modify packets (MitM assumed mitigated by TLS)

**Goals**:
- Intercept CSA artifacts
- Infer training data from network patterns

**Mitigations**:
- ✅ **TLS 1.3**: Encrypt all HTTP traffic (registry, coordinator)
- ✅ **Certificate pinning**: Prevent MitM (optional)
- ✅ **Homomorphic Encryption**: CSA deltas encrypted end-to-end (in HE mode)

**Residual Risk**: LOW (TLS provides strong confidentiality)

### A4: Insider (Registry Access)

**Capabilities**:
- Authenticated access to registry
- Can read CSA metadata, download artifacts
- May have database access

**Goals**:
- Tamper with CSA artifacts (integrity attack)
- Delete/rollback unauthorized versions
- Steal proprietary skills

**Attack Vectors**:
1. **Artifact tampering**: Modify CSA tarball, re-upload
2. **Database manipulation**: Alter provenance records
3. **Unauthorized rollback**: Deploy old vulnerable CSA

**Mitigations**:
- ✅ **Cryptographic signing**: CSAs signed with private key, verified before deployment
- ✅ **Checksums (SHA256)**: Detect tampered files
- ✅ **RBAC**: Role-based access control (read-only vs. admin)
- ✅ **Audit logs**: All operations logged with timestamps + user IDs
- ✅ **Immutable artifacts**: Once uploaded, CSAs cannot be modified (only new versions)
- ⚠️ **Secret management**: Use Vault/KMS for private keys (operational, not in code)

**Residual Risk**: MEDIUM (signing prevents tampering, but insider can still steal/delete)

### A5: Physical Attacker (Robot Safety)

**Capabilities**:
- Manipulate robot environment
- Cause unexpected states (e.g., place obstacle)

**Goals**:
- Cause robot to violate safety constraints
- Injure humans or damage property

**Mitigations**:
- ✅ **Safety envelope**: Velocity, force, torque, workspace bounds
- ✅ **Separation distance**: Minimum actor-to-actor distance enforced
- ✅ **Emergency stop**: Hardware e-stop + software abort triggers
- ✅ **BehaviorTree abort logic**: Phase machine can transition to abort state
- ✅ **MoveIt2 collision avoidance**: Real-time collision checking

**Residual Risk**: LOW (defense-in-depth for safety)

## Attack Scenarios

### Scenario 1: Gradient Poisoning to Inject Backdoor

**Attacker**: Malicious Site B
**Target**: Merged CSA deployed to all sites

**Attack Steps**:
1. Site B trains on poisoned demonstrations (e.g., drop object when see red marker)
2. Site B sends CSA delta with backdoor embedded
3. Coordinator aggregates with legitimate sites A, C
4. Merged CSA contains subtle backdoor

**Defense**:
1. **Robust aggregation (Krum)**: Detects Site B's update as outlier if >1 honest sites
2. **CSA test suite**: Run deterministic tests (does *not* cover trigger-specific behaviors)
3. **Manual inspection**: Adversarial evaluation on trigger dataset (operational)

**Outcome**: MITIGATED if robust aggregation used + manual review

### Scenario 2: Privacy Leakage via Update Correlation

**Attacker**: Honest-but-curious coordinator
**Target**: Site A's raw demonstration data

**Attack Steps**:
1. Coordinator receives CSA delta from Site A (multiple rounds)
2. Analyze gradient patterns to infer demonstration trajectories
3. Reconstruct approximate joint angles / end-effector poses

**Defense**:
1. **LDP**: Add Laplace noise (ε=1.0) → provable ε-LDP
2. **DP-SGD**: Clip + Gaussian noise → (ε=2.0, δ=1e-5)-DP
3. **Privacy accounting**: Track cumulative privacy budget across rounds

**Outcome**: MITIGATED (formal DP guarantees bound leakage)

### Scenario 3: Unauthorized CSA Rollback

**Attacker**: Insider with registry admin access
**Target**: Deploy old vulnerable CSA version

**Attack Steps**:
1. Insider identifies old CSA with known safety issue (e.g., excessive force)
2. Trigger rollback API to deploy old CSA to production site
3. Robot executes unsafe behavior

**Defense**:
1. **RBAC**: Restrict rollback permission to authorized operators
2. **Approval workflow**: Require multi-party approval for rollbacks
3. **Audit logs**: Record rollback with requester ID + timestamp
4. **Version pinning**: Sites can configure "minimum safe version"

**Outcome**: MITIGATED via access control + audit trail

### Scenario 4: Supply Chain Attack (Compromised Dependency)

**Attacker**: External (compromise robomimic/LeRobot/OpenFL)
**Target**: Poison training or aggregation logic

**Attack Steps**:
1. Attacker compromises PyPI package (e.g., malicious robomimic==0.3.1)
2. System installs compromised package
3. Training injects backdoor or exfiltrates data

**Defense**:
1. **Dependency pinning**: Use lockfiles (poetry.lock)
2. **SBOM generation**: Track all dependencies + versions
3. **Vulnerability scanning**: GitHub Dependabot, Snyk
4. **Hash verification**: Verify package hashes against known-good

**Outcome**: PARTIALLY MITIGATED (supply chain risk always present)

## Threat Summary Matrix

| Threat | Likelihood | Impact | Mitigation | Residual Risk |
|--------|-----------|--------|------------|---------------|
| Privacy leakage (A1) | High | High | LDP/DP-SGD/HE | Low |
| Poisoning attack (A2) | Medium | High | Robust agg + tests | Medium |
| Network eavesdrop (A3) | Low | Medium | TLS | Low |
| Insider tampering (A4) | Low | High | Signing + RBAC | Medium |
| Physical safety (A5) | Medium | Critical | Safety envelope | Low |
| Supply chain (A4) | Low | High | SBOM + scanning | Medium |

## Compliance

### GDPR
- ✅ **Right to be forgotten**: Federated unlearning
- ✅ **Data minimization**: Only share CSA deltas, not raw data
- ✅ **Purpose limitation**: CSAs tagged with privacy mode
- ✅ **Transparency**: Provenance tracking

### ISO 27001 (Information Security)
- ✅ **Access control**: RBAC for registry
- ✅ **Cryptography**: TLS, signing, HE
- ✅ **Audit logging**: All operations logged
- ⚠️ **Incident response**: Procedures documented in RUNBOOK.md

### ISO/TS 15066 (Collaborative Robots)
- ✅ **Force/torque limits**: Safety envelope
- ✅ **Separation distance**: Enforced runtime
- ✅ **Emergency stop**: Hardware + software
- ✅ **Risk assessment**: Safety envelope per skill

## Recommendations

### For Operators
1. **Enable privacy mode**: Use LDP (ε=1.0) or DP-SGD (ε=2.0, δ=1e-5) for untrusted coordinator
2. **Use robust aggregation**: Trimmed mean (trim_ratio=0.1) or Krum
3. **Verify signatures**: Always check CSA signatures before deployment
4. **Monitor audit logs**: Review registry access logs weekly
5. **Secret management**: Store signing keys in Vault/KMS, never in code
6. **Network isolation**: Run coordinator in separate network segment
7. **Rate limiting**: Limit swarm round frequency to prevent DoS

### For Developers
1. **No secrets in code**: Use environment variables + secret managers
2. **Dependency pinning**: Commit lockfiles, verify hashes
3. **Input validation**: Sanitize all external inputs (CSA uploads, API calls)
4. **Least privilege**: Services run as non-root, minimal permissions
5. **Regular updates**: Patch dependencies, scan for CVEs
6. **Penetration testing**: Annual security audit

### For Sites
1. **Isolate demonstration data**: Keep raw data local, never upload to coordinator
2. **Review CSA test suite**: Ensure tests cover safety-critical behaviors
3. **Gradual rollout**: Deploy new CSAs to test robots first
4. **Backup old CSAs**: Maintain last-known-good versions for rollback
5. **Monitor robot behavior**: Continuous safety monitoring in production

## Incident Response

See [RUNBOOK.md](RUNBOOK.md#incident-response) for detailed procedures.

**Quick Reference**:
- **Poisoned CSA detected**: Immediately rollback, quarantine CSA, notify sites
- **Privacy breach suspected**: Halt swarm rounds, audit logs, investigate coordinator
- **Safety violation**: Emergency stop, analyze logs, update safety envelope
- **Registry compromise**: Rotate signing keys, re-verify all CSAs, audit database

## References

### Privacy
- Zhao et al. (2020) "Local Differential Privacy-based Federated Learning for Internet of Things"
- Abadi et al. (2016) "Deep Learning with Differential Privacy"
- NTU DR: "FHE-Enabled Cloud-Edge Collaborative Computing Architecture"

### Robustness
- Blanchard et al. (2017) "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
- Yin et al. (2018) "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"

### Unlearning
- Bourtoule et al. (2021) "Machine Unlearning"
- DTC Federated Unlearning Publications

### Safety
- ISO/TS 15066:2016 "Robots and robotic devices — Collaborative robots"
- ISO 10218-1:2011 "Robots and robotic devices — Safety requirements for industrial robots"
