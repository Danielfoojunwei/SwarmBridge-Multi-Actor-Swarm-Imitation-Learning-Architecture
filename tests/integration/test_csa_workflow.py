"""Integration tests for CSA packaging and signing"""

import pytest
import torch
from pathlib import Path
from datetime import datetime

from ml.artifact import (
    CSAPackager,
    CSALoader,
    CSASigner,
    CSAVerifier,
    CooperativeSkillArtefact,
    RoleConfig,
    RoleType,
    PolicyAdapter,
    CoordinationEncoder,
    SafetyEnvelope,
    CSAMetadata,
)


@pytest.fixture
def sample_csa():
    """Create a sample CSA for testing"""
    roles = [
        RoleConfig(
            role_id="leader",
            role_type=RoleType.LEADER,
            observation_dims=10,
            action_dims=7,
            requires_coordination=True,
        ),
        RoleConfig(
            role_id="follower",
            role_type=RoleType.FOLLOWER,
            observation_dims=10,
            action_dims=7,
            requires_coordination=True,
        ),
    ]

    adapters = [
        PolicyAdapter(
            role_id="leader",
            adapter_type="lora",
            adapter_weights={
                "lora_A": torch.randn(16, 256),
                "lora_B": torch.randn(7, 16),
            },
        ),
        PolicyAdapter(
            role_id="follower",
            adapter_type="lora",
            adapter_weights={
                "lora_A": torch.randn(16, 256),
                "lora_B": torch.randn(7, 16),
            },
        ),
    ]

    encoder = CoordinationEncoder(
        encoder_type="transformer",
        encoder_weights={
            "layer_0": torch.randn(256, 256),
            "layer_1": torch.randn(256, 256),
        },
        latent_dim=64,
        sequence_length=16,
        fusion_strategy="attention",
    )

    phase_machine_xml = """<?xml version="1.0"?>
<root main_tree_to_execute="CooperativeAssembly">
    <BehaviorTree ID="CooperativeAssembly">
        <Sequence name="main_sequence">
            <Action ID="Approach" name="approach_object"/>
            <Action ID="Grasp" name="grasp_object"/>
            <Parallel success_threshold="2">
                <Action ID="Lift" name="lift_together"/>
                <Action ID="Monitor" name="monitor_force"/>
            </Parallel>
            <Action ID="Transfer" name="transfer_object"/>
            <Action ID="Place" name="place_object"/>
            <Action ID="Retreat" name="retreat_safe"/>
        </Sequence>
    </BehaviorTree>
</root>"""

    safety_envelope = SafetyEnvelope(
        max_velocity={
            "joint_0": 1.5,
            "joint_1": 1.5,
            "joint_2": 1.5,
            "cartesian": 0.5,
        },
        max_acceleration={
            "joint_0": 3.0,
            "joint_1": 3.0,
            "joint_2": 3.0,
        },
        max_force={"gripper": 50.0, "end_effector": 100.0},
        max_torque={
            "joint_0": 15.0,
            "joint_1": 15.0,
            "joint_2": 10.0,
        },
        min_separation_distance=0.5,
        workspace_bounds=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
        collision_primitives=[
            {"type": "sphere", "radius": 0.1, "center": [0.0, 0.0, 1.0]}
        ],
        emergency_stop_triggers=[
            "force_limit_exceeded",
            "workspace_violation",
            "collision_detected",
            "coordination_lost",
        ],
    )

    metadata = CSAMetadata(
        version="1.0.0",
        skill_name="cooperative_assembly",
        description="Two-actor cooperative assembly task with force coordination",
        num_demonstrations=3,
        training_sites=["site_a"],
        training_duration_seconds=150.0,
        base_model="robomimic_bc_rnn",
        compatible_robots=["ur5e", "franka_panda"],
        compatible_end_effectors=["robotiq_2f85", "schunk_svh"],
        min_actors=2,
        max_actors=2,
        privacy_mode="none",
        epsilon=None,
        delta=None,
        test_pass_rate=0.95,
        test_coverage=0.85,
    )

    csa = CooperativeSkillArtefact(
        roles=roles,
        policy_adapters=adapters,
        coordination_encoder=encoder,
        phase_machine_xml=phase_machine_xml,
        safety_envelope=safety_envelope,
        metadata=metadata,
        test_suite={
            "test_coordination": {"type": "unit", "status": "passed"},
            "test_safety": {"type": "integration", "status": "passed"},
        },
    )

    return csa


def test_csa_packaging_and_loading(sample_csa, tmp_path):
    """Test full packaging and loading cycle"""
    # Package
    packager = CSAPackager(output_dir=tmp_path)
    package_path = packager.package(sample_csa, output_name="test_csa.tar.gz")

    assert package_path.exists()
    assert package_path.stat().st_size > 0
    print(f"✓ Packaged CSA: {package_path.stat().st_size} bytes")

    # Load
    loader = CSALoader()
    loaded_csa = loader.load(package_path, verify_checksums=True)

    # Verify
    assert loaded_csa.metadata.skill_name == sample_csa.metadata.skill_name
    assert loaded_csa.metadata.version == sample_csa.metadata.version
    assert len(loaded_csa.roles) == len(sample_csa.roles)
    assert len(loaded_csa.policy_adapters) == len(sample_csa.policy_adapters)

    print(f"✓ Loaded CSA: {loaded_csa.metadata.skill_name} v{loaded_csa.metadata.version}")


def test_csa_signing_and_verification(sample_csa, tmp_path):
    """Test cryptographic signing and verification"""
    # Package CSA
    packager = CSAPackager(output_dir=tmp_path)
    package_path = packager.package(sample_csa)

    # Sign
    signer = CSASigner()
    sig_path = signer.sign_artifact(package_path, signer_id="test_site")

    assert sig_path.exists()
    print(f"✓ Signed CSA: {sig_path}")

    # Verify
    verifier = CSAVerifier()
    is_valid, message = verifier.verify_artifact(package_path, sig_path)

    assert is_valid
    print(f"✓ Signature verified: {message}")


def test_csa_test_suite_execution(sample_csa):
    """Test running CSA test suite"""
    all_passed, results = sample_csa.run_test_suite()

    print(f"\n✓ Test Suite Results:")
    for test_name, result in results.items():
        print(f"  - {test_name}: {result}")

    # Should have basic tests passing
    assert results["role_adapter_mapping"] is True
    assert results["safety_envelope_complete"] is True


def test_csa_compatibility_check(sample_csa):
    """Test hardware compatibility checking"""
    # Compatible hardware
    assert sample_csa.validate_compatibility("ur5e", "robotiq_2f85") is True
    assert sample_csa.validate_compatibility("franka_panda", "schunk_svh") is True

    # Incompatible hardware
    assert sample_csa.validate_compatibility("unknown_robot", "robotiq_2f85") is False
    assert sample_csa.validate_compatibility("ur5e", "unknown_gripper") is False

    print("✓ Compatibility checks working")


def test_safety_envelope_violations(sample_csa):
    """Test safety envelope violation detection"""
    import numpy as np

    # Safe state
    safe_pos = np.array([0.0, 0.0, 1.0])
    safe_vel = np.array([0.5, 0.5, 0.5])
    is_safe, violations = sample_csa.safety_envelope.validate_state(safe_pos, safe_vel)

    assert is_safe
    assert len(violations) == 0
    print("✓ Safe state detected correctly")

    # Unsafe state (velocity exceeded)
    unsafe_vel = np.array([2.0, 0.5, 0.5])  # Exceeds max_velocity
    is_safe, violations = sample_csa.safety_envelope.validate_state(safe_pos, unsafe_vel)

    assert not is_safe
    assert len(violations) > 0
    print(f"✓ Velocity violation detected: {violations}")

    # Unsafe state (workspace violation)
    unsafe_pos = np.array([2.0, 0.0, 1.0])  # Outside workspace
    is_safe, violations = sample_csa.safety_envelope.validate_state(unsafe_pos, safe_vel)

    assert not is_safe
    assert len(violations) > 0
    print(f"✓ Workspace violation detected: {violations}")
