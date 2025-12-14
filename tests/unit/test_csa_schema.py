"""Unit tests for CSA schema"""

import pytest
import torch
from datetime import datetime

from ml.artifact.schema import (
    CooperativeSkillArtefact,
    RoleConfig,
    RoleType,
    PolicyAdapter,
    CoordinationEncoder,
    SafetyEnvelope,
    CSAMetadata,
)


def test_role_config_creation():
    """Test RoleConfig initialization"""
    role = RoleConfig(
        role_id="leader",
        role_type=RoleType.LEADER,
        observation_dims=10,
        action_dims=7,
        requires_coordination=True,
    )

    assert role.role_id == "leader"
    assert role.role_type == RoleType.LEADER
    assert role.observation_dims == 10
    assert role.action_dims == 7


def test_policy_adapter_save_load(tmp_path):
    """Test PolicyAdapter save/load"""
    adapter = PolicyAdapter(
        role_id="leader",
        adapter_type="lora",
        adapter_weights={
            "lora_A": torch.randn(16, 256),
            "lora_B": torch.randn(7, 16),
        },
    )

    # Save
    save_path = tmp_path / "adapter.pt"
    adapter.save(save_path)

    # Load
    loaded_adapter = PolicyAdapter.load(save_path)

    assert loaded_adapter.role_id == adapter.role_id
    assert loaded_adapter.adapter_type == adapter.adapter_type
    assert torch.allclose(
        loaded_adapter.adapter_weights["lora_A"], adapter.adapter_weights["lora_A"]
    )


def test_coordination_encoder_save_load(tmp_path):
    """Test CoordinationEncoder save/load"""
    encoder = CoordinationEncoder(
        encoder_type="transformer",
        encoder_weights={"layer_0": torch.randn(256, 256)},
        latent_dim=64,
        sequence_length=16,
    )

    # Save
    save_path = tmp_path / "encoder.pt"
    encoder.save(save_path)

    # Load
    loaded_encoder = CoordinationEncoder.load(save_path)

    assert loaded_encoder.encoder_type == encoder.encoder_type
    assert loaded_encoder.latent_dim == encoder.latent_dim


def test_safety_envelope_validation():
    """Test SafetyEnvelope state validation"""
    envelope = SafetyEnvelope(
        max_velocity={"joint_0": 1.0, "joint_1": 1.0},
        max_acceleration={"joint_0": 2.0},
        max_force={"gripper": 50.0},
        max_torque={"joint_0": 10.0},
        min_separation_distance=0.3,
        workspace_bounds=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
        collision_primitives=[],
        emergency_stop_triggers=["force_limit"],
    )

    # Valid state
    positions = torch.tensor([0.5, 0.0, 1.0])
    velocities = torch.tensor([0.5, 0.5, 0.0])
    is_safe, violations = envelope.validate_state(positions, velocities)
    assert is_safe
    assert len(violations) == 0

    # Invalid state (out of workspace)
    positions = torch.tensor([2.0, 0.0, 1.0])  # x too large
    is_safe, violations = envelope.validate_state(positions, velocities)
    assert not is_safe
    assert len(violations) > 0


def test_csa_metadata_version_validation():
    """Test CSAMetadata semantic versioning"""
    # Valid version
    metadata = CSAMetadata(
        version="1.0.0",
        skill_name="test_skill",
        description="Test",
        num_demonstrations=5,
        training_sites=["site_a"],
        training_duration_seconds=100.0,
        compatible_robots=["ur5e"],
        compatible_end_effectors=["robotiq_2f85"],
        min_actors=2,
        max_actors=2,
        privacy_mode="none",
        test_pass_rate=0.95,
        test_coverage=0.80,
    )
    assert metadata.version == "1.0.0"

    # Invalid version
    with pytest.raises(ValueError):
        CSAMetadata(
            version="1.0",  # Missing patch version
            skill_name="test_skill",
            description="Test",
            num_demonstrations=5,
            training_sites=["site_a"],
            training_duration_seconds=100.0,
            compatible_robots=["ur5e"],
            compatible_end_effectors=["robotiq_2f85"],
            min_actors=2,
            max_actors=2,
            privacy_mode="none",
            test_pass_rate=0.95,
            test_coverage=0.80,
        )


def test_csa_run_test_suite():
    """Test CSA test suite execution"""
    # Create minimal CSA
    roles = [
        RoleConfig(
            role_id="leader",
            role_type=RoleType.LEADER,
            observation_dims=10,
            action_dims=7,
        )
    ]

    adapters = [
        PolicyAdapter(
            role_id="leader",
            adapter_type="linear",
            adapter_weights={"weight": torch.randn(7, 10)},
        )
    ]

    encoder = CoordinationEncoder(
        encoder_type="mlp",
        encoder_weights={"layer_0": torch.randn(64, 32)},
        latent_dim=64,
        sequence_length=16,
    )

    safety_envelope = SafetyEnvelope(
        max_velocity={"joint_0": 1.0},
        max_acceleration={"joint_0": 2.0},
        max_force={"gripper": 50.0},
        max_torque={"joint_0": 10.0},
        min_separation_distance=0.3,
        workspace_bounds=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
        collision_primitives=[],
        emergency_stop_triggers=["force_limit"],
    )

    metadata = CSAMetadata(
        version="1.0.0",
        skill_name="test_skill",
        description="Test CSA",
        num_demonstrations=3,
        training_sites=["site_a"],
        training_duration_seconds=100.0,
        compatible_robots=["ur5e"],
        compatible_end_effectors=["robotiq_2f85"],
        min_actors=1,
        max_actors=1,
        privacy_mode="none",
        test_pass_rate=0.95,
        test_coverage=0.80,
    )

    csa = CooperativeSkillArtefact(
        roles=roles,
        policy_adapters=adapters,
        coordination_encoder=encoder,
        phase_machine_xml="<root><BehaviorTree><Action/></BehaviorTree></root>",
        safety_envelope=safety_envelope,
        metadata=metadata,
    )

    # Run test suite
    all_passed, results = csa.run_test_suite()

    # Should pass basic tests
    assert results["role_adapter_mapping"] is True
    assert results["safety_envelope_complete"] is True
