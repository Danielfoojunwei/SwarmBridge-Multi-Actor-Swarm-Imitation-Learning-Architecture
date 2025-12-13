"""Tests for advanced multi-actor system"""

import pytest
import torch
import numpy as np

from ml.training.advanced_multi_actor import (
    IntentCommunicationModule,
    DynamicRoleAssigner,
    HierarchicalCoordinationEncoder,
    AdaptiveCoordinationPolicy,
    MultiActorSafetyVerifier,
    MultiActorState,
    ActorIntent,
    CoordinationMode,
)


def test_intent_communication_module():
    """Test intent communication and prediction"""
    num_actors = 3
    batch_size = 4
    state_dim = 10
    num_intent_types = len(ActorIntent)

    module = IntentCommunicationModule(
        num_actors=num_actors,
        intent_dim=32,
    )

    # Create sample inputs
    actor_states = torch.randn(batch_size, num_actors, state_dim)
    actor_intents = torch.zeros(batch_size, num_actors, num_intent_types)
    actor_intents[:, :, 0] = 1.0  # All actors have "grasp" intent

    # Forward pass
    intent_embeds, predicted_intents = module(actor_states, actor_intents)

    # Check shapes
    assert intent_embeds.shape == (batch_size, num_actors, 32)
    assert predicted_intents.shape == (batch_size, num_actors, num_intent_types)

    print("✓ Intent communication module working")


def test_dynamic_role_assigner():
    """Test dynamic role assignment"""
    num_actors = 4
    num_roles = 3
    batch_size = 2
    capability_dim = 16
    task_dim = 32

    assigner = DynamicRoleAssigner(
        num_actors=num_actors,
        num_roles=num_roles,
        capability_dim=capability_dim,
        task_embedding_dim=task_dim,
    )

    # Create sample inputs
    capabilities = torch.randn(batch_size, num_actors, capability_dim)
    task_requirements = torch.randn(batch_size, task_dim)

    # Get assignment matrix
    assignment_matrix = assigner(capabilities, task_requirements)

    # Check shape
    assert assignment_matrix.shape == (batch_size, num_actors, num_roles)

    # Check probabilities sum to 1 for each actor
    role_probs_sum = assignment_matrix.sum(dim=-1)
    assert torch.allclose(role_probs_sum, torch.ones_like(role_probs_sum), atol=1e-5)

    print("✓ Dynamic role assignment working")
    print(f"  Sample assignment:\n{assignment_matrix[0]}")


def test_hierarchical_coordination_encoder():
    """Test hierarchical coordination encoding"""
    num_actors = 3
    obs_dim = 15
    latent_dim = 64
    batch_size = 8

    encoder = HierarchicalCoordinationEncoder(
        num_actors=num_actors,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
    )

    # Create sample observations
    observations = torch.randn(batch_size, num_actors, obs_dim)

    # Encode
    outputs = encoder(observations)

    # Check all levels present
    assert "individual" in outputs
    assert "pairwise" in outputs
    assert "global" in outputs
    assert "fused" in outputs
    assert "attention_weights" in outputs

    # Check shapes
    assert outputs["individual"].shape == (batch_size, num_actors, latent_dim)
    assert outputs["pairwise"].shape == (batch_size, num_actors, latent_dim)
    assert outputs["global"].shape == (batch_size, num_actors, latent_dim)
    assert outputs["fused"].shape == (batch_size, num_actors, latent_dim)

    print("✓ Hierarchical coordination encoder working")
    print(f"  Individual latent: {outputs['individual'].shape}")
    print(f"  Pairwise latent: {outputs['pairwise'].shape}")
    print(f"  Global latent: {outputs['global'].shape}")
    print(f"  Fused latent: {outputs['fused'].shape}")


def test_adaptive_coordination_policy():
    """Test adaptive policy with mode switching"""
    role_configs = {
        "leader": 7,
        "follower_1": 7,
        "follower_2": 7,
    }
    coordination_latent_dim = 64
    batch_size = 4

    policy = AdaptiveCoordinationPolicy(
        role_configs=role_configs,
        coordination_latent_dim=coordination_latent_dim,
        num_coordination_modes=len(CoordinationMode),
    )

    # Create sample coordination latent
    coord_latent = torch.randn(batch_size, coordination_latent_dim)

    # Forward pass
    outputs = policy(coord_latent)

    # Check outputs
    assert "mode_probs" in outputs
    assert "actions" in outputs
    assert "uncertainty" in outputs

    # Check mode probabilities
    assert outputs["mode_probs"].shape == (batch_size, len(CoordinationMode))
    assert torch.allclose(
        outputs["mode_probs"].sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-5
    )

    # Check actions for all roles and modes
    for role_id, action_dim in role_configs.items():
        assert role_id in outputs["actions"]
        for mode in CoordinationMode:
            assert mode.value in outputs["actions"][role_id]
            assert outputs["actions"][role_id][mode.value].shape == (batch_size, action_dim)

    # Check uncertainty
    assert outputs["uncertainty"].shape == (batch_size, 1)
    assert (outputs["uncertainty"] >= 0).all()  # Uncertainty should be non-negative

    print("✓ Adaptive coordination policy working")
    print(f"  Mode probabilities: {outputs['mode_probs'][0]}")
    print(f"  Uncertainty: {outputs['uncertainty'][0].item():.4f}")


def test_multi_actor_safety_verifier():
    """Test comprehensive safety verification"""
    verifier = MultiActorSafetyVerifier(
        min_separation=0.5,
        max_relative_velocity=0.3,
        formation_tolerance=0.1,
    )

    # Test 1: Safe state
    safe_state = MultiActorState(
        actor_positions={
            "actor_0": np.array([0.0, 0.0, 1.0]),
            "actor_1": np.array([1.0, 0.0, 1.0]),
            "actor_2": np.array([0.5, 1.0, 1.0]),
        },
        actor_velocities={
            "actor_0": np.array([0.1, 0.0, 0.0]),
            "actor_1": np.array([0.1, 0.0, 0.0]),
            "actor_2": np.array([0.0, 0.1, 0.0]),
        },
        actor_intents={
            "actor_0": ActorIntent.GRASP,
            "actor_1": ActorIntent.SUPPORT,
            "actor_2": ActorIntent.MONITOR,
        },
        coordination_mode=CoordinationMode.HIERARCHICAL,
    )

    is_safe, violations, metrics = verifier.verify_state(safe_state)

    assert is_safe
    assert len(violations) == 0
    assert metrics["min_separation"] >= 0.5
    print("✓ Safe state verified")

    # Test 2: Unsafe state (actors too close)
    unsafe_state = MultiActorState(
        actor_positions={
            "actor_0": np.array([0.0, 0.0, 1.0]),
            "actor_1": np.array([0.2, 0.0, 1.0]),  # Too close!
            "actor_2": np.array([0.5, 1.0, 1.0]),
        },
        actor_velocities={
            "actor_0": np.array([0.1, 0.0, 0.0]),
            "actor_1": np.array([0.1, 0.0, 0.0]),
            "actor_2": np.array([0.0, 0.1, 0.0]),
        },
        actor_intents={
            "actor_0": ActorIntent.MOVE,
            "actor_1": ActorIntent.MOVE,
            "actor_2": ActorIntent.WAIT,
        },
        coordination_mode=CoordinationMode.PEER_TO_PEER,
    )

    is_safe, violations, metrics = verifier.verify_state(unsafe_state)

    assert not is_safe
    assert len(violations) > 0
    assert metrics["min_separation"] < 0.5
    print("✓ Unsafe state (collision) detected")
    print(f"  Violations: {violations}")

    # Test 3: Conflicting intents
    conflict_state = MultiActorState(
        actor_positions={
            "actor_0": np.array([0.0, 0.0, 1.0]),
            "actor_1": np.array([1.0, 0.0, 1.0]),
        },
        actor_velocities={
            "actor_0": np.array([0.1, 0.0, 0.0]),
            "actor_1": np.array([0.1, 0.0, 0.0]),
        },
        actor_intents={
            "actor_0": ActorIntent.GRASP,
            "actor_1": ActorIntent.GRASP,  # Both trying to grasp!
        },
        coordination_mode=CoordinationMode.CONSENSUS,
    )

    is_safe, violations, metrics = verifier.verify_state(conflict_state)

    assert not is_safe
    assert any("grasp" in v.lower() for v in violations)
    print("✓ Intent conflict detected")
    print(f"  Violations: {violations}")


def test_integration_hierarchical_to_adaptive():
    """Test integration of hierarchical encoder with adaptive policy"""
    num_actors = 3
    obs_dim = 12
    action_dim = 7
    latent_dim = 64
    batch_size = 4

    # Build components
    encoder = HierarchicalCoordinationEncoder(
        num_actors=num_actors,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
    )

    role_configs = {f"actor_{i}": action_dim for i in range(num_actors)}

    policy = AdaptiveCoordinationPolicy(
        role_configs=role_configs,
        coordination_latent_dim=latent_dim,
        num_coordination_modes=len(CoordinationMode),
    )

    # Create sample data
    observations = torch.randn(batch_size, num_actors, obs_dim)

    # Forward pass through encoder
    coord_outputs = encoder(observations)
    fused_latent = coord_outputs["fused"]

    # Average over actors to get single latent per sample
    global_latent = fused_latent.mean(dim=1)

    # Forward through policy
    policy_outputs = policy(global_latent)

    # Verify we can execute actions for all roles
    for role_id in role_configs.keys():
        assert role_id in policy_outputs["actions"]

        # Select action based on mode probabilities
        mode_probs = policy_outputs["mode_probs"]
        selected_mode_idx = mode_probs.argmax(dim=-1)

        # Get action for selected mode
        for i, mode in enumerate(CoordinationMode):
            if i == selected_mode_idx[0].item():
                action = policy_outputs["actions"][role_id][mode.value]
                assert action.shape == (batch_size, action_dim)

    print("✓ End-to-end hierarchical encoder → adaptive policy working")


def test_curriculum_learning_actor_count():
    """Test model can handle variable number of actors"""
    obs_dim = 10
    latent_dim = 32

    # Test with 2 actors
    encoder_2 = HierarchicalCoordinationEncoder(
        num_actors=2,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
    )

    obs_2 = torch.randn(4, 2, obs_dim)
    output_2 = encoder_2(obs_2)
    assert output_2["fused"].shape == (4, 2, latent_dim)
    print("✓ Works with 2 actors")

    # Test with 4 actors
    encoder_4 = HierarchicalCoordinationEncoder(
        num_actors=4,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
    )

    obs_4 = torch.randn(4, 4, obs_dim)
    output_4 = encoder_4(obs_4)
    assert output_4["fused"].shape == (4, 4, latent_dim)
    print("✓ Works with 4 actors")

    # Test with 6 actors
    encoder_6 = HierarchicalCoordinationEncoder(
        num_actors=6,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
    )

    obs_6 = torch.randn(4, 6, obs_dim)
    output_6 = encoder_6(obs_6)
    assert output_6["fused"].shape == (4, 6, latent_dim)
    print("✓ Works with 6 actors")

    print("✓ Curriculum learning support verified (2→4→6 actors)")
