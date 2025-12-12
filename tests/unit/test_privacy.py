"""Tests for privacy mechanisms"""

import pytest
import torch

from swarm.privacy import (
    PrivacyEngine,
    PrivacyMode,
    LocalDifferentialPrivacy,
    DPSGDWrapper,
)


def test_ldp_noise_addition():
    """Test Local Differential Privacy noise addition"""
    ldp = LocalDifferentialPrivacy(mechanism="laplace")

    # Create sample weights
    weights = {
        "layer_1": torch.randn(10, 10),
        "layer_2": torch.randn(10, 5),
    }

    # Add LDP noise
    noisy_weights = ldp.add_noise(weights, epsilon=1.0, sensitivity=1.0)

    # Verify noise was added (weights should be different)
    assert not torch.allclose(noisy_weights["layer_1"], weights["layer_1"])
    assert not torch.allclose(noisy_weights["layer_2"], weights["layer_2"])

    # Verify shapes preserved
    assert noisy_weights["layer_1"].shape == weights["layer_1"].shape
    assert noisy_weights["layer_2"].shape == weights["layer_2"].shape

    print("✓ LDP noise addition working correctly")


def test_ldp_clip_and_noise():
    """Test gradient clipping with LDP noise"""
    ldp = LocalDifferentialPrivacy()

    # Create large weights
    weights = {
        "layer_1": torch.randn(10, 10) * 100,  # Large magnitude
    }

    # Clip and add noise
    noisy_weights = ldp.clip_and_add_noise(
        weights, epsilon=1.0, clip_norm=1.0
    )

    # Verify clipping occurred
    total_norm = torch.sqrt(sum(torch.sum(w ** 2) for w in noisy_weights.values()))
    assert total_norm <= 2.0  # Should be close to clip_norm (plus noise)

    print(f"✓ Gradient clipping working: norm = {total_norm:.2f}")


def test_dp_sgd_privatization():
    """Test DP-SGD weight privatization"""
    dp_sgd = DPSGDWrapper()

    weights = {
        "layer_1": torch.randn(10, 10),
        "layer_2": torch.randn(5, 5),
    }

    # Privatize
    private_weights = dp_sgd.privatize(
        weights,
        epsilon=2.0,
        delta=1e-5,
        clip_norm=1.0,
    )

    # Verify privacy was applied
    assert not torch.allclose(private_weights["layer_1"], weights["layer_1"])

    # Check privacy accounting
    privacy_spent = dp_sgd.get_privacy_spent(delta=1e-5)
    assert privacy_spent["steps"] == 1
    assert privacy_spent["delta"] == 1e-5

    print(f"✓ DP-SGD privatization working: ε={privacy_spent['epsilon']:.2f}")


def test_privacy_engine_modes():
    """Test privacy engine with different modes"""
    engine = PrivacyEngine()

    weights_list = [
        {"layer_1": torch.randn(5, 5)},
        {"layer_1": torch.randn(5, 5)},
    ]

    # Test LDP mode
    ldp_weights = engine.apply_privacy(
        weights_list,
        mode=PrivacyMode.LDP,
        epsilon=1.0,
        clip_norm=1.0,
    )
    assert len(ldp_weights) == 2
    print("✓ Privacy engine LDP mode working")

    # Test DP-SGD mode
    dp_weights = engine.apply_privacy(
        weights_list,
        mode=PrivacyMode.DP_SGD,
        epsilon=2.0,
        delta=1e-5,
        clip_norm=1.0,
    )
    assert len(dp_weights) == 2
    print("✓ Privacy engine DP-SGD mode working")

    # Test no privacy
    plain_weights = engine.apply_privacy(
        weights_list,
        mode=PrivacyMode.NONE,
    )
    assert len(plain_weights) == 2
    print("✓ Privacy engine NONE mode working")


def test_privacy_budget_accounting():
    """Test privacy budget tracking"""
    engine = PrivacyEngine()

    # Get accounting for DP-SGD
    accounting = engine.get_privacy_accountant(
        mode=PrivacyMode.DP_SGD,
        num_steps=100,
        delta=1e-5,
    )

    assert accounting is not None
    assert "epsilon" in accounting
    assert "delta" in accounting
    assert accounting["delta"] == 1e-5

    print(f"✓ Privacy accounting: ε={accounting['epsilon']:.2f}, δ={accounting['delta']:.2e}")


def test_ldp_different_mechanisms():
    """Test different LDP noise mechanisms"""
    # Laplace mechanism
    ldp_laplace = LocalDifferentialPrivacy(mechanism="laplace")
    weights = {"w": torch.randn(5, 5)}
    noisy_laplace = ldp_laplace.add_noise(weights, epsilon=1.0, sensitivity=1.0)
    assert not torch.allclose(noisy_laplace["w"], weights["w"])
    print("✓ Laplace mechanism working")

    # Gaussian mechanism
    ldp_gaussian = LocalDifferentialPrivacy(mechanism="gaussian")
    noisy_gaussian = ldp_gaussian.add_noise(weights, epsilon=1.0, sensitivity=1.0)
    assert not torch.allclose(noisy_gaussian["w"], weights["w"])
    print("✓ Gaussian mechanism working")
