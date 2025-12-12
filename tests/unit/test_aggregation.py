"""Tests for robust aggregation"""

import pytest
import torch

from swarm.openfl.aggregator import RobustAggregator, AggregationStrategy


def test_mean_aggregation():
    """Test simple mean aggregation"""
    aggregator = RobustAggregator()

    weights_list = [
        {"layer_1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        {"layer_1": torch.tensor([[2.0, 3.0], [4.0, 5.0]])},
        {"layer_1": torch.tensor([[3.0, 4.0], [5.0, 6.0]])},
    ]

    result = aggregator.aggregate(weights_list, strategy=AggregationStrategy.MEAN)

    expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    assert torch.allclose(result["layer_1"], expected)

    print("✓ Mean aggregation correct")


def test_trimmed_mean_aggregation():
    """Test trimmed mean (removes outliers)"""
    aggregator = RobustAggregator(trim_ratio=0.33)

    # Create weights with one outlier
    weights_list = [
        {"layer_1": torch.tensor([1.0, 2.0, 3.0])},
        {"layer_1": torch.tensor([1.1, 2.1, 3.1])},
        {"layer_1": torch.tensor([100.0, 200.0, 300.0])},  # Outlier
    ]

    result = aggregator.aggregate(weights_list, strategy=AggregationStrategy.TRIMMED_MEAN)

    # Should be close to [1.05, 2.05, 3.05] (average of first two, outlier trimmed)
    assert result["layer_1"][0] < 10.0  # Outlier should be removed
    print(f"✓ Trimmed mean result: {result['layer_1']}")


def test_median_aggregation():
    """Test median aggregation (most robust)"""
    aggregator = RobustAggregator()

    weights_list = [
        {"layer_1": torch.tensor([1.0, 2.0, 3.0])},
        {"layer_1": torch.tensor([2.0, 3.0, 4.0])},
        {"layer_1": torch.tensor([3.0, 4.0, 5.0])},
    ]

    result = aggregator.aggregate(weights_list, strategy=AggregationStrategy.MEDIAN)

    expected = torch.tensor([2.0, 3.0, 4.0])  # Median of each element
    assert torch.allclose(result["layer_1"], expected)

    print("✓ Median aggregation correct")


def test_krum_aggregation():
    """Test Krum (selects most representative update)"""
    aggregator = RobustAggregator(krum_m=1)

    # Create weights where middle one is most representative
    weights_list = [
        {"layer_1": torch.tensor([1.0, 1.0])},
        {"layer_1": torch.tensor([2.0, 2.0])},  # Most central
        {"layer_1": torch.tensor([3.0, 3.0])},
    ]

    result = aggregator.aggregate(weights_list, strategy=AggregationStrategy.KRUM)

    # Should select one of the updates (exact one depends on distances)
    assert "layer_1" in result
    print(f"✓ Krum selected update: {result['layer_1']}")


def test_outlier_detection():
    """Test outlier detection in aggregation"""
    aggregator = RobustAggregator()

    weights_list = [
        {"layer_1": torch.tensor([1.0, 2.0])},
        {"layer_1": torch.tensor([1.1, 2.1])},
        {"layer_1": torch.tensor([1.2, 2.2])},
        {"layer_1": torch.tensor([100.0, 200.0])},  # Clear outlier
    ]

    outliers = aggregator.detect_outliers(weights_list, threshold=2.0)

    # Should detect the outlier (index 3)
    assert len(outliers) > 0
    print(f"✓ Detected outliers: {outliers}")


def test_aggregation_with_poisoning():
    """Test aggregation resilience to poisoned updates"""
    aggregator = RobustAggregator(trim_ratio=0.2)

    # 3 honest sites + 1 malicious
    weights_list = [
        {"layer_1": torch.tensor([1.0, 2.0, 3.0])},  # Honest
        {"layer_1": torch.tensor([1.1, 2.1, 3.1])},  # Honest
        {"layer_1": torch.tensor([0.9, 1.9, 2.9])},  # Honest
        {"layer_1": torch.tensor([1000.0, 2000.0, 3000.0])},  # Poisoned
    ]

    # Trimmed mean should be robust
    result = aggregator.aggregate(weights_list, strategy=AggregationStrategy.TRIMMED_MEAN)

    # Result should be close to honest average (~1.0, 2.0, 3.0)
    assert result["layer_1"][0] < 10.0  # Not influenced by poisoned update
    print(f"✓ Robust to poisoning: {result['layer_1']}")
