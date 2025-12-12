"""
Robust Aggregation Strategies

Implements secure aggregation methods resistant to poisoning attacks:
- Trimmed mean
- Median
- Krum
- Coordinate-wise median
"""

from enum import Enum
from typing import Dict, List

import numpy as np
import torch


class AggregationStrategy(str, Enum):
    """Supported aggregation strategies"""

    MEAN = "mean"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    KRUM = "krum"
    COORDINATE_MEDIAN = "coordinate_median"


class RobustAggregator:
    """Robust aggregation for federated learning"""

    def __init__(self, trim_ratio: float = 0.1, krum_m: int = 1):
        """
        Args:
            trim_ratio: Fraction to trim from each end (for trimmed mean)
            krum_m: Number of neighbors to average (for Krum)
        """
        self.trim_ratio = trim_ratio
        self.krum_m = krum_m

    def aggregate(
        self,
        weights_list: List[Dict[str, torch.Tensor]],
        strategy: AggregationStrategy = AggregationStrategy.TRIMMED_MEAN,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model weights using specified strategy

        Args:
            weights_list: List of model weight dictionaries
            strategy: Aggregation strategy to use

        Returns:
            Aggregated weights dictionary
        """
        if not weights_list:
            raise ValueError("No weights to aggregate")

        if strategy == AggregationStrategy.MEAN:
            return self._aggregate_mean(weights_list)
        elif strategy == AggregationStrategy.TRIMMED_MEAN:
            return self._aggregate_trimmed_mean(weights_list)
        elif strategy == AggregationStrategy.MEDIAN:
            return self._aggregate_median(weights_list)
        elif strategy == AggregationStrategy.COORDINATE_MEDIAN:
            return self._aggregate_coordinate_median(weights_list)
        elif strategy == AggregationStrategy.KRUM:
            return self._aggregate_krum(weights_list)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _aggregate_mean(
        self, weights_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Simple averaging"""
        aggregated = {}
        n = len(weights_list)

        for key in weights_list[0].keys():
            stacked = torch.stack([w[key] for w in weights_list])
            aggregated[key] = stacked.mean(dim=0)

        return aggregated

    def _aggregate_trimmed_mean(
        self, weights_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Trimmed mean: Remove outliers and average

        Removes trim_ratio from each end, then averages.
        More robust to poisoning than simple mean.
        """
        aggregated = {}
        n = len(weights_list)
        trim_count = int(n * self.trim_ratio)

        if trim_count * 2 >= n:
            # Not enough samples to trim, fall back to mean
            return self._aggregate_mean(weights_list)

        for key in weights_list[0].keys():
            stacked = torch.stack([w[key] for w in weights_list])  # [n, *weight_shape]

            # Sort along batch dimension and trim
            sorted_weights, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_weights[trim_count : n - trim_count]

            # Average trimmed values
            aggregated[key] = trimmed.mean(dim=0)

        return aggregated

    def _aggregate_median(
        self, weights_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Element-wise median (most robust, but can be slow)"""
        aggregated = {}

        for key in weights_list[0].keys():
            stacked = torch.stack([w[key] for w in weights_list])
            aggregated[key] = torch.median(stacked, dim=0)[0]

        return aggregated

    def _aggregate_coordinate_median(
        self, weights_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise median

        More efficient than full median for high dimensions.
        """
        aggregated = {}

        for key in weights_list[0].keys():
            stacked = torch.stack([w[key] for w in weights_list])
            # Flatten weights for each participant
            flattened = [w.flatten() for w in stacked]
            stacked_flat = torch.stack(flattened)

            # Compute median coordinate-wise
            median_flat = torch.median(stacked_flat, dim=0)[0]

            # Reshape back
            aggregated[key] = median_flat.reshape(weights_list[0][key].shape)

        return aggregated

    def _aggregate_krum(
        self, weights_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Krum: Select most representative update

        Selects the update with smallest sum of distances to its
        m nearest neighbors. More robust to Byzantine attacks.
        """
        n = len(weights_list)
        m = min(self.krum_m, n - 2)

        # Flatten all weights for distance computation
        flattened = []
        for weights in weights_list:
            flat = torch.cat([w.flatten() for w in weights.values()])
            flattened.append(flat)

        # Compute pairwise distances
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flattened[i] - flattened[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # For each update, compute sum of distances to m nearest neighbors
        scores = []
        for i in range(n):
            dists = distances[i].clone()
            dists[i] = float("inf")  # Exclude self
            nearest_dists, _ = torch.topk(dists, m, largest=False)
            score = nearest_dists.sum()
            scores.append(score)

        # Select update with smallest score
        best_idx = np.argmin(scores)
        return weights_list[best_idx]

    def detect_outliers(
        self, weights_list: List[Dict[str, torch.Tensor]], threshold: float = 3.0
    ) -> List[int]:
        """
        Detect outlier updates using z-score

        Args:
            weights_list: List of weight dictionaries
            threshold: Z-score threshold for outlier detection

        Returns:
            List of indices of outlier updates
        """
        n = len(weights_list)

        # Flatten weights
        flattened = []
        for weights in weights_list:
            flat = torch.cat([w.flatten() for w in weights.values()])
            flattened.append(flat)

        # Compute pairwise distances
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flattened[i] - flattened[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute mean distance for each update
        mean_distances = distances.mean(dim=1)

        # Compute z-scores
        mean = mean_distances.mean()
        std = mean_distances.std()
        z_scores = (mean_distances - mean) / (std + 1e-8)

        # Identify outliers
        outliers = torch.where(torch.abs(z_scores) > threshold)[0].tolist()

        return outliers
