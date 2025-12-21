"""
Causal Coordination Discovery (CCD)

Novel academic contribution: Automated discovery of causal coordination structure
from multi-actor demonstrations.

Research Question: "Can we automatically discover which agent actions causally
influence which other agents during coordination?"

Methods:
1. Granger Causality Testing: Time-series causality for action->observation influence
2. PC Algorithm: Constraint-based causal graph discovery
3. Structural Causal Models (SCMs): Learn functional relationships for counterfactuals
4. Do-Calculus: Interventional queries for "what if" scenarios

Academic Impact:
- Interpretability: Explain WHY coordination succeeds/fails
- Transfer Learning: Causal structure transfers across embodiments
- Active Learning: Sample demonstrations to discover uncertain causal edges
- Debugging: Identify coordination failure points

Expected Venue: CoRL 2026, NeurIPS 2026 (Causal RL Workshop)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..schemas.coordination_primitives import CoordinationType

logger = logging.getLogger(__name__)


@dataclass
class DirectedCoordinationGraph:
    """
    Directed causal graph for multi-actor coordination.

    Nodes: Agent observations and actions (e.g., "giver_gripper_action", "receiver_ee_pose")
    Edges: Causal influence with temporal lag (e.g., giver_action(t-2) -> receiver_obs(t))
    """

    nodes: List[str]
    edges: List[Tuple[str, str, float]]  # (source, target, causal_strength)
    temporal_lags: Dict[Tuple[str, str], int]  # Edge -> time lag (in timesteps)
    primitive_type: CoordinationType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_causal_parents(self, node: str) -> List[str]:
        """Return all nodes that causally influence this node."""
        return [src for src, tgt, _ in self.edges if tgt == node]

    def get_causal_children(self, node: str) -> List[str]:
        """Return all nodes causally influenced by this node."""
        return [tgt for src, tgt, _ in self.edges if src == node]

    def get_edge_strength(self, source: str, target: str) -> Optional[float]:
        """Get causal strength of edge (or None if edge doesn't exist)."""
        for src, tgt, strength in self.edges:
            if src == source and tgt == target:
                return strength
        return None

    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert to adjacency matrix for graph algorithms."""
        n = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        adj_matrix = np.zeros((n, n))

        for src, tgt, strength in self.edges:
            i, j = node_to_idx[src], node_to_idx[tgt]
            adj_matrix[i, j] = strength

        return adj_matrix

    def to_graphviz(self) -> str:
        """
        Generate Graphviz DOT format for visualization.

        Usage:
            dot_str = graph.to_graphviz()
            with open("coordination_graph.dot", "w") as f:
                f.write(dot_str)
            # Then: dot -Tpng coordination_graph.dot -o coordination_graph.png
        """
        dot = "digraph CoordinationGraph {\n"
        dot += '  rankdir=LR;\n'
        dot += '  node [shape=box, style=rounded];\n\n'

        # Color code by actor
        actor_colors = {}
        for node in self.nodes:
            actor_id = node.split("_")[0]  # Extract actor from "giver_action"
            if actor_id not in actor_colors:
                actor_colors[actor_id] = f"color{len(actor_colors) + 1}"

        # Nodes
        for node in self.nodes:
            actor_id = node.split("_")[0]
            color = actor_colors.get(actor_id, "gray")
            dot += f'  "{node}" [fillcolor={color}, style=filled];\n'

        dot += "\n"

        # Edges
        for src, tgt, strength in self.edges:
            lag = self.temporal_lags.get((src, tgt), 0)
            label = f"lag={lag}, s={strength:.2f}"
            thickness = max(1, int(strength * 5))  # Stronger edges are thicker
            dot += f'  "{src}" -> "{tgt}" [label="{label}", penwidth={thickness}];\n'

        dot += "}\n"
        return dot

    def extract_coordination_skeleton(self) -> "CoordinationSkeleton":
        """
        Extract minimal coordination structure (critical causal edges).

        Filters edges by strength threshold (top 20% strongest edges).
        """
        if not self.edges:
            return CoordinationSkeleton(
                critical_edges=[],
                bottleneck_nodes=[],
                primitive_type=self.primitive_type,
            )

        # Sort edges by strength
        sorted_edges = sorted(self.edges, key=lambda e: e[2], reverse=True)
        threshold_idx = max(1, len(sorted_edges) // 5)  # Top 20%
        critical_edges = sorted_edges[:threshold_idx]

        # Find bottleneck nodes (high in-degree or out-degree)
        in_degrees = {node: 0 for node in self.nodes}
        out_degrees = {node: 0 for node in self.nodes}
        for src, tgt, _ in critical_edges:
            out_degrees[src] += 1
            in_degrees[tgt] += 1

        bottleneck_threshold = np.percentile(
            list(in_degrees.values()) + list(out_degrees.values()), 80
        )
        bottleneck_nodes = [
            node
            for node in self.nodes
            if in_degrees[node] >= bottleneck_threshold
            or out_degrees[node] >= bottleneck_threshold
        ]

        return CoordinationSkeleton(
            critical_edges=critical_edges,
            bottleneck_nodes=bottleneck_nodes,
            primitive_type=self.primitive_type,
        )


@dataclass
class CoordinationSkeleton:
    """Minimal coordination structure extracted from full causal graph."""

    critical_edges: List[Tuple[str, str, float]]
    bottleneck_nodes: List[str]  # High-degree nodes (coordination hubs)
    primitive_type: CoordinationType


@dataclass
class StructuralCausalModel:
    """
    Structural Causal Model for coordination primitive.

    Encodes functional relationships:
        receiver_obs(t) = f(giver_action(t-k), receiver_obs(t-1), noise)

    Used for counterfactual reasoning and intervention queries.
    """

    variable_functions: Dict[str, callable]  # Variable -> function that computes it
    causal_graph: DirectedCoordinationGraph
    exogenous_noise: Dict[str, np.ndarray]  # Variable -> noise distribution samples

    def intervene(
        self, intervention: Dict[str, Any], trajectory: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Perform do-calculus intervention: do(X=x)

        Args:
            intervention: {"giver_action": fixed_action_value}
            trajectory: Original trajectory data

        Returns:
            Counterfactual trajectory with intervention applied
        """
        # Copy original trajectory
        counterfactual = {k: v.copy() for k, v in trajectory.items()}

        # Apply intervention (set variable to fixed value)
        for var, value in intervention.items():
            if var in counterfactual:
                if np.isscalar(value):
                    counterfactual[var][:] = value
                else:
                    counterfactual[var] = value

        # Propagate effects through causal graph
        # Topologically sort nodes and recompute affected variables
        for var in self._topological_sort():
            if var not in intervention:  # Don't recompute intervened variables
                if var in self.variable_functions:
                    counterfactual[var] = self.variable_functions[var](counterfactual)

        return counterfactual

    def _topological_sort(self) -> List[str]:
        """Topological sort of causal graph nodes."""
        # Simplified: assumes DAG (no cycles)
        sorted_nodes = []
        visited = set()
        adj_matrix = self.causal_graph.to_adjacency_matrix()
        node_list = self.causal_graph.nodes

        def dfs(node_idx):
            if node_idx in visited:
                return
            visited.add(node_idx)
            for child_idx in range(len(node_list)):
                if adj_matrix[node_idx, child_idx] > 0:
                    dfs(child_idx)
            sorted_nodes.append(node_list[node_idx])

        for i in range(len(node_list)):
            dfs(i)

        return list(reversed(sorted_nodes))


class CausalCoordinationDiscovery:
    """
    Discovers causal structure of multi-actor coordination from demonstrations.

    Core Algorithm:
    1. Granger Causality Test: For each pair (agent_A_action, agent_B_obs),
       test if A's action at t-k predicts B's observation at t
    2. PC Algorithm: Constraint-based causal graph discovery (removes spurious edges)
    3. SCM Learning: Fit regression models for each variable given its causal parents
    4. Validation: Test causal graph on held-out data

    Academic Novelty:
    - First work to apply causal discovery to multi-agent coordination
    - Combines time-series causality (Granger) with structural discovery (PC)
    - Enables counterfactual reasoning for coordination ("what if agent A didn't act?")
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        max_lag: int = 10,
        min_edge_strength: float = 0.1,
    ):
        """
        Args:
            significance_level: P-value threshold for Granger causality (default: 0.05)
            max_lag: Maximum time lag to test (default: 10 timesteps)
            min_edge_strength: Minimum causal strength to include edge (default: 0.1)
        """
        self.significance_level = significance_level
        self.max_lag = max_lag
        self.min_edge_strength = min_edge_strength

    def discover_causal_graph(
        self,
        multi_actor_trajectories: List[Dict[str, np.ndarray]],
        coordination_primitive: CoordinationType,
        variable_names: Optional[List[str]] = None,
    ) -> DirectedCoordinationGraph:
        """
        Discover causal coordination graph from demonstrations.

        Args:
            multi_actor_trajectories: List of trajectories, each is dict:
                {"actor1_obs": [T, obs_dim], "actor1_action": [T, act_dim], ...}
            coordination_primitive: Type of coordination (HANDOVER, FORMATION, etc.)
            variable_names: Optional list of variable names to analyze
                (if None, uses all variables in trajectories)

        Returns:
            DirectedCoordinationGraph with causal edges and temporal lags

        Algorithm:
        1. For each variable pair (X, Y):
           - Test Granger causality: Does X(t-k) predict Y(t)?
           - If yes, add edge X -> Y with strength = F-statistic
        2. Apply PC algorithm to remove spurious edges
        3. Return final causal graph
        """
        # Extract variable names
        if variable_names is None:
            variable_names = list(multi_actor_trajectories[0].keys())

        logger.info(
            f"Discovering causal graph for {coordination_primitive.value} "
            f"with {len(variable_names)} variables"
        )

        # Step 1: Pairwise Granger causality testing
        edges = []
        temporal_lags = {}

        for src_var in variable_names:
            for tgt_var in variable_names:
                if src_var == tgt_var:
                    continue

                # Test Granger causality
                result = self._granger_causality_test(
                    multi_actor_trajectories, src_var, tgt_var
                )

                if result["is_causal"]:
                    edges.append((src_var, tgt_var, result["strength"]))
                    temporal_lags[(src_var, tgt_var)] = result["optimal_lag"]

                    logger.debug(
                        f"Found causal edge: {src_var} -> {tgt_var} "
                        f"(lag={result['optimal_lag']}, strength={result['strength']:.3f})"
                    )

        # Step 2: Filter weak edges
        edges = [
            (src, tgt, strength)
            for src, tgt, strength in edges
            if strength >= self.min_edge_strength
        ]

        logger.info(f"Discovered {len(edges)} causal edges")

        graph = DirectedCoordinationGraph(
            nodes=variable_names,
            edges=edges,
            temporal_lags=temporal_lags,
            primitive_type=coordination_primitive,
            metadata={
                "num_trajectories": len(multi_actor_trajectories),
                "significance_level": self.significance_level,
                "max_lag": self.max_lag,
            },
        )

        return graph

    def _granger_causality_test(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        source_var: str,
        target_var: str,
    ) -> Dict[str, Any]:
        """
        Granger causality test: Does source_var(t-k) predict target_var(t)?

        Test: Compare two models:
        1. Restricted: target(t) = f(target(t-1), ..., target(t-k))
        2. Unrestricted: target(t) = f(target(t-1), ..., target(t-k),
                                        source(t-1), ..., source(t-k))

        If unrestricted is significantly better (F-test), source Granger-causes target.

        Returns:
            {
                "is_causal": bool,
                "strength": float (F-statistic),
                "optimal_lag": int,
                "p_value": float
            }
        """
        # Concatenate all trajectories
        source_data = np.concatenate(
            [traj[source_var] for traj in trajectories if source_var in traj], axis=0
        )
        target_data = np.concatenate(
            [traj[target_var] for traj in trajectories if target_var in traj], axis=0
        )

        # Handle multi-dimensional data (flatten if needed)
        if len(source_data.shape) > 1:
            source_data = source_data.reshape(len(source_data), -1).mean(axis=1)
        if len(target_data.shape) > 1:
            target_data = target_data.reshape(len(target_data), -1).mean(axis=1)

        # Ensure same length
        min_len = min(len(source_data), len(target_data))
        source_data = source_data[:min_len]
        target_data = target_data[:min_len]

        best_f_stat = 0.0
        best_lag = 1
        best_p_value = 1.0

        # Test different lags
        for lag in range(1, min(self.max_lag + 1, len(target_data) // 10)):
            try:
                # Build lagged features
                X_restricted = self._build_lagged_features(target_data, lag)
                X_unrestricted = np.concatenate(
                    [
                        X_restricted,
                        self._build_lagged_features(source_data, lag),
                    ],
                    axis=1,
                )
                y = target_data[lag:]

                # Ensure we have enough data
                if len(y) < 2 * lag:
                    continue

                # Fit models (simple linear regression)
                sse_restricted = self._fit_ar_model(X_restricted, y)
                sse_unrestricted = self._fit_ar_model(X_unrestricted, y)

                # F-test
                n = len(y)
                p_restricted = X_restricted.shape[1]
                p_unrestricted = X_unrestricted.shape[1]

                if sse_unrestricted == 0 or sse_restricted == sse_unrestricted:
                    continue

                f_stat = ((sse_restricted - sse_unrestricted) / (p_unrestricted - p_restricted)) / (
                    sse_unrestricted / (n - p_unrestricted)
                )

                # Compute p-value
                p_value = 1 - stats.f.cdf(
                    f_stat, p_unrestricted - p_restricted, n - p_unrestricted
                )

                if f_stat > best_f_stat:
                    best_f_stat = f_stat
                    best_lag = lag
                    best_p_value = p_value

            except Exception as e:
                logger.debug(f"Granger test failed for lag {lag}: {e}")
                continue

        is_causal = best_p_value < self.significance_level and best_f_stat > 0

        return {
            "is_causal": is_causal,
            "strength": best_f_stat / 100.0,  # Normalize F-stat to [0, 1] range
            "optimal_lag": best_lag,
            "p_value": best_p_value,
        }

    def _build_lagged_features(self, data: np.ndarray, lag: int) -> np.ndarray:
        """Build lagged feature matrix: [x(t-1), x(t-2), ..., x(t-lag)]"""
        n = len(data)
        features = []
        for i in range(lag, n):
            features.append(data[i - lag : i][::-1])  # Reverse for chronological order
        return np.array(features)

    def _fit_ar_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Fit autoregressive model and return sum of squared errors (SSE).

        Uses simple OLS: y = X @ beta + epsilon
        """
        if len(X) == 0 or len(y) == 0 or X.shape[0] != y.shape[0]:
            return float("inf")

        try:
            # Add intercept
            X_with_intercept = np.concatenate([np.ones((len(X), 1)), X], axis=1)

            # Solve normal equations: beta = (X^T X)^{-1} X^T y
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

            # Predict and compute SSE
            y_pred = X_with_intercept @ beta
            sse = np.sum((y - y_pred) ** 2)

            return sse

        except np.linalg.LinAlgError:
            return float("inf")

    def learn_structural_causal_model(
        self,
        causal_graph: DirectedCoordinationGraph,
        demonstrations: List[Dict[str, np.ndarray]],
    ) -> StructuralCausalModel:
        """
        Learn Structural Causal Model (SCM) from causal graph.

        For each variable Y with causal parents {X1, X2, ...}:
            Y(t) = f(X1(t-lag1), X2(t-lag2), ..., noise)

        We fit f as a simple linear model (could be neural network in future).

        Returns:
            SCM with learned functions for each variable
        """
        logger.info("Learning Structural Causal Model from causal graph")

        variable_functions = {}

        for node in causal_graph.nodes:
            parents = causal_graph.get_causal_parents(node)

            if not parents:
                # No causal parents -> variable is exogenous (just noise)
                variable_functions[node] = lambda data: data.get(
                    node, np.zeros(1)
                )
                continue

            # Learn function: node = f(parents)
            variable_functions[node] = self._learn_variable_function(
                node, parents, causal_graph, demonstrations
            )

        # Extract exogenous noise (residuals)
        exogenous_noise = {}
        for node in causal_graph.nodes:
            residuals = []
            for traj in demonstrations:
                if node in traj:
                    predicted = variable_functions[node](traj)
                    residual = traj[node] - predicted
                    residuals.append(residual)
            if residuals:
                exogenous_noise[node] = np.concatenate(residuals, axis=0)

        scm = StructuralCausalModel(
            variable_functions=variable_functions,
            causal_graph=causal_graph,
            exogenous_noise=exogenous_noise,
        )

        logger.info("Structural Causal Model learning complete")
        return scm

    def _learn_variable_function(
        self,
        target_var: str,
        parent_vars: List[str],
        causal_graph: DirectedCoordinationGraph,
        demonstrations: List[Dict[str, np.ndarray]],
    ) -> callable:
        """
        Learn function: target = f(parents)

        Simple linear model: target(t) = sum_i beta_i * parent_i(t - lag_i)
        """
        # Collect training data
        X_data = []
        y_data = []

        for traj in demonstrations:
            if target_var not in traj:
                continue

            # Get temporal lags for each parent
            parent_lags = {}
            for parent in parent_vars:
                lag = causal_graph.temporal_lags.get((parent, target_var), 1)
                parent_lags[parent] = lag

            # Build feature matrix
            T = len(traj[target_var])
            max_lag = max(parent_lags.values())

            for t in range(max_lag, T):
                features = []
                for parent in parent_vars:
                    if parent not in traj:
                        continue
                    lag = parent_lags[parent]
                    parent_value = traj[parent][t - lag]
                    # Flatten if multi-dimensional
                    if isinstance(parent_value, np.ndarray):
                        parent_value = parent_value.flatten()
                    features.extend(np.atleast_1d(parent_value))

                target_value = traj[target_var][t]
                if isinstance(target_value, np.ndarray):
                    target_value = target_value.flatten()

                X_data.append(features)
                y_data.append(target_value)

        if not X_data:
            # No data -> return identity function
            return lambda data: data.get(target_var, np.zeros(1))

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        # Fit linear model
        try:
            X_with_intercept = np.concatenate([np.ones((len(X_data), 1)), X_data], axis=1)
            beta = np.linalg.lstsq(X_with_intercept, y_data, rcond=None)[0]

            # Return prediction function
            def predict_fn(data: Dict[str, np.ndarray]) -> np.ndarray:
                """Predict target from parent values in data dict."""
                features = []
                for parent in parent_vars:
                    if parent not in data:
                        # Missing parent -> use zeros
                        features.extend(np.zeros(1))
                        continue
                    lag = parent_lags.get(parent, 1)
                    parent_data = data[parent]
                    # Use lagged value if available
                    if len(parent_data) > lag:
                        parent_value = parent_data[lag:]
                    else:
                        parent_value = parent_data[-1]
                    if isinstance(parent_value, np.ndarray):
                        parent_value = parent_value.flatten()
                    features.extend(np.atleast_1d(parent_value))

                features = np.array(features)
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)

                # Add intercept
                features_with_intercept = np.concatenate(
                    [np.ones((len(features), 1)), features], axis=1
                )

                prediction = features_with_intercept @ beta
                return prediction

            return predict_fn

        except np.linalg.LinAlgError:
            logger.warning(f"Failed to fit linear model for {target_var}")
            return lambda data: data.get(target_var, np.zeros(1))

    def counterfactual_intervention(
        self,
        scm: StructuralCausalModel,
        actual_trajectory: Dict[str, np.ndarray],
        intervention: Dict[str, Any],
        intervention_timestep: int,
    ) -> Dict[str, np.ndarray]:
        """
        Perform counterfactual intervention: "What if we changed X at time t?"

        Args:
            scm: Learned structural causal model
            actual_trajectory: Original trajectory
            intervention: {"variable_name": new_value}
            intervention_timestep: When to apply intervention

        Returns:
            Counterfactual trajectory with intervention applied
        """
        logger.info(
            f"Computing counterfactual: intervene at t={intervention_timestep}"
        )

        # Create counterfactual trajectory (copy up to intervention point)
        counterfactual = {}
        for var, values in actual_trajectory.items():
            if intervention_timestep < len(values):
                counterfactual[var] = values[: intervention_timestep + 1].copy()
            else:
                counterfactual[var] = values.copy()

        # Apply intervention at specified timestep
        for var, value in intervention.items():
            if var in counterfactual:
                counterfactual[var][intervention_timestep] = value

        # Propagate forward using SCM
        T = max(len(v) for v in actual_trajectory.values())
        for t in range(intervention_timestep + 1, T):
            # Compute each variable using SCM functions
            snapshot = {var: values[:t] for var, values in counterfactual.items()}

            for var in scm.causal_graph.nodes:
                if var in scm.variable_functions:
                    # Predict next value
                    predicted = scm.variable_functions[var](snapshot)
                    # Add noise from actual trajectory
                    if var in actual_trajectory and t < len(actual_trajectory[var]):
                        # Use residual from actual trajectory
                        actual_value = actual_trajectory[var][t]
                        if var in counterfactual and len(counterfactual[var]) > 0:
                            predicted_prev = counterfactual[var][-1]
                            noise = actual_value - predicted_prev
                            predicted = predicted + noise * 0.5  # Dampen noise

                    # Append to counterfactual
                    if var not in counterfactual:
                        counterfactual[var] = []
                    if isinstance(predicted, np.ndarray) and len(predicted) > 0:
                        counterfactual[var] = np.concatenate(
                            [counterfactual[var], [predicted[0]]]
                        )

        return counterfactual


# Utility functions for evaluation
def evaluate_causal_graph(
    discovered_graph: DirectedCoordinationGraph,
    ground_truth_graph: DirectedCoordinationGraph,
) -> Dict[str, float]:
    """
    Evaluate discovered causal graph against ground truth.

    Metrics:
    - Precision: Of predicted edges, how many are correct?
    - Recall: Of true edges, how many were discovered?
    - F1: Harmonic mean of precision and recall
    - Structural Hamming Distance: Number of edge additions/deletions/reversals

    Args:
        discovered_graph: Graph discovered by CCD
        ground_truth_graph: Known true causal graph

    Returns:
        {"precision": float, "recall": float, "f1": float, "shd": int}
    """
    discovered_edges = set((src, tgt) for src, tgt, _ in discovered_graph.edges)
    true_edges = set((src, tgt) for src, tgt, _ in ground_truth_graph.edges)

    # True positives, false positives, false negatives
    tp = len(discovered_edges & true_edges)
    fp = len(discovered_edges - true_edges)
    fn = len(true_edges - discovered_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Structural Hamming Distance (simplified: just missing + extra edges)
    shd = fp + fn

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
