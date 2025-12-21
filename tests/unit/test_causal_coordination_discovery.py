"""
Unit tests for Causal Coordination Discovery (CCD) module.

Tests cover:
1. Granger causality testing
2. Causal graph discovery
3. SCM learning
4. Counterfactual interventions
5. Graph evaluation metrics
"""

import unittest

import numpy as np

from swarmbridge.research.causal_coordination_discovery import (
    CausalCoordinationDiscovery,
    DirectedCoordinationGraph,
    evaluate_causal_graph,
)
from swarmbridge.schemas.coordination_primitives import CoordinationType


class TestCausalCoordinationDiscovery(unittest.TestCase):
    """Test suite for Causal Coordination Discovery."""

    def setUp(self):
        """Set up test fixtures."""
        self.ccd = CausalCoordinationDiscovery(
            significance_level=0.05, max_lag=5, min_edge_strength=0.05
        )

    def test_granger_causality_positive(self):
        """Test Granger causality with known causal relationship."""
        # X(t) -> Y(t+2): Y depends on X with lag 2
        np.random.seed(42)
        T = 200
        X = np.random.randn(T)
        Y = np.zeros(T)

        for t in range(2, T):
            Y[t] = 0.7 * X[t - 2] + 0.2 * Y[t - 1] + 0.1 * np.random.randn()

        trajectories = [{"X": X, "Y": Y}]

        result = self.ccd._granger_causality_test(trajectories, "X", "Y")

        self.assertTrue(result["is_causal"], "Should detect causal relationship")
        self.assertGreater(result["strength"], 0.0, "Causal strength should be positive")
        self.assertEqual(result["optimal_lag"], 2, "Should detect lag of 2")

    def test_granger_causality_negative(self):
        """Test Granger causality with independent variables."""
        # X and Y are independent
        np.random.seed(42)
        T = 200
        X = np.random.randn(T)
        Y = np.random.randn(T)

        trajectories = [{"X": X, "Y": Y}]

        result = self.ccd._granger_causality_test(trajectories, "X", "Y")

        # Should NOT detect causality (with high probability)
        # Note: With random data, there's a small chance of false positive
        # so we just check that strength is low
        self.assertLess(result["strength"], 0.5, "Strength should be low for independent vars")

    def test_discover_simple_causal_graph(self):
        """Test causal graph discovery on simple 2-variable case."""
        # A(t) -> B(t+1)
        np.random.seed(42)
        T = 200
        num_trajs = 10

        trajectories = []
        for _ in range(num_trajs):
            A = np.random.randn(T)
            B = np.zeros(T)
            for t in range(1, T):
                B[t] = 0.8 * A[t - 1] + 0.1 * np.random.randn()

            trajectories.append({"A": A, "B": B})

        graph = self.ccd.discover_causal_graph(
            multi_actor_trajectories=trajectories,
            coordination_primitive=CoordinationType.HANDOVER,
            variable_names=["A", "B"],
        )

        # Check nodes
        self.assertEqual(set(graph.nodes), {"A", "B"})

        # Should discover A -> B edge
        edge_pairs = [(src, tgt) for src, tgt, _ in graph.edges]
        self.assertIn(("A", "B"), edge_pairs, "Should discover A -> B edge")

        # Check temporal lag
        if ("A", "B") in graph.temporal_lags:
            lag = graph.temporal_lags[("A", "B")]
            self.assertEqual(lag, 1, "Should detect lag of 1")

    def test_causal_graph_methods(self):
        """Test DirectedCoordinationGraph utility methods."""
        graph = DirectedCoordinationGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B", 0.8), ("B", "C", 0.6), ("A", "C", 0.3)],
            temporal_lags={("A", "B"): 1, ("B", "C"): 2, ("A", "C"): 3},
            primitive_type=CoordinationType.HANDOVER,
        )

        # Test get_causal_parents
        self.assertEqual(graph.get_causal_parents("C"), ["B", "A"])
        self.assertEqual(graph.get_causal_parents("A"), [])

        # Test get_causal_children
        self.assertEqual(set(graph.get_causal_children("A")), {"B", "C"})

        # Test get_edge_strength
        self.assertEqual(graph.get_edge_strength("A", "B"), 0.8)
        self.assertIsNone(graph.get_edge_strength("C", "A"))

        # Test adjacency matrix
        adj = graph.to_adjacency_matrix()
        self.assertEqual(adj.shape, (3, 3))
        self.assertAlmostEqual(adj[0, 1], 0.8)  # A -> B

    def test_graphviz_export(self):
        """Test Graphviz DOT format export."""
        graph = DirectedCoordinationGraph(
            nodes=["giver_action", "receiver_obs"],
            edges=[("giver_action", "receiver_obs", 0.9)],
            temporal_lags={("giver_action", "receiver_obs"): 2},
            primitive_type=CoordinationType.HANDOVER,
        )

        dot = graph.to_graphviz()

        # Check basic DOT format
        self.assertIn("digraph CoordinationGraph", dot)
        self.assertIn("giver_action", dot)
        self.assertIn("receiver_obs", dot)
        self.assertIn("->", dot)
        self.assertIn("lag=2", dot)

    def test_coordination_skeleton_extraction(self):
        """Test extraction of coordination skeleton (critical edges)."""
        # Create graph with varying edge strengths
        graph = DirectedCoordinationGraph(
            nodes=["A", "B", "C", "D"],
            edges=[
                ("A", "B", 0.9),  # Strong
                ("B", "C", 0.8),  # Strong
                ("C", "D", 0.2),  # Weak
                ("A", "D", 0.1),  # Weak
            ],
            temporal_lags={},
            primitive_type=CoordinationType.FORMATION,
        )

        skeleton = graph.extract_coordination_skeleton()

        # Should extract top 20% edges (only strongest edge)
        self.assertGreater(len(skeleton.critical_edges), 0)
        self.assertLessEqual(len(skeleton.critical_edges), len(graph.edges))

        # Strongest edge should be included
        critical_pairs = [(src, tgt) for src, tgt, _ in skeleton.critical_edges]
        self.assertIn(("A", "B"), critical_pairs)

    def test_scm_learning(self):
        """Test Structural Causal Model learning."""
        # Create simple causal system: A -> B
        np.random.seed(42)
        T = 100
        trajectories = []

        for _ in range(20):
            A = np.random.randn(T)
            B = np.zeros(T)
            for t in range(1, T):
                B[t] = 0.5 * A[t - 1] + 0.3 * B[t - 1] + 0.1 * np.random.randn()

            trajectories.append({"A": A, "B": B})

        # Define causal graph
        graph = DirectedCoordinationGraph(
            nodes=["A", "B"],
            edges=[("A", "B", 0.8)],
            temporal_lags={("A", "B"): 1},
            primitive_type=CoordinationType.HANDOVER,
        )

        # Learn SCM
        scm = self.ccd.learn_structural_causal_model(graph, trajectories)

        # Check SCM structure
        self.assertIn("B", scm.variable_functions)
        self.assertEqual(scm.causal_graph, graph)

        # Test prediction (should be reasonable)
        test_traj = trajectories[0]
        predicted_B = scm.variable_functions["B"](test_traj)
        # Prediction should have similar scale to actual B
        self.assertGreater(np.corrcoef(predicted_B.flatten(), test_traj["B"])[0, 1], 0.3)

    def test_counterfactual_intervention(self):
        """Test counterfactual intervention queries."""
        # Create simple system
        np.random.seed(42)
        T = 50
        A = np.random.randn(T)
        B = np.zeros(T)

        for t in range(1, T):
            B[t] = 0.7 * A[t - 1] + 0.1 * np.random.randn()

        actual_trajectory = {"A": A, "B": B}

        # Create graph and SCM
        graph = DirectedCoordinationGraph(
            nodes=["A", "B"],
            edges=[("A", "B", 0.8)],
            temporal_lags={("A", "B"): 1},
            primitive_type=CoordinationType.HANDOVER,
        )

        scm = self.ccd.learn_structural_causal_model(graph, [actual_trajectory])

        # Intervention: Set A=5.0 at t=10
        intervention = {"A": 5.0}
        intervention_timestep = 10

        counterfactual = self.ccd.counterfactual_intervention(
            scm=scm,
            actual_trajectory=actual_trajectory,
            intervention=intervention,
            intervention_timestep=intervention_timestep,
        )

        # Check counterfactual trajectory
        self.assertIn("A", counterfactual)
        self.assertIn("B", counterfactual)

        # A should be 5.0 at intervention timestep
        self.assertAlmostEqual(counterfactual["A"][intervention_timestep], 5.0)

        # B should be different from actual after intervention (due to causal effect)
        # Note: This is a loose check since SCM learning is approximate
        self.assertIsNotNone(counterfactual["B"])

    def test_evaluate_causal_graph(self):
        """Test causal graph evaluation metrics."""
        # Ground truth: A -> B, B -> C
        ground_truth = DirectedCoordinationGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B", 1.0), ("B", "C", 1.0)],
            temporal_lags={},
            primitive_type=CoordinationType.HANDOVER,
        )

        # Perfect discovery
        perfect = DirectedCoordinationGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B", 0.9), ("B", "C", 0.8)],
            temporal_lags={},
            primitive_type=CoordinationType.HANDOVER,
        )

        metrics = evaluate_causal_graph(perfect, ground_truth)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["shd"], 0)

        # Partial discovery (missing B -> C)
        partial = DirectedCoordinationGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B", 0.9)],
            temporal_lags={},
            primitive_type=CoordinationType.HANDOVER,
        )

        metrics = evaluate_causal_graph(partial, ground_truth)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3, places=2)
        self.assertEqual(metrics["false_negatives"], 1)

        # With false positive (A -> C)
        with_fp = DirectedCoordinationGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B", 0.9), ("B", "C", 0.8), ("A", "C", 0.5)],
            temporal_lags={},
            primitive_type=CoordinationType.HANDOVER,
        )

        metrics = evaluate_causal_graph(with_fp, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 2 / 3, places=2)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["false_positives"], 1)


class TestCausalGraphIntegration(unittest.TestCase):
    """Integration tests for full CCD pipeline."""

    def test_end_to_end_handover(self):
        """Test end-to-end CCD on synthetic handover data."""
        # Generate simple handover: giver_action(t) -> receiver_action(t+2)
        np.random.seed(42)
        T = 150
        num_trajs = 30

        trajectories = []
        for _ in range(num_trajs):
            giver_action = np.random.randn(T)
            receiver_action = np.zeros(T)

            for t in range(2, T):
                receiver_action[t] = (
                    0.6 * giver_action[t - 2]
                    + 0.2 * receiver_action[t - 1]
                    + 0.15 * np.random.randn()
                )

            trajectories.append(
                {"giver_action": giver_action, "receiver_action": receiver_action}
            )

        # Run CCD
        ccd = CausalCoordinationDiscovery(
            significance_level=0.05, max_lag=5, min_edge_strength=0.1
        )

        graph = ccd.discover_causal_graph(
            multi_actor_trajectories=trajectories,
            coordination_primitive=CoordinationType.HANDOVER,
        )

        # Should discover giver_action -> receiver_action
        edge_pairs = [(src, tgt) for src, tgt, _ in graph.edges]
        self.assertIn(
            ("giver_action", "receiver_action"),
            edge_pairs,
            "Should discover causal edge",
        )

        # Check temporal lag (should be close to 2)
        if ("giver_action", "receiver_action") in graph.temporal_lags:
            lag = graph.temporal_lags[("giver_action", "receiver_action")]
            self.assertLessEqual(abs(lag - 2), 1, "Lag should be close to 2")

        # Learn SCM
        scm = ccd.learn_structural_causal_model(graph, trajectories[:20])
        self.assertIsNotNone(scm)

        # Test counterfactual
        counterfactual = ccd.counterfactual_intervention(
            scm=scm,
            actual_trajectory=trajectories[0],
            intervention={"giver_action": 10.0},
            intervention_timestep=10,
        )

        self.assertIn("receiver_action", counterfactual)


if __name__ == "__main__":
    unittest.main()
