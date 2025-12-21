# SwarmBridge Academic Research Roadmap

**Vision**: Make SwarmBridge the state-of-the-art platform for **structured multi-actor imitation learning** with novel research contributions in causal coordination discovery, temporal credit assignment, and privacy-preserving federated swarm learning.

---

## 🎯 **Research Gap Analysis**

### Current Multi-Agent IL Limitations:
1. **Black-box coordination**: Learned policies don't explain *why* agents coordinate
2. **Flat temporal modeling**: No hierarchical understanding of coordination phases
3. **Privacy gaps**: Federated multi-robot learning lacks practical secure aggregation
4. **Data inefficiency**: No principled active learning for multi-actor demonstrations
5. **Transfer failure**: Coordination skills don't transfer across embodiments
6. **Credit assignment**: Unknown which agent's actions contributed to success
7. **Rare event handling**: Coordination failures (collisions, dropped objects) are ignored

---

## 📊 **7 Novel Research Directions**

---

### **1. Causal Coordination Discovery (CCD)** ⭐⭐⭐

**Academic Gap**:
- Existing work: Communication learning (CommNet, TarMAC), emergent coordination (QMIX, MAPPO)
- Missing: Explicit causal graphs showing *which agent actions influence which other agents*
- No work on discovering coordination **structure** from demonstrations

**Research Question**:
*"Can we automatically discover causal coordination dependencies from multi-actor demonstrations?"*

**Novel Contribution**:
- **Granger Causality** for action-observation influence detection
- **Structural Causal Models (SCMs)** for coordination primitives
- **Interventional queries**: "What if agent A didn't perform handover gesture?"

**Implementation**:
```python
# swarmbridge/research/causal_coordination_discovery.py

class CausalCoordinationDiscovery:
    """
    Discovers causal structure of coordination from demonstrations.

    Methods:
    1. Granger Causality Test: Does agent A's action at t-k predict agent B's obs at t?
    2. Structural Equation Modeling: Learn SCM of coordination primitive
    3. Do-calculus Interventions: Counterfactual "what if" scenarios

    Output: DirectedCoordinationGraph with edge weights = causal strength
    """

    def discover_causal_graph(
        self,
        multi_actor_trajectories: List[Dict[str, np.ndarray]],
        coordination_primitive: CoordinationType,
        lag_window: int = 5,
    ) -> DirectedCoordinationGraph:
        """
        Granger causality + PC algorithm for causal discovery.

        Returns:
            graph: A -> B edge if agent A's actions Granger-cause agent B's observations
            edge_weights: Strength of causal influence (F-statistic from Granger test)
            temporal_lags: At which time lag does causation occur?
        """

    def learn_structural_causal_model(
        self,
        causal_graph: DirectedCoordinationGraph,
        demonstrations: List[Trajectory],
    ) -> StructuralCausalModel:
        """
        Learn SCM: o_B(t) = f(a_A(t-k), o_B(t-1), noise)

        For handover:
        - receiver_gripper_state(t) = f(giver_release_action(t-2), receiver_approach(t-1))
        """

    def counterfactual_intervention(
        self,
        scm: StructuralCausalModel,
        intervention: Dict[str, Any],  # {"giver_action": "no_release"}
    ) -> Dict[str, np.ndarray]:
        """
        Answer: "What would receiver do if giver didn't release object?"

        Used for:
        - Debugging coordination failures
        - Explaining policy decisions
        - Safety validation ("what if collision avoidance failed?")
        """

# Output Format:
@dataclass
class DirectedCoordinationGraph:
    nodes: List[str]  # ["giver_gripper_action", "receiver_ee_obs", ...]
    edges: List[Tuple[str, str, float]]  # (source, target, causal_strength)
    temporal_lags: Dict[Tuple[str, str], int]  # Edge -> time lag
    primitive_type: CoordinationType  # HANDOVER, FORMATION, etc.

    def to_graphviz(self) -> str:
        """Visualize causal coordination graph"""

    def extract_coordination_skeleton(self) -> CoordinationSkeleton:
        """Extract minimal coordination structure (like task graph but causal)"""
```

**Academic Impact**:
- **Interpretability**: Explain *why* coordination succeeds/fails
- **Transfer learning**: Causal structure transfers across embodiments
- **Active learning**: Sample demonstrations to discover uncertain causal edges
- **Venue**: CoRL, ICRA, NeurIPS (causal RL workshop)

**Baseline Comparisons**:
- vs. CommNet: Communication learned, but not causal
- vs. QMIX: Joint Q-value, but no causal structure
- vs. Attention mechanisms: Correlation ≠ causation

**Metrics**:
- F1 score of discovered causal edges vs. ground truth (in simulation)
- Transfer success rate: Train on robot A+B, test on robot C+D
- Counterfactual prediction error: Accuracy of intervention queries

---

### **2. Temporal Coordination Credit Assignment (TCCA)** ⭐⭐⭐

**Academic Gap**:
- Existing work: Credit assignment in single-agent RL (eligibility traces, hindsight)
- Missing: Which agent's action at which timestep contributed to coordination success?
- Example: Handover succeeds at t=50. Was it giver's approach (t=10) or receiver's grasp (t=45)?

**Research Question**:
*"How do we assign credit to individual actor actions in temporally-extended coordination?"*

**Novel Contribution**:
- **Shapley Values** for multi-agent action contributions
- **Influence Traces**: Backpropagate coordination success through causal graph
- **Counterfactual Baseline**: Compare actual trajectory vs. "agent A did nothing"

**Implementation**:
```python
# swarmbridge/research/temporal_credit_assignment.py

class TemporalCoordinationCreditAssignment:
    """
    Assigns credit to individual agent actions across time.

    Problem: Handover succeeds at t=50. Credit to:
    - Giver's approach trajectory (t=0-30)?
    - Giver's release timing (t=35)?
    - Receiver's grasp (t=45)?
    """

    def compute_shapley_values(
        self,
        trajectory: MultiActorTrajectory,
        coordination_outcome: float,  # Success metric (1.0 = success, 0.0 = failure)
        causal_graph: DirectedCoordinationGraph,
    ) -> Dict[Tuple[str, int], float]:
        """
        Shapley value for each (agent_id, timestep) action.

        Returns:
            credit_map: {("giver", 35): 0.42, ("receiver", 45): 0.38, ...}

        Interpretation: 42% of coordination success due to giver's action at t=35
        """

    def influence_backpropagation(
        self,
        coordination_success_timestep: int,
        causal_graph: DirectedCoordinationGraph,
        trajectory: MultiActorTrajectory,
    ) -> Dict[Tuple[str, int], float]:
        """
        Backpropagate influence through causal graph.

        Like backprop in neural nets, but for causal coordination graph:
        - Start at success event (e.g., object grasped)
        - Backprop through causal edges with temporal lags
        - Accumulate influence scores
        """

    def counterfactual_removal_credit(
        self,
        trajectory: MultiActorTrajectory,
        actor_id: str,
        timestep: int,
    ) -> float:
        """
        Credit = Actual success - Success if actor did nothing at timestep

        Uses learned SCM to simulate counterfactual trajectories.
        """

# Output Format:
@dataclass
class CreditAssignment:
    per_actor_credit: Dict[str, float]  # Total credit per actor
    per_action_credit: Dict[Tuple[str, int], float]  # Credit per (actor, timestep)
    critical_timesteps: List[int]  # High-credit timesteps (coordination keyframes)

    def visualize_credit_heatmap(self) -> plt.Figure:
        """Heatmap: Actor (y-axis) x Timestep (x-axis), color = credit"""
```

**Use Cases**:
1. **Data augmentation**: Replay demonstrations from critical timesteps only
2. **Curriculum learning**: Train on high-credit sub-trajectories first
3. **Failure diagnosis**: Low credit = coordination breakdown point
4. **Active demonstration**: Ask human to re-demonstrate low-credit segments

**Academic Impact**:
- **Interpretability**: "This coordination succeeded because of agent A's action at t=35"
- **Data efficiency**: Focus on critical coordination moments
- **Venue**: CoRL, ICML (credit assignment is core RL problem)

**Baselines**:
- Uniform credit (all actions equal)
- Temporal distance to success (closer = more credit)
- Gradient-based attribution (if using differentiable policy)

---

### **3. Hierarchical Multi-Actor Imitation Learning (HMAIL)** ⭐⭐

**Academic Gap**:
- Existing work: Hierarchical RL (options, feudal networks), multi-agent IL (MABC)
- Missing: Hierarchical decomposition of multi-actor coordination
- Example: "Collaborative assembly" = [approach, handover, place, validate] sub-skills

**Research Question**:
*"Can we learn hierarchical coordination skills with compositional primitives?"*

**Novel Contribution**:
- **Automatic primitive segmentation** from demonstrations
- **Hierarchical policy**: High-level (which primitive) + Low-level (how to execute)
- **Compositional generalization**: Recombine primitives for new tasks

**Implementation**:
```python
# swarmbridge/research/hierarchical_coordination.py

class HierarchicalCoordinationLearner:
    """
    Learn 2-level hierarchy:
    1. High-level: Sequence of coordination primitives
    2. Low-level: Execution of each primitive

    Example: "Collaborative assembly" =
        [approach, handover, dual_grasp, place_object, validate]
    """

    def segment_primitives(
        self,
        demonstrations: List[MultiActorTrajectory],
        primitive_library: List[CoordinationType],
    ) -> List[SegmentedTrajectory]:
        """
        Segment demonstration into primitive chunks.

        Methods:
        1. Change-point detection (coordination encoder output changes)
        2. HMM with primitive states
        3. Hierarchical clustering of coordination patterns
        """

    def learn_high_level_policy(
        self,
        segmented_demos: List[SegmentedTrajectory],
    ) -> HighLevelPolicy:
        """
        Learn: context -> sequence of primitives

        Architecture: Transformer encoder-decoder
        - Input: Task context (object type, goal)
        - Output: Sequence of primitive IDs [APPROACH, HANDOVER, PLACE]
        """

    def learn_low_level_policies(
        self,
        segmented_demos: List[SegmentedTrajectory],
    ) -> Dict[CoordinationType, CooperativeBCModel]:
        """
        One BC policy per primitive type.

        Shared coordination encoder across primitives for transfer.
        """

# Novel Aspect: Compositional Generalization
class PrimitiveComposer:
    """
    Recombine learned primitives for new tasks.

    Example:
    - Trained on: Task A = [handover, place], Task B = [dual_grasp, rotate]
    - Zero-shot: Task C = [handover, dual_grasp, rotate, place]
    """

    def compose_skill(
        self,
        primitive_sequence: List[CoordinationType],
        low_level_policies: Dict[CoordinationType, CooperativeBCModel],
    ) -> ComposedSkill:
        """
        Chain primitives with transition conditions.

        Transitions:
        - handover -> place: Condition = object in receiver's gripper
        - dual_grasp -> rotate: Condition = both grippers closed
        """
```

**Academic Impact**:
- **Compositional generalization**: Key challenge in AI (Lake et al., Chollet)
- **Data efficiency**: Learn primitives once, reuse for many tasks
- **Venue**: ICLR, NeurIPS, CoRL

**Baselines**:
- Flat BC (no hierarchy)
- Manual primitive segmentation (vs. automatic)
- Single-task learning (vs. compositional transfer)

---

### **4. Privacy-Preserving Federated Multi-Robot Learning** ⭐⭐

**Academic Gap**:
- Existing work: Federated learning (FedAvg), secure aggregation (Bonawitz et al.)
- Missing: **Practical implementation** for multi-robot systems with HE encryption
- SwarmBridge has stubs—complete implementation would be novel

**Research Question**:
*"Can we achieve differential privacy + homomorphic encryption for federated multi-robot IL?"*

**Novel Contribution**:
- **DP-SGD + Pyfhel HE**: Dual privacy (gradient clipping + encryption)
- **Adaptive privacy budget allocation**: More budget for rare coordination patterns
- **Secure multi-party computation** for aggregation

**Implementation** (Complete existing stubs):
```python
# swarmbridge/research/privacy_preserving_fl.py

class DifferentiallyPrivateFederatedTrainer:
    """
    Complete DP-SGD integration with Opacus.

    Currently SwarmBridge has placeholder—this makes it real.
    """

    def __init__(
        self,
        epsilon: float = 1.0,  # Privacy budget
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.5,
    ):
        from opacus import PrivacyEngine
        self.privacy_engine = PrivacyEngine()

    def train_with_dp(
        self,
        model: CooperativeBCModel,
        dataloader: DataLoader,
        epochs: int,
    ) -> Tuple[CooperativeBCModel, float]:
        """
        DP-SGD training loop.

        Returns:
            model: Trained model
            epsilon_spent: Actual privacy budget consumed
        """
        model, optimizer, dataloader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        # Standard training loop
        for epoch in range(epochs):
            for batch in dataloader:
                # Opacus handles per-sample gradients + Gaussian noise
                loss.backward()
                optimizer.step()

        # Get privacy spent
        epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
        return model, epsilon

class HomomorphicEncryptionAggregation:
    """
    Complete Pyfhel integration (currently stub in federated_adapter_flower.py).

    Enables: Aggregate encrypted model updates without decrypting.
    """

    def __init__(self):
        from Pyfhel import Pyfhel, PyCtxt
        self.HE = Pyfhel()
        self.HE.contextGen(scheme='ckks', n=2**14, scale=2**30, qi_sizes=[30]*5)
        self.HE.keyGen()
        self.HE.relinKeyGen()

    def encrypt_model_update(
        self,
        local_model: CooperativeBCModel,
    ) -> List[PyCtxt]:
        """Encrypt all model parameters with CKKS."""
        encrypted_params = []
        for param in local_model.parameters():
            flat_param = param.data.cpu().numpy().flatten()
            encrypted = self.HE.encryptFrac(flat_param)
            encrypted_params.append(encrypted)
        return encrypted_params

    def federated_average_encrypted(
        self,
        encrypted_updates: List[List[PyCtxt]],
    ) -> List[PyCtxt]:
        """
        FedAvg in encrypted space (homomorphic addition + scalar multiplication).

        avg = (1/n) * sum(encrypted_params)
        """
        n_clients = len(encrypted_updates)
        aggregated = []

        for param_idx in range(len(encrypted_updates[0])):
            # Sum all clients' encrypted param
            sum_encrypted = encrypted_updates[0][param_idx]
            for client_idx in range(1, n_clients):
                sum_encrypted += encrypted_updates[client_idx][param_idx]

            # Divide by n (scalar multiplication in HE)
            avg_encrypted = sum_encrypted * (1.0 / n_clients)
            aggregated.append(avg_encrypted)

        return aggregated

# Novel: Adaptive Privacy Budget Allocation
class AdaptivePrivacyBudget:
    """
    Allocate more privacy budget to rare coordination patterns.

    Intuition: Common handovers don't need much privacy, but novel
    coordination failures are rare and informative—spend more budget.
    """

    def allocate_budget(
        self,
        total_epsilon: float,
        pattern_frequencies: Dict[str, int],
    ) -> Dict[str, float]:
        """
        epsilon_pattern = total_epsilon * (1 / frequency)^alpha

        Rare patterns get more budget (less noise).
        """
```

**Academic Impact**:
- **Practical privacy**: Most HE federated learning is theoretical—this is real robots
- **Adaptive budgets**: Novel contribution (not in standard DP-SGD)
- **Venue**: S&P, USENIX Security, ICML (privacy workshop)

**Metrics**:
- Privacy-utility tradeoff curves (epsilon vs. task success rate)
- Computational overhead (HE encryption time vs. plaintext)
- Communication cost (encrypted param size vs. plaintext)

---

### **5. Active Multi-Actor Demonstration Sampling** ⭐⭐

**Academic Gap**:
- Existing work: Active learning for single-agent IL (Judah et al., BAIL)
- Missing: Which multi-actor demonstrations to request from humans?
- Challenge: Multi-actor demos are expensive (need multiple humans or teleoperation)

**Research Question**:
*"How do we select the most informative multi-actor demonstrations to request?"*

**Novel Contribution**:
- **Uncertainty-based sampling**: Request demos with high policy disagreement
- **Coverage-based sampling**: Request demos in unexplored coordination regions
- **Failure-driven sampling**: Request demos after coordination failures

**Implementation**:
```python
# swarmbridge/research/active_demonstration_sampling.py

class ActiveMultiActorSampler:
    """
    Select which multi-actor demonstrations to request from human.

    Goal: Minimize number of expensive multi-actor demos needed.
    """

    def uncertainty_sampling(
        self,
        policy_ensemble: List[CooperativeBCModel],
        candidate_contexts: List[Dict],
    ) -> List[Dict]:
        """
        Sample contexts where policy ensemble has high disagreement.

        Disagreement = variance of predicted actions across ensemble.

        Example: 5 policies trained on different demo subsets.
        - Context A: All policies agree on action -> don't need demo
        - Context B: Policies disagree -> REQUEST DEMO
        """

    def coordination_coverage_sampling(
        self,
        demonstrated_trajectories: List[MultiActorTrajectory],
        causal_graph: DirectedCoordinationGraph,
        candidate_contexts: List[Dict],
    ) -> List[Dict]:
        """
        Sample contexts with unexplored causal coordination patterns.

        Coverage metric: Which edges in causal graph are under-represented?

        Example:
        - Have 50 demos of "giver releases early"
        - Only 2 demos of "giver releases late"
        -> Request demo with late release
        """

    def failure_driven_sampling(
        self,
        failed_executions: List[MultiActorTrajectory],
        failure_analyzer: CausalCoordinationDiscovery,
    ) -> List[Dict]:
        """
        Request demos similar to recent failures.

        Process:
        1. Robot attempts coordination, fails
        2. Use CCD to find causal failure point
        3. Request human demo of similar scenario
        """

# Novel: Multi-Actor Demo Cost Model
class DemonstrationCostModel:
    """
    Account for fact that multi-actor demos are more expensive.

    Costs:
    - Single robot demo: 1x
    - Dual robot demo: 3x (need 2 humans or complex teleoperation)
    - 4-robot swarm demo: 10x
    """

    def expected_value_of_information(
        self,
        context: Dict,
        current_policy: CooperativeBCModel,
        num_actors: int,
    ) -> float:
        """
        EVOI = Expected improvement / Cost

        Only request demo if expected improvement > cost.
        """
```

**Academic Impact**:
- **Data efficiency**: Critical for real robotics (demos are expensive)
- **Multi-actor focus**: Most active IL is single-agent
- **Venue**: CoRL, RSS, ICRA

**Baselines**:
- Random sampling
- Uniform coverage (grid sampling)
- Single-agent active learning methods (adapted)

---

### **6. Cross-Embodiment Coordination Transfer** ⭐⭐

**Academic Gap**:
- Existing work: Zero-shot coordination (Hu et al.), morphology transfer (Hejna et al.)
- Missing: Transfer coordination learned on robot A+B to robot C+D
- Challenge: Observation/action spaces differ, but coordination structure is same

**Research Question**:
*"Can coordination structure transfer across different robot embodiments?"*

**Novel Contribution**:
- **Embodiment-invariant coordination encoding**: Normalize obs/actions
- **Causal structure transfer**: Transfer causal graph, retrain low-level policies
- **Meta-learning for coordination**: Learn to coordinate with unseen partners

**Implementation**:
```python
# swarmbridge/research/cross_embodiment_transfer.py

class EmbodimentInvariantCoordination:
    """
    Learn coordination that transfers across robot types.

    Example: Learn handover on UR5 arms, transfer to Franka Panda arms.
    """

    def normalize_observations(
        self,
        obs: np.ndarray,
        embodiment_config: Dict,
    ) -> np.ndarray:
        """
        Normalize to embodiment-invariant space.

        - Joint angles -> End-effector pose (SE(3))
        - Gripper state -> Binary open/closed
        - Object pose -> Relative to robot base
        """

    def transfer_coordination(
        self,
        source_causal_graph: DirectedCoordinationGraph,
        target_embodiment_configs: List[Dict],
    ) -> CooperativeBCModel:
        """
        Transfer process:
        1. Keep causal graph structure (same handover dependencies)
        2. Retrain low-level policies for target embodiments
        3. Fine-tune with few target demos
        """

class MetaCoordinationLearning:
    """
    MAML for multi-actor coordination.

    Learn coordination that quickly adapts to new partners.
    """

    def meta_train(
        self,
        task_distribution: List[Tuple[RoleConfig, RoleConfig]],
        demonstrations_per_task: int = 10,
    ) -> CooperativeBCModel:
        """
        Meta-training on multiple robot pairs.

        Inner loop: Adapt to specific robot pair
        Outer loop: Meta-update for fast adaptation
        """
```

**Academic Impact**:
- **Generalization**: Core challenge in robotics
- **Practical**: Reuse coordination across different robot fleets
- **Venue**: CoRL, RSS, ICLR

**Metrics**:
- Transfer success rate (train on A+B, test on C+D)
- Few-shot adaptation (demos needed for target embodiment)
- Zero-shot coordination (no target demos at all)

---

### **7. Counterfactual Coordination Reasoning** ⭐

**Academic Gap**:
- Existing work: Counterfactual RL (Buesing et al.), hindsight experience replay
- Missing: Multi-agent counterfactual reasoning for coordination
- Question: "What if the giver agent had released the object 1 second earlier?"

**Research Question**:
*"Can we reason about alternative coordination outcomes using counterfactuals?"*

**Novel Contribution**:
- **Learned world models** for multi-actor dynamics
- **Counterfactual simulation**: "What if agent A did X instead of Y?"
- **Failure diagnosis**: Generate counterfactuals that succeed

**Implementation**:
```python
# swarmbridge/research/counterfactual_reasoning.py

class CounterfactualCoordinationReasoner:
    """
    Answer "what if" questions about coordination.

    Uses learned world model to simulate alternative trajectories.
    """

    def learn_world_model(
        self,
        demonstrations: List[MultiActorTrajectory],
    ) -> MultiActorWorldModel:
        """
        Learn: (s_t, a_1_t, a_2_t, ..., a_n_t) -> s_{t+1}

        Architecture: Transformer or RSSM (Dreamer-style)
        """

    def simulate_counterfactual(
        self,
        actual_trajectory: MultiActorTrajectory,
        intervention: Dict[str, Any],  # {"giver": {"action": release_early}}
        intervention_timestep: int,
    ) -> CounterfactualTrajectory:
        """
        Simulate: What if we intervened at timestep t?

        Process:
        1. Rollout actual trajectory until t
        2. Apply intervention (change agent action)
        3. Rollout from t+1 using world model
        """

    def generate_success_counterfactuals(
        self,
        failed_trajectory: MultiActorTrajectory,
        causal_graph: DirectedCoordinationGraph,
    ) -> List[CounterfactualTrajectory]:
        """
        Search for interventions that would make coordination succeed.

        Use causal graph to guide search:
        - Intervene on high-credit actions (from TCCA)
        - Intervene on causal failure points (from CCD)
        """

# Application: Failure Diagnosis
class CoordinationFailureDiagnosis:
    """
    When coordination fails, explain why and how to fix.
    """

    def diagnose_failure(
        self,
        failed_trajectory: MultiActorTrajectory,
        reasoner: CounterfactualCoordinationReasoner,
    ) -> FailureDiagnosis:
        """
        Output:
        - Root cause: "Giver released too early (t=35)"
        - Counterfactual fix: "If giver waited until t=40, success prob = 0.85"
        - Minimal intervention: "Change only 1 action at t=38"
        """
```

**Academic Impact**:
- **Interpretability**: Explain failures with counterfactuals
- **Debugging**: Guide engineers to fix coordination issues
- **Venue**: ICLR, NeurIPS, ICML

---

## 📈 **Implementation Priority Ranking**

| **Research Direction** | **Academic Impact** | **Implementation Effort** | **Novelty** | **Priority** |
|---|---|---|---|---|
| 1. Causal Coordination Discovery | ⭐⭐⭐ | Medium | Very High | **P0** |
| 2. Temporal Credit Assignment | ⭐⭐⭐ | Medium | High | **P0** |
| 3. Privacy-Preserving FL | ⭐⭐ | Low (complete stubs) | Medium | **P1** |
| 4. Hierarchical MAIL | ⭐⭐ | High | High | **P1** |
| 5. Active Demo Sampling | ⭐⭐ | Medium | Medium | **P2** |
| 6. Cross-Embodiment Transfer | ⭐⭐ | High | Medium | **P2** |
| 7. Counterfactual Reasoning | ⭐ | High | Medium | **P3** |

---

## 🎯 **Recommended First Implementation: Causal Coordination Discovery (CCD)**

**Why CCD is the highest-impact research contribution:**

1. **Foundational**: Other methods build on it (credit assignment uses causal graph, transfer uses causal structure)
2. **Publishable**: No existing work on causal discovery for multi-agent coordination
3. **Practical**: Enables interpretability, debugging, and transfer learning
4. **Feasible**: Granger causality + PC algorithm are well-established tools

**Implementation Steps**:
```bash
# Week 1-2: Core Granger Causality
swarmbridge/research/causal_coordination_discovery.py
- Implement Granger causality test for action->observation influence
- Test on synthetic 2-agent handover data

# Week 3-4: Causal Graph Discovery
- PC algorithm for full causal graph
- Visualization with graphviz
- Test on real robot demonstrations

# Week 5-6: Structural Causal Model Learning
- Learn SCM: o_B(t) = f(a_A(t-k), ...)
- Counterfactual intervention queries
- Test: Predict coordination outcome under interventions

# Week 7-8: Experimental Validation
- Benchmark on 3 coordination primitives (handover, formation, dual-grasp)
- Compare to baselines (correlation, attention weights)
- Metrics: F1 score of causal edges, transfer success rate, counterfactual accuracy
```

**Expected Publications**:
- **CoRL 2026**: "Causal Coordination Discovery for Multi-Robot Imitation Learning"
- **ICML 2026 Workshop**: Causal RL workshop (counterfactual reasoning focus)
- **NeurIPS 2026**: Full paper with hierarchical coordination + CCD

---

## 🔬 **Academic Positioning Strategy**

### **Unique Selling Points**:
1. **Only framework** for causal discovery in multi-agent coordination
2. **Only system** with temporal credit assignment for multi-actor IL
3. **Only practical implementation** of privacy-preserving federated multi-robot learning
4. **Only approach** combining Dynamical skill platform + multi-actor swarm extension

### **Target Venues** (in order):
1. **CoRL 2026**: Causal Coordination Discovery (best paper candidate)
2. **ICRA 2027**: Privacy-Preserving Federated Multi-Robot IL (applications focus)
3. **ICLR 2027**: Hierarchical Multi-Actor IL with Compositional Generalization
4. **NeurIPS 2027**: Full system paper with all 7 contributions

### **Competitive Analysis**:

| **Existing Work** | **Limitation** | **SwarmBridge Advantage** |
|---|---|---|
| QMIX, MAPPO (MARL) | Emergent coordination (black box) | Explicit causal coordination structure |
| CommNet, TarMAC | Learned communication (not causal) | Causal influence discovery |
| MABC (Multi-Agent BC) | Flat temporal modeling | Hierarchical + temporal credit assignment |
| Bonawitz et al. (Secure Agg) | No robotics implementation | Practical HE for multi-robot FL |
| Meta-learning (MAML) | Single-agent focus | Multi-agent coordination meta-learning |

---

## 📚 **Key References to Build On**

### Causal Discovery:
- Spirtes et al. (2000): *Causation, Prediction, and Search* (PC algorithm)
- Granger (1969): "Investigating Causal Relations" (Granger causality)
- Pearl (2009): *Causality* (do-calculus, SCMs)

### Multi-Agent IL:
- Zolna et al. (2022): "Task-Relevant Adversarial Imitation Learning" (RAIL)
- Le et al. (2017): "Coordinated Multi-Agent Imitation Learning" (CMAIL)
- Barde et al. (2023): "Model-Based Multi-Agent Imitation Learning" (MA-MIL)

### Federated Learning + Privacy:
- McMahan et al. (2017): "FedAvg" (foundational)
- Bonawitz et al. (2017): "Secure Aggregation" (HE for FL)
- Abadi et al. (2016): "Deep Learning with Differential Privacy" (DP-SGD)

### Hierarchical RL:
- Bacon et al. (2017): "The Option-Critic Architecture"
- Nachum et al. (2018): "Data-Efficient Hierarchical RL" (HIRO)
- Fox et al. (2017): "Multi-Level Discovery of Deep Options"

---

## 🚀 **Next Steps**

**Immediate Actions** (Week 1):
1. Create `swarmbridge/research/` directory for novel contributions
2. Implement basic Granger causality test on synthetic handover data
3. Set up experiment tracking (Weights & Biases for research experiments)
4. Write research plan document for advisor review

**Month 1 Goal**:
- Working CCD prototype with visualization
- Synthetic dataset of 100 dual-arm handover demonstrations
- Baseline comparisons (correlation vs. causation)

**Month 3 Goal**:
- CoRL 2026 workshop paper submission (4 pages)
- Full experimental validation on 3 coordination primitives

**Month 6 Goal**:
- CoRL 2026 full paper submission (8 pages)
- Open-source release of CCD module

---

## 💡 **Bonus: Potential Collaborations**

**Suggested co-authors / collaborators**:
- **Causal inference experts**: Judea Pearl's group (UCLA), Bernhard Schölkopf (MPI-IS)
- **Multi-agent RL**: Jakob Foerster (Oxford), Shimon Whiteson (Oxford)
- **Robot learning**: Sergey Levine (Berkeley), Chelsea Finn (Stanford), Animesh Garg (Georgia Tech)
- **Privacy-preserving ML**: Nicolas Papernot (Toronto), Florian Tramèr (ETH)

**Potential grant opportunities**:
- NSF CISE: "Causal Inference for Multi-Robot Coordination"
- DARPA TIAMAT: Multi-agent coordination for defense applications
- EU Horizon: Privacy-preserving federated robotics
- OpenPhilanthropy AI Fellowship: Interpretable multi-agent coordination

---

**END OF RESEARCH ROADMAP**
