"""
Tri-System Integration Tests

Tests the complete integration across:
- Dynamical-SIL (training)
- Edge Platform (deployment)
- SwarmBrain (orchestration)
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import torch
import numpy as np

from integrations.swarmbrain.adapters.csa_to_swarmbrain import (
    CSAToSwarmBrainAdapter,
)
from integrations.swarmbrain.orchestration.mission_bridge import (
    SwarmBrainMissionBridge,
)
from integrations.tri_system.coordinator.unified_coordinator import (
    TriSystemCoordinator,
    WorkflowStage,
)
from integrations.tri_system.encryption.unified_encryption import (
    UnifiedEncryptionBridge,
    EncryptionScheme,
    TriSystemPrivacyBudgetTracker,
)
from integrations.tri_system.config.tri_system_config import TriSystemConfig


class TestCSAToSwarmBrainAdapter:
    """Test CSA to SwarmBrain conversion"""

    def test_adapter_initialization(self):
        """Test adapter can be initialized"""
        adapter = CSAToSwarmBrainAdapter()
        assert adapter is not None

    def test_task_graph_creation(self):
        """Test task graph generation for different coordination types"""
        adapter = CSAToSwarmBrainAdapter()

        manifest = {
            "skill_name": "test_skill",
            "roles": [
                {"role_id": "leader"},
                {"role_id": "follower"},
            ],
        }

        # Test handover coordination
        task_graph = adapter._create_task_graph(manifest, "handover")
        assert "tasks" in task_graph
        assert any(t["task_id"] == "handover_sync" for t in task_graph["tasks"])

        # Test mutex coordination
        task_graph = adapter._create_task_graph(manifest, "mutex")
        assert any(t.get("coordination", {}).get("type") == "mutex" for t in task_graph["tasks"])

        # Test barrier coordination
        task_graph = adapter._create_task_graph(manifest, "barrier")
        assert any(t.get("coordination", {}).get("type") == "barrier" for t in task_graph["tasks"])


class TestSwarmBrainMissionBridge:
    """Test SwarmBrain mission bridge"""

    def test_bridge_initialization(self):
        """Test mission bridge initialization"""
        bridge = SwarmBrainMissionBridge(
            swarmbrain_url="http://localhost:8000",
            sil_registry_url="http://localhost:8001",
            edge_api_url="http://localhost:8002",
        )

        assert bridge.swarmbrain_url == "http://localhost:8000"
        assert bridge.sil_url == "http://localhost:8001"
        assert bridge.edge_url == "http://localhost:8002"


class TestTriSystemCoordinator:
    """Test tri-system unified coordinator"""

    def test_coordinator_initialization(self):
        """Test coordinator initialization"""
        coordinator = TriSystemCoordinator(
            sil_registry_url="http://localhost:8000",
            sil_coordinator_url="http://localhost:8001",
            edge_api_url="http://localhost:8002",
            swarmbrain_url="http://localhost:8003",
        )

        assert coordinator.workflow_counter == 0
        assert len(coordinator.active_workflows) == 0

    def test_workflow_tracking(self):
        """Test workflow tracking"""
        coordinator = TriSystemCoordinator(
            sil_registry_url="http://localhost:8000",
            sil_coordinator_url="http://localhost:8001",
            edge_api_url="http://localhost:8002",
            swarmbrain_url="http://localhost:8003",
        )

        # Manually create a workflow for testing
        from integrations.tri_system.coordinator.unified_coordinator import TriSystemWorkflow
        from datetime import datetime

        workflow = TriSystemWorkflow(
            workflow_id="test_workflow",
            skill_name="test_skill",
            current_stage=WorkflowStage.TRAINING,
        )

        coordinator.active_workflows["test_workflow"] = workflow

        workflows = coordinator.list_workflows()
        assert len(workflows) == 1
        assert workflows[0]["workflow_id"] == "test_workflow"
        assert workflows[0]["current_stage"] == "training"


class TestUnifiedEncryptionBridge:
    """Test unified encryption bridge"""

    def test_bridge_initialization(self):
        """Test encryption bridge initialization"""
        bridge = UnifiedEncryptionBridge()
        assert bridge is not None

    def test_encrypt_for_dynamical_sil(self):
        """Test encrypting for Dynamical-SIL"""
        bridge = UnifiedEncryptionBridge()

        weights = torch.randn(10, 10)
        encrypted = bridge.encrypt(
            weights,
            target_system="dynamical_sil",
            scheme=EncryptionScheme.PYFHEL_CKKS,
        )

        assert "encrypted_data" in encrypted
        assert encrypted["context"].source_system == "dynamical_sil"
        assert encrypted["context"].scheme == EncryptionScheme.PYFHEL_CKKS

    def test_encrypt_for_edge_platform(self):
        """Test encrypting for Edge Platform"""
        bridge = UnifiedEncryptionBridge()

        weights = torch.randn(10, 10)
        encrypted = bridge.encrypt(
            weights,
            target_system="edge_platform",
        )

        assert encrypted["context"].source_system == "edge_platform"
        assert encrypted["context"].scheme == EncryptionScheme.N2HE_128

    def test_encrypt_for_swarmbrain(self):
        """Test encrypting for SwarmBrain"""
        bridge = UnifiedEncryptionBridge()

        weights = torch.randn(10, 10)
        encrypted = bridge.encrypt(
            weights,
            target_system="swarmbrain",
            scheme=EncryptionScheme.OPENFHE_BFV,
        )

        assert encrypted["context"].source_system == "swarmbrain"
        assert encrypted["context"].scheme == EncryptionScheme.OPENFHE_BFV

    def test_encrypt_decrypt_round_trip(self):
        """Test encryption and decryption round trip"""
        bridge = UnifiedEncryptionBridge()

        original_weights = torch.randn(10, 10)

        # Test all systems
        for system in ["dynamical_sil", "edge_platform", "swarmbrain"]:
            encrypted = bridge.encrypt(original_weights, target_system=system)
            decrypted = bridge.decrypt(encrypted)

            assert decrypted.shape == original_weights.shape

    def test_cross_system_aggregation(self):
        """Test cross-system encrypted aggregation"""
        bridge = UnifiedEncryptionBridge()

        # Create encrypted weights from all three systems
        weights1 = torch.randn(10, 10)
        weights2 = torch.randn(10, 10)
        weights3 = torch.randn(10, 10)

        encrypted_sil = bridge.encrypt(weights1, target_system="dynamical_sil")
        encrypted_edge = bridge.encrypt(weights2, target_system="edge_platform")
        encrypted_swarm = bridge.encrypt(weights3, target_system="swarmbrain")

        # Aggregate across all systems
        aggregated = bridge.aggregate_cross_system(
            [encrypted_sil, encrypted_edge, encrypted_swarm],
            strategy="mean",
        )

        assert aggregated["context"].source_system == "tri_system_aggregated"
        assert aggregated["context"].scheme == EncryptionScheme.OPENFHE_CKKS

        # Decrypt and verify shape
        decrypted = bridge.decrypt(aggregated)
        assert decrypted.shape == weights1.shape


class TestTriSystemPrivacyBudgetTracker:
    """Test tri-system privacy budget tracking"""

    def test_tracker_initialization(self):
        """Test tracker initialization"""
        tracker = TriSystemPrivacyBudgetTracker()

        assert len(tracker.budgets) == 3
        assert "dynamical_sil" in tracker.budgets
        assert "edge_platform" in tracker.budgets
        assert "swarmbrain" in tracker.budgets

    def test_add_operation(self):
        """Test adding privacy operations"""
        tracker = TriSystemPrivacyBudgetTracker()

        tracker.add_operation(
            system="dynamical_sil",
            operation_type="dp",
            epsilon=1.0,
            delta=1e-5,
        )

        tracker.add_operation(
            system="edge_platform",
            operation_type="he",
            he_depth=2,
        )

        tracker.add_operation(
            system="swarmbrain",
            operation_type="dp",
            epsilon=0.5,
            delta=1e-6,
        )

        total = tracker.get_total_budget()

        assert total["total_epsilon"] == 1.5
        assert total["total_delta"] == pytest.approx(1.1e-5)
        assert total["total_he_depth"] == 2
        assert total["total_operations"] == 3

    def test_budget_exceeded(self):
        """Test budget exceeded detection"""
        tracker = TriSystemPrivacyBudgetTracker()

        # Add operations
        tracker.add_operation("dynamical_sil", "dp", epsilon=5.0, delta=1e-6)
        tracker.add_operation("edge_platform", "dp", epsilon=6.0, delta=1e-6)

        # Should not exceed default limits
        assert not tracker.is_budget_exceeded(epsilon_limit=20.0)

        # Should exceed lower limit
        assert tracker.is_budget_exceeded(epsilon_limit=5.0)


class TestTriSystemConfig:
    """Test tri-system configuration"""

    def test_default_config(self):
        """Test creating default configuration"""
        config = TriSystemConfig()

        assert config.endpoints.sil_registry == "http://localhost:8000"
        assert config.endpoints.swarmbrain_orchestrator == "http://localhost:8003"
        assert config.workflow.sil_training_rounds == 5
        assert config.workflow.swarm_num_robots == 3
        assert config.encryption.security_bits == 128
        assert config.privacy.epsilon_limit == 10.0

    def test_config_save_load(self):
        """Test saving and loading configuration"""
        config = TriSystemConfig()
        config.workflow.sil_training_rounds = 10
        config.workflow.swarm_num_robots = 5

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            config_path = Path(tmp.name)

            # Save
            config.to_yaml(config_path)

            # Load
            loaded_config = TriSystemConfig.from_yaml(config_path)

            assert loaded_config.workflow.sil_training_rounds == 10
            assert loaded_config.workflow.swarm_num_robots == 5

            config_path.unlink()


@pytest.mark.integration
class TestEndToEndTriSystem:
    """End-to-end tri-system integration tests (requires all systems running)"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all three systems running")
    async def test_complete_workflow(self):
        """Test complete end-to-end workflow across all systems"""

        coordinator = TriSystemCoordinator(
            sil_registry_url="http://localhost:8000",
            sil_coordinator_url="http://localhost:8001",
            edge_api_url="http://jetson-orin:8002",
            swarmbrain_url="http://localhost:8003",
        )

        # Start complete workflow
        workflow_id = await coordinator.start_complete_workflow(
            skill_name="cooperative_assembly",
            num_sil_sites=3,
            num_edge_devices=2,
            num_robots=3,
            work_order={
                "task_type": "assembly",
                "objects": ["cube_red", "cube_blue"],
            },
            sil_training_rounds=2,
            coordination_type="handover",
        )

        # Verify workflow was created
        assert workflow_id.startswith("workflow_")

        # Check status
        status = await coordinator.get_workflow_status(workflow_id)
        assert status["workflow_id"] == workflow_id
        assert status["skill_name"] == "cooperative_assembly"


@pytest.mark.integration
class TestDataFlow:
    """Test data flow between systems"""

    def test_csa_to_swarmbrain_to_edge_flow(self):
        """
        Test data flow:
        CSA → SwarmBrain skill → Execute → Edge skill update
        """

        # This test would verify:
        # 1. CSA converted to SwarmBrain format
        # 2. Mission executed using SwarmBrain
        # 3. Learned improvements sent to Edge Platform
        # 4. Edge skill updated and redistributed

        pass

    def test_tri_system_federated_learning_flow(self):
        """
        Test federated learning flow:
        SIL training → Edge deployment → SwarmBrain execution → FL round → Update all
        """

        # This test would verify:
        # 1. Initial training on SIL (OpenFL)
        # 2. Deployment to Edge (MoE format)
        # 3. Execution on SwarmBrain (Flower FL)
        # 4. Aggregated updates distributed to all three systems

        pass
