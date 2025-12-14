"""
Integration Tests: Dynamical-SIL ↔ Edge Platform

Tests the complete integration layer between systems.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import torch
import numpy as np

from integrations.edge_platform.adapters.csa_to_moe import (
    CSAToMoEAdapter,
    MoESkillValidator,
)
from integrations.edge_platform.bridges.api_bridge import EdgePlatformAPIBridge
from integrations.edge_platform.bridges.encryption_bridge import (
    EncryptionBridge,
    PrivacyBudgetTracker,
)
from integrations.edge_platform.sync.federated_sync import (
    FederatedSyncService,
    SyncMode,
)
from integrations.edge_platform.config.integration_config import IntegrationConfig
from integrations.edge_platform.converters.data_converters import (
    ObservationConverter,
    ActionConverter,
    MetadataConverter,
)


class TestCSAToMoEAdapter:
    """Test CSA to MoE skill conversion"""

    def test_adapter_initialization(self):
        """Test adapter can be initialized"""
        adapter = CSAToMoEAdapter(device="cpu")
        assert adapter.device == "cpu"

    def test_moe_skill_validation(self):
        """Test MoE skill validation"""
        validator = MoESkillValidator()

        # Create a mock valid skill
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            skill_path = Path(tmp.name)

            mock_skill = {
                "metadata": {
                    "skill_name": "test_skill",
                    "version": "1.0",
                    "num_experts": 2,
                    "input_dim": 15,
                    "output_dim": 7,
                },
                "experts": [
                    {"expert_id": "leader", "weights": {"layer1": torch.randn(10, 10)}},
                    {"expert_id": "follower", "weights": {"layer1": torch.randn(10, 10)}},
                ],
                "router": {"fc1.weight": torch.randn(2, 64)},
            }

            torch.save(mock_skill, skill_path)

            is_valid, message = validator.validate_skill(skill_path)
            assert is_valid, f"Validation failed: {message}"

            skill_path.unlink()

    def test_expert_creation(self):
        """Test creating MoE expert from CSA adapter"""
        adapter = CSAToMoEAdapter()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            adapter_path = Path(tmp.name)

            # Create mock adapter weights
            mock_weights = {
                "adapter.weight": torch.randn(10, 10),
                "adapter.bias": torch.randn(10),
            }
            torch.save(mock_weights, adapter_path)

            role_config = {
                "role_id": "leader",
                "observation_dim": 15,
                "action_dim": 7,
            }

            expert = adapter._create_expert_from_adapter(adapter_path, role_config)

            assert expert.expert_id == "leader"
            assert expert.input_dim == 15
            assert expert.output_dim == 7
            assert "adapter.weight" in expert.weights

            adapter_path.unlink()


class TestAPIBridge:
    """Test API bridge between systems"""

    def test_bridge_initialization(self):
        """Test API bridge can be initialized"""
        bridge = EdgePlatformAPIBridge(
            sil_registry_url="http://localhost:8000",
            edge_api_url="http://localhost:8001",
        )

        assert bridge.sil_url == "http://localhost:8000"
        assert bridge.edge_url == "http://localhost:8001"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check (will fail if services not running)"""
        bridge = EdgePlatformAPIBridge(
            sil_registry_url="http://localhost:8000",
            edge_api_url="http://localhost:8001",
        )

        health = await bridge.health_check()

        assert "sil_registry" in health
        assert "edge_platform" in health
        # Note: Will be False if services not running


class TestEncryptionBridge:
    """Test encryption bridge"""

    def test_encryption_initialization(self):
        """Test encryption bridge initialization"""
        bridge = EncryptionBridge()
        assert bridge is not None

    def test_encrypt_for_edge(self):
        """Test encrypting weights for Edge Platform"""
        bridge = EncryptionBridge()

        weights = torch.randn(10, 10)
        encrypted = bridge.encrypt_for_edge(weights, key_id="test_key")

        assert encrypted.encryption_context.scheme == "N2HE"
        assert encrypted.shape == (10, 10)
        assert len(encrypted.checksum) == 64  # SHA256 hex

    def test_decrypt_from_edge(self):
        """Test decrypting weights from Edge Platform"""
        bridge = EncryptionBridge()

        original_weights = torch.randn(10, 10)

        # Encrypt then decrypt
        encrypted = bridge.encrypt_for_edge(original_weights)
        decrypted = bridge.decrypt_from_edge(encrypted)

        assert decrypted.shape == original_weights.shape

    def test_encrypt_for_sil(self):
        """Test encrypting weights for Dynamical-SIL"""
        bridge = EncryptionBridge()

        weights = torch.randn(10, 10)
        encrypted = bridge.encrypt_for_sil(weights, scheme="CKKS")

        assert encrypted.encryption_context.scheme == "CKKS"

    def test_aggregate_encrypted(self):
        """Test homomorphic aggregation"""
        bridge = EncryptionBridge()

        # Create multiple encrypted weights
        weights_list = [torch.randn(10, 10) for _ in range(3)]
        encrypted_list = [bridge.encrypt_for_edge(w) for w in weights_list]

        # Aggregate
        aggregated = bridge.aggregate_encrypted(encrypted_list, strategy="mean")

        assert aggregated.shape == (10, 10)
        assert aggregated.encryption_context.scheme == "N2HE"

    def test_privacy_budget_tracker(self):
        """Test privacy budget tracking"""
        tracker = PrivacyBudgetTracker()

        tracker.add_dp_operation(
            epsilon=1.0,
            delta=1e-5,
            operation="gradient_release",
            system="dynamical_sil",
        )

        tracker.add_he_operation(
            depth=2,
            operation="encrypted_aggregation",
            system="edge_platform",
        )

        status = tracker.get_budget_status()

        assert status["differential_privacy"]["epsilon_total"] == 1.0
        assert status["homomorphic_encryption"]["depth_used"] == 2

        # Check if exceeded
        assert not tracker.is_budget_exceeded(epsilon_limit=10.0)
        assert tracker.is_budget_exceeded(epsilon_limit=0.5)


class TestFederatedSync:
    """Test federated synchronization service"""

    def test_sync_service_initialization(self):
        """Test sync service initialization"""
        sync = FederatedSyncService(
            sil_coordinator_url="http://localhost:8000",
            edge_api_url="http://localhost:8001",
        )

        assert sync.round_counter == 0
        assert len(sync.active_rounds) == 0

    def test_round_tracking(self):
        """Test round metadata tracking"""
        sync = FederatedSyncService(
            sil_coordinator_url="http://localhost:8000",
            edge_api_url="http://localhost:8001",
        )

        # Manually create a round for testing
        from integrations.edge_platform.sync.federated_sync import FederatedRound
        from datetime import datetime

        round_info = FederatedRound(
            round_id=1,
            start_time=datetime.now(),
            sil_sites=["site_1", "site_2"],
            edge_devices=["device_1"],
            status="training",
        )

        sync.active_rounds[1] = round_info

        rounds = sync.get_all_rounds()
        assert len(rounds) == 1
        assert rounds[0]["round_id"] == 1
        assert rounds[0]["num_sil_sites"] == 2
        assert rounds[0]["num_edge_devices"] == 1


class TestDataConverters:
    """Test data format converters"""

    def test_observation_converter_sil_to_edge(self):
        """Test converting SIL observations to Edge format"""
        sil_obs = {
            "multi_actor_observations": np.random.randn(2, 10, 15),  # [N, T, D]
        }

        edge_obs = ObservationConverter.sil_to_edge(sil_obs)

        assert "visual" in edge_obs
        assert isinstance(edge_obs["visual"], torch.Tensor)

    def test_observation_converter_edge_to_sil(self):
        """Test converting Edge observations to SIL format"""
        edge_obs = {
            "visual": torch.randn(10, 15),
            "proprioception": torch.randn(7),
            "language_instruction": "pick up the cube",
        }

        sil_obs = ObservationConverter.edge_to_sil(edge_obs, num_actors=3)

        assert "multi_actor_observations" in sil_obs
        assert sil_obs["multi_actor_observations"].shape[0] == 3  # 3 actors

    def test_action_converter(self):
        """Test action conversion"""
        # SIL to Edge
        sil_actions = {
            "multi_actor_actions": np.random.randn(2, 7),  # [N, A]
        }

        edge_actions = ActionConverter.sil_to_edge(sil_actions)
        assert isinstance(edge_actions, torch.Tensor)

        # Edge to SIL
        edge_action = torch.randn(7)
        sil_actions_back = ActionConverter.edge_to_sil(edge_action, num_actors=2)

        assert "multi_actor_actions" in sil_actions_back
        assert sil_actions_back["multi_actor_actions"].shape == (2, 7)

    def test_metadata_converter(self):
        """Test metadata conversion"""
        csa_meta = {
            "skill_name": "cooperative_assembly",
            "version": "1.0",
            "roles": [
                {"role_id": "leader", "capabilities": ["grasp"]},
                {"role_id": "follower", "capabilities": ["support"]},
            ],
            "observation_dim": 15,
            "action_dim": 7,
            "csa_id": "csa_123",
        }

        skill_meta = MetadataConverter.csa_metadata_to_skill_metadata(csa_meta)

        assert skill_meta["skill_name"] == "cooperative_assembly"
        assert skill_meta["num_experts"] == 2
        assert "leader" in skill_meta["expert_specializations"]

        # Reverse conversion
        csa_meta_back = MetadataConverter.skill_metadata_to_csa_metadata(skill_meta)

        assert csa_meta_back["skill_name"] == "cooperative_assembly"
        assert csa_meta_back["num_actors"] == 2


class TestIntegrationConfig:
    """Test integration configuration"""

    def test_default_config(self):
        """Test creating default configuration"""
        config = IntegrationConfig()

        assert config.endpoints.sil_registry == "http://localhost:8000"
        assert config.encryption.security_bits == 128
        assert config.federated.num_sil_sites == 3
        assert config.sync.auto_sync is True

    def test_config_save_load(self):
        """Test saving and loading configuration"""
        config = IntegrationConfig()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            config_path = Path(tmp.name)

            # Save
            config.to_yaml(config_path)

            # Load
            loaded_config = IntegrationConfig.from_yaml(config_path)

            assert loaded_config.endpoints.sil_registry == config.endpoints.sil_registry
            assert loaded_config.encryption.security_bits == config.encryption.security_bits

            config_path.unlink()


@pytest.mark.integration
class TestEndToEndIntegration:
    """End-to-end integration tests (requires both systems running)"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires both systems running")
    async def test_full_csa_to_skill_workflow(self):
        """Test complete workflow: CSA → MoE → Upload → Download → CSA"""

        # This would test:
        # 1. Create a mock CSA
        # 2. Convert to MoE skill
        # 3. Upload to Edge Platform
        # 4. Download from Edge Platform
        # 5. Convert back to CSA
        # 6. Verify integrity

        pass

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires both systems running")
    async def test_federated_round_workflow(self):
        """Test complete federated round across both systems"""

        # This would test:
        # 1. Start coordinated round
        # 2. Monitor progress
        # 3. Aggregate results
        # 4. Distribute updates
        # 5. Verify convergence

        pass
