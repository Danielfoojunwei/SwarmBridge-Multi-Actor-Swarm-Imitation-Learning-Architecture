"""
Unit tests for production SwarmBridge components

Tests:
- Flower federated learning adapter
- PyTorch cooperative BC trainer
- Schema validation
- Error handling
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, AsyncMock, patch
import numpy as np


class TestCooperativeBCTrainer:
    """Test production PyTorch trainer"""
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values"""
        from swarmbridge.training.cooperative_bc_trainer import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.coordination_encoder_type == "transformer"
        assert config.policy_hidden_dims == [256, 256, 256]
    
    def test_coordination_encoder_transformer(self):
        """Test transformer coordination encoder"""
        from swarmbridge.training.cooperative_bc_trainer import CoordinationEncoder
        
        encoder = CoordinationEncoder(
            encoder_type="transformer",
            num_actors=2,
            obs_dim=15,
            latent_dim=64,
        )
        
        # Test forward pass
        batch_size = 8
        all_obs = torch.randn(batch_size, 2, 15)
        latent = encoder(all_obs)
        
        assert latent.shape == (batch_size, 64)
        assert not torch.isnan(latent).any()
    
    def test_coordination_encoder_rnn(self):
        """Test RNN coordination encoder"""
        from swarmbridge.training.cooperative_bc_trainer import CoordinationEncoder
        
        encoder = CoordinationEncoder(
            encoder_type="rnn",
            num_actors=2,
            obs_dim=15,
            latent_dim=64,
        )
        
        batch_size = 8
        all_obs = torch.randn(batch_size, 2, 15)
        latent = encoder(all_obs)
        
        assert latent.shape == (batch_size, 64)
    
    def test_coordination_encoder_mlp(self):
        """Test MLP coordination encoder"""
        from swarmbridge.training.cooperative_bc_trainer import CoordinationEncoder
        
        encoder = CoordinationEncoder(
            encoder_type="mlp",
            num_actors=2,
            obs_dim=15,
            latent_dim=64,
        )
        
        batch_size = 8
        all_obs = torch.randn(batch_size, 2, 15)
        latent = encoder(all_obs)
        
        assert latent.shape == (batch_size, 64)
    
    def test_role_conditioned_policy(self):
        """Test role-conditioned policy"""
        from swarmbridge.training.cooperative_bc_trainer import RoleConditionedPolicy
        
        policy = RoleConditionedPolicy(
            role_id="giver",
            obs_dim=15,
            action_dim=7,
            coordination_latent_dim=64,
            hidden_dims=[256, 256],
        )
        
        batch_size = 8
        own_obs = torch.randn(batch_size, 15)
        coord_latent = torch.randn(batch_size, 64)
        
        action = policy(own_obs, coord_latent)
        
        assert action.shape == (batch_size, 7)
        assert not torch.isnan(action).any()
    
    def test_cooperative_bc_model(self):
        """Test complete cooperative BC model"""
        from swarmbridge.training.cooperative_bc_trainer import (
            CooperativeBCModel,
            TrainingConfig,
        )
        
        role_configs = [
            {"role_id": "giver", "observation_dim": 15, "action_dim": 7},
            {"role_id": "receiver", "observation_dim": 15, "action_dim": 7},
        ]
        
        config = TrainingConfig(
            coordination_encoder_type="mlp",
            coordination_latent_dim=32,
            policy_hidden_dims=[128, 128],
        )
        
        model = CooperativeBCModel(
            num_actors=2,
            role_configs=role_configs,
            config=config,
        )
        
        # Test forward pass
        batch_size = 8
        all_obs = torch.randn(batch_size, 2, 15)
        role_ids = ["giver", "receiver"]
        
        actions = model(all_obs, role_ids)
        
        assert "giver" in actions
        assert "receiver" in actions
        assert actions["giver"].shape == (batch_size, 7)
        assert actions["receiver"].shape == (batch_size, 7)


class TestFlowerAdapter:
    """Test Flower federated learning adapter"""
    
    @pytest.mark.asyncio
    async def test_flower_adapter_init(self):
        """Test Flower adapter initialization"""
        from swarmbridge.adapters.federated_adapter_flower import FederatedLearningAdapter
        
        adapter = FederatedLearningAdapter(
            server_address="localhost:8080",
            use_encryption=True,
        )
        
        assert adapter.config.server_address == "localhost:8080"
        assert adapter.config.use_encryption is True
    
    @pytest.mark.asyncio
    async def test_flower_client_get_parameters(self):
        """Test Flower client parameter extraction"""
        from swarmbridge.adapters.federated_adapter_flower import (
            SwarmBridgeFlowerClient,
            FlowerConfig,
        )
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        
        # Create mock data loader
        train_loader = Mock()
        train_loader.dataset = Mock()
        train_loader.dataset.__len__ = Mock(return_value=100)
        
        client = SwarmBridgeFlowerClient(
            model=model,
            train_loader=train_loader,
            config=FlowerConfig(),
        )
        
        params = client.get_parameters(config={})
        
        assert len(params) == 4  # 2 weights + 2 biases
        assert all(isinstance(p, np.ndarray) for p in params)
    
    @pytest.mark.asyncio
    async def test_flower_client_set_parameters(self):
        """Test Flower client parameter setting"""
        from swarmbridge.adapters.federated_adapter_flower import (
            SwarmBridgeFlowerClient,
            FlowerConfig,
        )
        
        model = nn.Sequential(nn.Linear(10, 5))
        train_loader = Mock()
        train_loader.dataset = Mock()
        train_loader.dataset.__len__ = Mock(return_value=100)
        
        client = SwarmBridgeFlowerClient(
            model=model,
            train_loader=train_loader,
            config=FlowerConfig(),
        )
        
        # Get current parameters
        old_params = client.get_parameters(config={})
        
        # Create new parameters
        new_params = [p + 0.1 for p in old_params]
        
        # Set new parameters
        client.set_parameters(new_params)
        
        # Verify parameters changed
        updated_params = client.get_parameters(config={})
        
        for old, new, updated in zip(old_params, new_params, updated_params):
            np.testing.assert_array_almost_equal(new, updated)


class TestSchemas:
    """Test shared schemas"""
    
    def test_role_schema_handover(self):
        """Test role schema for handover coordination"""
        from swarmbridge.schemas import SharedRoleSchema
        
        roles = SharedRoleSchema.create_role_set(2, "handover")
        
        assert len(roles) == 2
        assert roles[0].role_id == "giver"
        assert roles[1].role_id == "receiver"
        assert "handover" in roles[0].capabilities
        assert "receive" in roles[1].capabilities
    
    def test_role_schema_collaborative_manipulation(self):
        """Test role schema for collaborative manipulation"""
        from swarmbridge.schemas import SharedRoleSchema
        
        roles = SharedRoleSchema.create_role_set(2, "collaborative_manipulation")
        
        assert len(roles) == 2
        assert "left_arm" in [r.role_id for r in roles]
        assert "right_arm" in [r.role_id for r in roles]
    
    def test_coordination_primitive_handover(self):
        """Test handover coordination primitive"""
        from swarmbridge.schemas import CoordinationPrimitives, CoordinationType
        
        primitive = CoordinationPrimitives.get_primitive(
            CoordinationType.HANDOVER,
            roles=["giver", "receiver"],
        )
        
        assert primitive.coordination_type == CoordinationType.HANDOVER
        assert len(primitive.participating_roles) == 2
        assert "object_id" in primitive.parameters
    
    def test_coordination_primitive_validation(self):
        """Test primitive validation"""
        from swarmbridge.schemas import CoordinationPrimitives, CoordinationType
        
        # Valid primitive
        valid_primitive = CoordinationPrimitives.get_primitive(
            CoordinationType.HANDOVER,
            roles=["giver", "receiver"],
        )
        is_valid, msg = CoordinationPrimitives.validate_primitive(valid_primitive)
        assert is_valid
        assert msg == "Valid"
        
        # Invalid primitive (wrong number of roles for handover)
        invalid_primitive = CoordinationPrimitives.get_primitive(
            CoordinationType.HANDOVER,
            roles=["actor1", "actor2", "actor3"],  # Should be 2
        )
        is_valid, msg = CoordinationPrimitives.validate_primitive(invalid_primitive)
        assert not is_valid
        assert "2 roles" in msg
    
    def test_task_graph_generation(self):
        """Test SwarmBrain task graph generation"""
        from swarmbridge.schemas import CoordinationPrimitives, CoordinationType
        
        primitive = CoordinationPrimitives.get_primitive(
            CoordinationType.HANDOVER,
            roles=["giver", "receiver"],
        )
        
        task_graph = CoordinationPrimitives.to_swarmbrain_task_graph(primitive)
        
        assert "coordination_type" in task_graph
        assert "tasks" in task_graph
        assert len(task_graph["tasks"]) == 3  # approach, handover_sync, transfer
        
        # Check task dependencies
        task_ids = [t["task_id"] for t in task_graph["tasks"]]
        assert "approach" in task_ids
        assert "handover_sync" in task_ids
        assert "transfer" in task_ids


class TestErrorHandling:
    """Test error handling in production code"""
    
    def test_invalid_encoder_type(self):
        """Test invalid encoder type raises error"""
        from swarmbridge.training.cooperative_bc_trainer import CoordinationEncoder
        
        with pytest.raises(ValueError, match="Unknown encoder type"):
            CoordinationEncoder(
                encoder_type="invalid_type",
                num_actors=2,
                obs_dim=15,
                latent_dim=64,
            )
    
    def test_empty_role_list_validation(self):
        """Test validation with empty role list"""
        from swarmbridge.schemas import CoordinationPrimitives, CoordinationType
        from swarmbridge.schemas.coordination_primitives import CoordinationPrimitive
        
        primitive = CoordinationPrimitive(
            coordination_type=CoordinationType.HANDOVER,
            participating_roles=[],  # Empty
        )
        
        is_valid, msg = CoordinationPrimitives.validate_primitive(primitive)
        assert not is_valid
        assert "No participating roles" in msg
    
    def test_negative_timeout_validation(self):
        """Test validation with negative timeout"""
        from swarmbridge.schemas import CoordinationPrimitives, CoordinationType
        from swarmbridge.schemas.coordination_primitives import CoordinationPrimitive
        
        primitive = CoordinationPrimitive(
            coordination_type=CoordinationType.HANDOVER,
            participating_roles=["giver", "receiver"],
            timeout_s=-1.0,  # Negative
        )
        
        is_valid, msg = CoordinationPrimitives.validate_primitive(primitive)
        assert not is_valid
        assert "positive" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
