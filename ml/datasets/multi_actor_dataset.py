"""
Advanced Multi-Actor Dataset Handler

Features:
- Variable number of actors support
- Temporal synchronization across actors
- Role-specific data augmentation
- Formation-aware sampling
- Intent annotation integration
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset


class MultiActorDemonstrationDataset(Dataset):
    """
    Enhanced dataset for multi-actor demonstrations

    Handles:
    - Variable actor counts (2-N actors)
    - Time-synchronized observations
    - Role labels and intent annotations
    - Formation configurations
    - Object state tracking
    """

    def __init__(
        self,
        dataset_path: Path,
        sequence_length: int = 16,
        actors_per_sample: Optional[int] = None,  # None = variable
        augment: bool = True,
        formation_aware: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.sequence_length = sequence_length
        self.actors_per_sample = actors_per_sample
        self.augment = augment
        self.formation_aware = formation_aware

        # Load dataset
        self.demonstrations = self._load_demonstrations()

        # Build index
        self.index = self._build_index()

        # Compute normalization statistics
        self.obs_mean, self.obs_std = self._compute_normalization_stats()

    def _load_demonstrations(self) -> List[Dict]:
        """Load multi-actor demonstrations from HDF5"""
        demonstrations = []

        with h5py.File(self.dataset_path, 'r') as f:
            num_demos = len([k for k in f.keys() if k.startswith('demo_')])

            for demo_idx in range(num_demos):
                demo_key = f'demo_{demo_idx}'
                demo_group = f[demo_key]

                demo = {
                    'length': demo_group.attrs.get('length', 0),
                    'num_actors': demo_group.attrs.get('num_actors', 0),
                    'actors': {},
                    'formation': demo_group.attrs.get('formation', 'free'),
                    'task_type': demo_group.attrs.get('task_type', 'unknown'),
                }

                # Load per-actor data
                for actor_key in demo_group.keys():
                    if actor_key.startswith('actor_'):
                        actor_group = demo_group[actor_key]
                        demo['actors'][actor_key] = {
                            'role': actor_group.attrs.get('role', 'unknown'),
                            'observations': actor_group['observations'][:],
                            'actions': actor_group['actions'][:],
                            'intents': actor_group.get('intents', None),
                            'timestamps': actor_group.get('timestamps', None),
                        }

                demonstrations.append(demo)

        return demonstrations

    def _build_index(self) -> List[Tuple[int, int, List[str]]]:
        """
        Build index of valid sequences

        Returns:
            List of (demo_id, start_timestep, selected_actors)
        """
        index = []

        for demo_id, demo in enumerate(self.demonstrations):
            demo_length = demo['length']
            num_actors = demo['num_actors']

            # If fixed actor count, sample that many
            if self.actors_per_sample:
                # Generate all valid windows
                for t in range(demo_length - self.sequence_length + 1):
                    # Randomly select subset of actors if needed
                    actor_ids = list(demo['actors'].keys())
                    if len(actor_ids) > self.actors_per_sample:
                        # Multiple ways to select actors
                        import itertools
                        for actor_subset in itertools.combinations(
                            actor_ids, self.actors_per_sample
                        ):
                            index.append((demo_id, t, list(actor_subset)))
                    else:
                        index.append((demo_id, t, actor_ids))
            else:
                # Use all actors
                for t in range(demo_length - self.sequence_length + 1):
                    index.append((demo_id, t, list(demo['actors'].keys())))

        return index

    def _compute_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std for observation normalization"""
        all_obs = []

        for demo in self.demonstrations:
            for actor_data in demo['actors'].values():
                all_obs.append(actor_data['observations'])

        if all_obs:
            all_obs = np.concatenate(all_obs, axis=0)
            mean = all_obs.mean(axis=0)
            std = all_obs.std(axis=0) + 1e-8
        else:
            # Default normalization
            mean = np.zeros(1)
            std = np.ones(1)

        return mean, std

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get multi-actor sequence

        Returns:
            Dict with:
            - multi_actor_observations: [num_actors, seq_len, obs_dim]
            - multi_actor_actions: [num_actors, seq_len, action_dim]
            - multi_actor_intents: [num_actors, seq_len, num_intent_types]
            - role_ids: List[str]
            - formation_type: str
            - current_observations: [num_actors, obs_dim]
            - current_actions: [num_actors, action_dim]
        """
        demo_id, start_t, actor_ids = self.index[idx]
        demo = self.demonstrations[demo_id]
        end_t = start_t + self.sequence_length

        # Extract sequences for each actor
        observations_list = []
        actions_list = []
        intents_list = []
        role_ids = []

        for actor_id in actor_ids:
            actor_data = demo['actors'][actor_id]

            # Extract sequence
            obs_seq = actor_data['observations'][start_t:end_t]
            action_seq = actor_data['actions'][start_t:end_t]

            # Normalize observations
            obs_seq = (obs_seq - self.obs_mean) / self.obs_std

            # Augmentation
            if self.augment:
                obs_seq, action_seq = self._augment_sequence(obs_seq, action_seq)

            observations_list.append(obs_seq)
            actions_list.append(action_seq)

            # Intent sequence (if available)
            if actor_data['intents'] is not None:
                intent_seq = actor_data['intents'][start_t:end_t]
                intents_list.append(intent_seq)

            role_ids.append(actor_data['role'])

        # Stack into tensors
        multi_actor_obs = torch.from_numpy(
            np.stack(observations_list)
        ).float()  # [num_actors, seq_len, obs_dim]

        multi_actor_actions = torch.from_numpy(
            np.stack(actions_list)
        ).float()  # [num_actors, seq_len, action_dim]

        if intents_list:
            multi_actor_intents = torch.from_numpy(
                np.stack(intents_list)
            ).float()
        else:
            # Default: all "move" intent
            multi_actor_intents = torch.zeros(
                len(actor_ids), self.sequence_length, 6
            )  # 6 intent types
            multi_actor_intents[:, :, 1] = 1.0  # "move" intent

        # Current state (last timestep)
        current_obs = multi_actor_obs[:, -1, :]
        current_actions = multi_actor_actions[:, -1, :]

        return {
            'multi_actor_observations': multi_actor_obs,
            'multi_actor_actions': multi_actor_actions,
            'multi_actor_intents': multi_actor_intents,
            'role_ids': role_ids,
            'formation_type': demo['formation'],
            'task_type': demo['task_type'],
            'current_observations': current_obs,
            'current_actions': current_actions,
            'num_actors': len(actor_ids),
        }

    def _augment_sequence(
        self, obs_seq: np.ndarray, action_seq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data augmentation for multi-actor sequences

        Augmentations:
        - Temporal jittering
        - Noise injection
        - Spatial perturbations (for positions)
        """
        # Noise injection
        obs_noise = np.random.normal(0, 0.01, obs_seq.shape)
        obs_seq = obs_seq + obs_noise

        action_noise = np.random.normal(0, 0.005, action_seq.shape)
        action_seq = action_seq + action_noise

        # Temporal jittering (slight time shifts)
        if np.random.rand() < 0.3:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                obs_seq = np.roll(obs_seq, shift, axis=0)
                action_seq = np.roll(action_seq, shift, axis=0)

        return obs_seq, action_seq


class FormationAwareSampler:
    """
    Smart sampling based on formation types

    Ensures balanced sampling across different formations:
    - Line formation
    - Triangle formation
    - Square formation
    - Free-form coordination
    """

    def __init__(self, dataset: MultiActorDemonstrationDataset):
        self.dataset = dataset
        self.formation_indices = self._group_by_formation()

    def _group_by_formation(self) -> Dict[str, List[int]]:
        """Group indices by formation type"""
        formation_groups = {}

        for idx, (demo_id, start_t, actor_ids) in enumerate(self.dataset.index):
            formation = self.dataset.demonstrations[demo_id]['formation']

            if formation not in formation_groups:
                formation_groups[formation] = []

            formation_groups[formation].append(idx)

        return formation_groups

    def sample_balanced(self, batch_size: int) -> List[int]:
        """Sample ensuring formation diversity"""
        samples_per_formation = batch_size // len(self.formation_indices)
        indices = []

        for formation, formation_idx_list in self.formation_indices.items():
            sampled = np.random.choice(
                formation_idx_list,
                size=min(samples_per_formation, len(formation_idx_list)),
                replace=False,
            )
            indices.extend(sampled.tolist())

        # Fill remaining
        if len(indices) < batch_size:
            all_indices = [i for group in self.formation_indices.values() for i in group]
            remaining = batch_size - len(indices)
            extra = np.random.choice(all_indices, size=remaining, replace=False)
            indices.extend(extra.tolist())

        np.random.shuffle(indices)
        return indices[:batch_size]


def create_multi_actor_dataloader(
    dataset_path: Path,
    batch_size: int = 32,
    sequence_length: int = 16,
    num_workers: int = 4,
    formation_balanced: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for multi-actor training

    Args:
        dataset_path: Path to HDF5 dataset
        batch_size: Batch size
        sequence_length: Sequence length for temporal data
        num_workers: Number of data loading workers
        formation_balanced: Whether to use formation-aware sampling

    Returns:
        DataLoader instance
    """
    dataset = MultiActorDemonstrationDataset(
        dataset_path=dataset_path,
        sequence_length=sequence_length,
        augment=True,
        formation_aware=formation_balanced,
    )

    if formation_balanced:
        sampler = FormationAwareSampler(dataset)

        # Custom batch sampler
        class FormationBalancedBatchSampler:
            def __init__(self, sampler, batch_size):
                self.sampler = sampler
                self.batch_size = batch_size
                self.num_batches = len(dataset) // batch_size

            def __iter__(self):
                for _ in range(self.num_batches):
                    yield self.sampler.sample_balanced(self.batch_size)

            def __len__(self):
                return self.num_batches

        batch_sampler = FormationBalancedBatchSampler(sampler, batch_size)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloader
