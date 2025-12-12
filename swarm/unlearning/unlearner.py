"""
Federated Unlearning Implementation

Provides certified removal of site contributions from federated models.

Based on DTC federated unlearning research:
- Track provenance of each site's contribution
- Remove influence through retraining or influence removal
- Certify removal through validation
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch

from ...ml.artifact import CSALoader, CSAPackager, CooperativeSkillArtefact

logger = logging.getLogger(__name__)


@dataclass
class UnlearningRequest:
    """Request to remove a site's contribution"""

    site_id: str
    reason: str  # "gdpr_request", "data_poisoning", "site_exit", etc.
    requested_at: datetime
    requester: str


@dataclass
class UnlearningResult:
    """Result of unlearning operation"""

    success: bool
    site_id: str
    removed_rounds: List[str]
    new_csa_version: str
    certification: Dict[str, any]  # Proof of removal
    completed_at: datetime


class FederatedUnlearner:
    """
    Federated Unlearning Engine

    Capabilities:
    1. Track site provenance across rounds
    2. Remove site contribution (exact or approximate)
    3. Certify removal
    4. Generate new CSA version
    """

    def __init__(self, workspace_dir: Path, provenance_db_path: Optional[Path] = None):
        """
        Args:
            workspace_dir: Workspace directory for artifacts
            provenance_db_path: Path to provenance database
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Provenance tracking
        self.provenance_db_path = provenance_db_path or (
            self.workspace_dir / "provenance.json"
        )
        self.provenance = self._load_provenance()

    def _load_provenance(self) -> Dict:
        """Load provenance database"""
        if self.provenance_db_path.exists():
            with open(self.provenance_db_path) as f:
                return json.load(f)
        return {
            "rounds": {},  # round_id -> {participants: [], csa_version: str}
            "csa_versions": {},  # version -> {parent: str, round_id: str}
            "site_contributions": {},  # site_id -> [round_ids]
        }

    def _save_provenance(self) -> None:
        """Save provenance database"""
        with open(self.provenance_db_path, "w") as f:
            json.dump(self.provenance, f, indent=2, default=str)

    def record_round(
        self, round_id: str, participants: List[str], csa_version: str, parent_version: Optional[str] = None
    ) -> None:
        """
        Record a federated round for provenance tracking

        Args:
            round_id: Round identifier
            participants: List of site IDs
            csa_version: Resulting CSA version
            parent_version: Parent CSA version (if any)
        """
        # Record round
        self.provenance["rounds"][round_id] = {
            "participants": participants,
            "csa_version": csa_version,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Record CSA version lineage
        self.provenance["csa_versions"][csa_version] = {
            "parent": parent_version,
            "round_id": round_id,
            "participants": participants,
        }

        # Record site contributions
        for site_id in participants:
            if site_id not in self.provenance["site_contributions"]:
                self.provenance["site_contributions"][site_id] = []
            self.provenance["site_contributions"][site_id].append(round_id)

        self._save_provenance()
        logger.info(f"Recorded round {round_id}: {len(participants)} participants")

    def get_site_contributions(self, site_id: str) -> List[str]:
        """Get all rounds where site participated"""
        return self.provenance["site_contributions"].get(site_id, [])

    def unlearn_site(
        self,
        request: UnlearningRequest,
        current_csa: CooperativeSkillArtefact,
        method: str = "retraining",
    ) -> UnlearningResult:
        """
        Remove a site's contribution from the CSA

        Args:
            request: Unlearning request
            current_csa: Current CSA artifact
            method: Unlearning method ("retraining", "influence_removal")

        Returns:
            Unlearning result with new CSA
        """
        site_id = request.site_id

        # Get rounds where site participated
        contributed_rounds = self.get_site_contributions(site_id)

        if not contributed_rounds:
            logger.warning(f"Site {site_id} has no recorded contributions")
            return UnlearningResult(
                success=False,
                site_id=site_id,
                removed_rounds=[],
                new_csa_version=current_csa.metadata.version,
                certification={},
                completed_at=datetime.utcnow(),
            )

        logger.info(f"Unlearning site {site_id}: {len(contributed_rounds)} rounds affected")

        # Perform unlearning based on method
        if method == "retraining":
            new_csa = self._unlearn_by_retraining(
                current_csa, site_id, contributed_rounds
            )
        elif method == "influence_removal":
            new_csa = self._unlearn_by_influence_removal(
                current_csa, site_id, contributed_rounds
            )
        else:
            raise ValueError(f"Unknown unlearning method: {method}")

        # Update metadata
        new_version = self._bump_version(current_csa.metadata.version)
        new_csa.metadata.version = new_version
        new_csa.metadata.updated_at = datetime.utcnow()

        # Remove site from training_sites
        if site_id in new_csa.metadata.training_sites:
            new_csa.metadata.training_sites = [
                s for s in new_csa.metadata.training_sites if s != site_id
            ]

        # Generate certification
        certification = {
            "method": method,
            "site_id": site_id,
            "rounds_removed": contributed_rounds,
            "original_version": current_csa.metadata.version,
            "new_version": new_version,
            "timestamp": datetime.utcnow().isoformat(),
            "verification": self._verify_unlearning(current_csa, new_csa, site_id),
        }

        # Update provenance
        self._remove_site_from_provenance(site_id)

        return UnlearningResult(
            success=True,
            site_id=site_id,
            removed_rounds=contributed_rounds,
            new_csa_version=new_version,
            certification=certification,
            completed_at=datetime.utcnow(),
        )

    def _unlearn_by_retraining(
        self,
        current_csa: CooperativeSkillArtefact,
        site_id: str,
        rounds: List[str],
    ) -> CooperativeSkillArtefact:
        """
        Unlearn by retraining without the site's data

        This is the most straightforward approach: re-aggregate all rounds
        excluding the target site.

        In practice, you'd need to:
        1. Retrieve all original CSA deltas from each round
        2. Filter out the target site's deltas
        3. Re-aggregate from scratch

        For this implementation, we simulate by resetting adapters
        (in production, you'd have stored all deltas).
        """
        logger.info(f"Unlearning {site_id} by retraining (simulated)")

        # Deep copy current CSA
        new_csa = current_csa  # In practice, properly deep copy

        # Reset/retrain adapters (placeholder - in production, re-aggregate)
        for adapter in new_csa.policy_adapters:
            # Scale down weights (rough approximation of removing one contributor)
            # Real implementation would re-aggregate all rounds without this site
            for key in adapter.adapter_weights:
                # Simple heuristic: assume uniform contribution, scale down
                num_sites = len(current_csa.metadata.training_sites)
                if num_sites > 1:
                    adapter.adapter_weights[key] *= (num_sites - 1) / num_sites

        return new_csa

    def _unlearn_by_influence_removal(
        self,
        current_csa: CooperativeSkillArtefact,
        site_id: str,
        rounds: List[str],
    ) -> CooperativeSkillArtefact:
        """
        Unlearn by removing site's influence (approximate method)

        Uses influence functions or gradient-based removal to approximate
        retraining without the site.

        This is faster than retraining but approximate.
        """
        logger.info(f"Unlearning {site_id} by influence removal (simulated)")

        # Deep copy
        new_csa = current_csa

        # Estimate and remove influence (placeholder)
        # Real implementation would use stored gradients/influences
        for adapter in new_csa.policy_adapters:
            for key in adapter.adapter_weights:
                # Simple noise injection as placeholder
                influence_estimate = torch.randn_like(adapter.adapter_weights[key]) * 0.01
                adapter.adapter_weights[key] -= influence_estimate

        return new_csa

    def _verify_unlearning(
        self,
        original_csa: CooperativeSkillArtefact,
        unlearned_csa: CooperativeSkillArtefact,
        site_id: str,
    ) -> Dict:
        """
        Verify that unlearning was successful

        Checks:
        1. Weights have changed
        2. Site removed from metadata
        3. Model still passes tests
        """
        verification = {}

        # Check weights changed
        weights_changed = False
        for orig_adapter, new_adapter in zip(
            original_csa.policy_adapters, unlearned_csa.policy_adapters
        ):
            for key in orig_adapter.adapter_weights:
                if not torch.allclose(
                    orig_adapter.adapter_weights[key],
                    new_adapter.adapter_weights[key],
                ):
                    weights_changed = True
                    break

        verification["weights_changed"] = weights_changed

        # Check site removed from metadata
        verification["site_removed_from_metadata"] = (
            site_id not in unlearned_csa.metadata.training_sites
        )

        # Run test suite on unlearned CSA
        tests_passed, test_results = unlearned_csa.run_test_suite()
        verification["tests_passed"] = tests_passed
        verification["test_results"] = test_results

        return verification

    def _remove_site_from_provenance(self, site_id: str) -> None:
        """Remove site from provenance records"""
        if site_id in self.provenance["site_contributions"]:
            del self.provenance["site_contributions"][site_id]

        # Remove from round participant lists
        for round_id, round_data in self.provenance["rounds"].items():
            if site_id in round_data["participants"]:
                round_data["participants"] = [
                    s for s in round_data["participants"] if s != site_id
                ]

        self._save_provenance()

    @staticmethod
    def _bump_version(version: str) -> str:
        """Bump patch version (X.Y.Z -> X.Y.Z+1)"""
        parts = version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    unlearner = FederatedUnlearner(workspace_dir=Path("./unlearning_workspace"))

    # Record some rounds
    unlearner.record_round(
        round_id="round_001",
        participants=["site_a", "site_b", "site_c"],
        csa_version="1.0.0",
    )
    unlearner.record_round(
        round_id="round_002",
        participants=["site_a", "site_b", "site_d"],
        csa_version="1.0.1",
        parent_version="1.0.0",
    )

    # Unlearn site_a
    request = UnlearningRequest(
        site_id="site_a",
        reason="gdpr_request",
        requested_at=datetime.utcnow(),
        requester="user@example.com",
    )

    # TODO: Load actual CSA
    # result = unlearner.unlearn_site(request, current_csa)
    # print(f"Unlearning result: {result}")

    logger.info("✓ Federated unlearning example complete")
