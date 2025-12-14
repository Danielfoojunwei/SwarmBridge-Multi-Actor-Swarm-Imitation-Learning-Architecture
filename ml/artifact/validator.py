"""
CSA Validation Framework

Runs deterministic offline tests and invariant checks on CSA artifacts
before deployment or swarm aggregation.
"""

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import numpy as np

from .schema import CooperativeSkillArtefact


class CSAValidator:
    """Validate CSA artifacts against safety and correctness invariants"""

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator

        Args:
            strict_mode: If True, any test failure rejects the CSA
        """
        self.strict_mode = strict_mode

    def validate(self, csa: CooperativeSkillArtefact) -> Tuple[bool, Dict[str, Any]]:
        """
        Run complete validation suite

        Returns:
            (is_valid, detailed_results)
        """
        results = {}
        is_valid = True

        # 1. Schema validation
        schema_valid, schema_results = self._validate_schema(csa)
        results["schema"] = schema_results
        if not schema_valid:
            is_valid = False

        # 2. Phase machine validation
        phase_valid, phase_results = self._validate_phase_machine(csa.phase_machine_xml)
        results["phase_machine"] = phase_results
        if not phase_valid:
            is_valid = False

        # 3. Safety envelope validation
        safety_valid, safety_results = self._validate_safety_envelope(csa)
        results["safety_envelope"] = safety_results
        if not safety_valid:
            is_valid = False

        # 4. Role-adapter consistency
        role_valid, role_results = self._validate_role_consistency(csa)
        results["role_consistency"] = role_results
        if not role_valid:
            is_valid = False

        # 5. Coordination encoder validation
        coord_valid, coord_results = self._validate_coordination_encoder(csa)
        results["coordination_encoder"] = coord_results
        if not coord_valid:
            is_valid = False

        # 6. Metadata completeness
        meta_valid, meta_results = self._validate_metadata(csa)
        results["metadata"] = meta_results
        if not meta_valid:
            is_valid = False

        # 7. Run CSA's own test suite
        suite_valid, suite_results = csa.run_test_suite()
        results["test_suite"] = suite_results
        if not suite_valid:
            is_valid = False

        return is_valid, results

    def _validate_schema(self, csa: CooperativeSkillArtefact) -> Tuple[bool, Dict[str, Any]]:
        """Validate basic schema requirements"""
        results = {}
        is_valid = True

        # Check minimum required actors
        if len(csa.roles) < csa.metadata.min_actors:
            results["min_actors"] = False
            is_valid = False
        else:
            results["min_actors"] = True

        # Check maximum actors
        if len(csa.roles) > csa.metadata.max_actors:
            results["max_actors"] = False
            is_valid = False
        else:
            results["max_actors"] = True

        # Check adapters exist
        if len(csa.policy_adapters) == len(csa.roles):
            results["adapter_count"] = True
        else:
            results["adapter_count"] = False
            is_valid = False

        return is_valid, results

    def _validate_phase_machine(self, phase_xml: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate BehaviorTree phase machine"""
        results = {}
        is_valid = True

        try:
            # Parse XML
            root = ET.fromstring(phase_xml)
            results["xml_parseable"] = True

            # Check for required BehaviorTree structure
            if root.tag != "root":
                results["root_tag"] = False
                is_valid = False
            else:
                results["root_tag"] = True

            # Check for BehaviorTree element
            bt_tree = root.find("BehaviorTree")
            if bt_tree is None:
                results["has_behavior_tree"] = False
                is_valid = False
            else:
                results["has_behavior_tree"] = True

                # Count nodes
                all_nodes = list(bt_tree.iter())
                results["node_count"] = len(all_nodes)

                # Check for control nodes (Sequence, Fallback, etc.)
                control_nodes = [
                    n
                    for n in all_nodes
                    if n.tag in ["Sequence", "Fallback", "Parallel", "ReactiveSequence"]
                ]
                results["control_node_count"] = len(control_nodes)

                # Check for action nodes
                action_nodes = [n for n in all_nodes if n.tag == "Action"]
                results["action_node_count"] = len(action_nodes)

        except ET.ParseError as e:
            results["xml_parseable"] = False
            results["parse_error"] = str(e)
            is_valid = False

        return is_valid, results

    def _validate_safety_envelope(self, csa: CooperativeSkillArtefact) -> Tuple[bool, Dict[str, Any]]:
        """Validate safety envelope completeness"""
        results = {}
        is_valid = True

        envelope = csa.safety_envelope

        # Check velocity limits exist and are positive
        if envelope.max_velocity and all(v > 0 for v in envelope.max_velocity.values()):
            results["velocity_limits"] = True
        else:
            results["velocity_limits"] = False
            is_valid = False

        # Check acceleration limits
        if envelope.max_acceleration and all(a > 0 for a in envelope.max_acceleration.values()):
            results["acceleration_limits"] = True
        else:
            results["acceleration_limits"] = False
            is_valid = False

        # Check separation distance
        if envelope.min_separation_distance > 0:
            results["separation_distance"] = True
        else:
            results["separation_distance"] = False
            is_valid = False

        # Check workspace bounds are valid
        min_bound, max_bound = envelope.workspace_bounds
        if len(min_bound) == len(max_bound) == 3:
            if all(min_b < max_b for min_b, max_b in zip(min_bound, max_bound)):
                results["workspace_bounds"] = True
            else:
                results["workspace_bounds"] = False
                is_valid = False
        else:
            results["workspace_bounds"] = False
            is_valid = False

        # Check emergency stop triggers defined
        if envelope.emergency_stop_triggers:
            results["emergency_stops"] = True
        else:
            results["emergency_stops"] = False
            is_valid = False

        return is_valid, results

    def _validate_role_consistency(self, csa: CooperativeSkillArtefact) -> Tuple[bool, Dict[str, Any]]:
        """Validate role-adapter consistency"""
        results = {}
        is_valid = True

        role_ids = {r.role_id for r in csa.roles}
        adapter_ids = {a.role_id for a in csa.policy_adapters}

        # Check all roles have adapters
        if role_ids == adapter_ids:
            results["role_adapter_match"] = True
        else:
            results["role_adapter_match"] = False
            results["missing_adapters"] = list(role_ids - adapter_ids)
            results["extra_adapters"] = list(adapter_ids - role_ids)
            is_valid = False

        # Check adapter dimensions consistency
        for role in csa.roles:
            adapter = csa.get_role_adapter(role.role_id)
            if adapter:
                # TODO: Validate adapter weight dimensions match role config
                results[f"{role.role_id}_dims"] = True
            else:
                results[f"{role.role_id}_dims"] = False
                is_valid = False

        return is_valid, results

    def _validate_coordination_encoder(self, csa: CooperativeSkillArtefact) -> Tuple[bool, Dict[str, Any]]:
        """Validate coordination encoder"""
        results = {}
        is_valid = True

        encoder = csa.coordination_encoder

        # Check latent dimension is positive
        if encoder.latent_dim > 0:
            results["latent_dim"] = True
        else:
            results["latent_dim"] = False
            is_valid = False

        # Check sequence length is positive
        if encoder.sequence_length > 0:
            results["sequence_length"] = True
        else:
            results["sequence_length"] = False
            is_valid = False

        # Check encoder type is recognized
        valid_types = ["transformer", "rnn", "lstm", "gru", "mlp"]
        if encoder.encoder_type in valid_types:
            results["encoder_type"] = True
        else:
            results["encoder_type"] = False
            results["invalid_type"] = encoder.encoder_type
            is_valid = False

        return is_valid, results

    def _validate_metadata(self, csa: CooperativeSkillArtefact) -> Tuple[bool, Dict[str, Any]]:
        """Validate metadata completeness"""
        results = {}
        is_valid = True

        meta = csa.metadata

        # Check version format
        try:
            parts = meta.version.split(".")
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                results["version_format"] = True
            else:
                results["version_format"] = False
                is_valid = False
        except Exception:
            results["version_format"] = False
            is_valid = False

        # Check compatibility lists are non-empty
        if meta.compatible_robots:
            results["has_compatible_robots"] = True
        else:
            results["has_compatible_robots"] = False
            is_valid = False

        if meta.compatible_end_effectors:
            results["has_compatible_end_effectors"] = True
        else:
            results["has_compatible_end_effectors"] = False
            is_valid = False

        # Check test metrics are in valid range
        if 0.0 <= meta.test_pass_rate <= 1.0:
            results["test_pass_rate_valid"] = True
        else:
            results["test_pass_rate_valid"] = False
            is_valid = False

        # Check privacy mode is recognized
        valid_modes = ["ldp", "dp_sgd", "he", "fhe", "none"]
        if meta.privacy_mode in valid_modes:
            results["privacy_mode_valid"] = True
        else:
            results["privacy_mode_valid"] = False
            is_valid = False

        return is_valid, results
