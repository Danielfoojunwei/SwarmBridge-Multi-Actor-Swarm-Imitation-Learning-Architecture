"""
CSA Packaging and Loading

Handles serialization, compression, and extraction of Cooperative Skill Artefacts.
"""

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import torch

from .schema import (
    CooperativeSkillArtefact,
    CoordinationEncoder,
    CSAMetadata,
    PolicyAdapter,
    RoleConfig,
    RoleType,
    SafetyEnvelope,
)


class CSAPackager:
    """Package CSA into distributable tarball"""

    def __init__(self, output_dir: Path = Path("./artifacts")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def package(
        self, csa: CooperativeSkillArtefact, output_name: Optional[str] = None
    ) -> Path:
        """
        Package CSA into .tar.gz archive

        Archive structure:
            csa_v1.0.0.tar.gz
            ├── manifest.json          # Metadata and structure
            ├── roles/
            │   ├── leader_adapter.pt
            │   ├── follower_adapter.pt
            │   └── ...
            ├── coordination_encoder.pt
            ├── phase_machine.xml
            ├── safety_envelope.json
            ├── tests/
            │   └── test_suite.json
            └── checksums.sha256       # File integrity
        """
        if output_name is None:
            output_name = f"{csa.metadata.skill_name}_v{csa.metadata.version}.tar.gz"

        output_path = self.output_dir / output_name

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 1. Save manifest
            manifest = {
                "schema_version": "1.0",
                "metadata": csa.metadata.model_dump(),
                "roles": [
                    {
                        "role_id": r.role_id,
                        "role_type": r.role_type.value,
                        "observation_dims": r.observation_dims,
                        "action_dims": r.action_dims,
                        "requires_coordination": r.requires_coordination,
                    }
                    for r in csa.roles
                ],
                "files": {
                    "coordination_encoder": "coordination_encoder.pt",
                    "phase_machine": "phase_machine.xml",
                    "safety_envelope": "safety_envelope.json",
                },
            }

            with open(tmpdir_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2, default=str)

            # 2. Save role adapters
            roles_dir = tmpdir_path / "roles"
            roles_dir.mkdir()
            for adapter in csa.policy_adapters:
                adapter.save(roles_dir / f"{adapter.role_id}_adapter.pt")

            # 3. Save coordination encoder
            csa.coordination_encoder.save(tmpdir_path / "coordination_encoder.pt")

            # 4. Save phase machine (BehaviorTree XML)
            with open(tmpdir_path / "phase_machine.xml", "w") as f:
                f.write(csa.phase_machine_xml)

            # 5. Save safety envelope
            safety_data = {
                "max_velocity": csa.safety_envelope.max_velocity,
                "max_acceleration": csa.safety_envelope.max_acceleration,
                "max_force": csa.safety_envelope.max_force,
                "max_torque": csa.safety_envelope.max_torque,
                "min_separation_distance": csa.safety_envelope.min_separation_distance,
                "workspace_bounds": csa.safety_envelope.workspace_bounds,
                "collision_primitives": csa.safety_envelope.collision_primitives,
                "emergency_stop_triggers": csa.safety_envelope.emergency_stop_triggers,
            }
            with open(tmpdir_path / "safety_envelope.json", "w") as f:
                json.dump(safety_data, f, indent=2)

            # 6. Save test suite
            tests_dir = tmpdir_path / "tests"
            tests_dir.mkdir()
            with open(tests_dir / "test_suite.json", "w") as f:
                json.dump(csa.test_suite, f, indent=2)

            # 7. Generate checksums
            checksums = {}
            for file_path in tmpdir_path.rglob("*"):
                if file_path.is_file() and file_path.name != "checksums.sha256":
                    rel_path = file_path.relative_to(tmpdir_path)
                    checksum = self._compute_sha256(file_path)
                    checksums[str(rel_path)] = checksum

            with open(tmpdir_path / "checksums.sha256", "w") as f:
                for path, checksum in sorted(checksums.items()):
                    f.write(f"{checksum}  {path}\n")

            # 8. Create tarball
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(tmpdir_path, arcname=".")

        print(f"✓ Packaged CSA to: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")
        return output_path

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class CSALoader:
    """Load and validate CSA from tarball"""

    def load(self, archive_path: Path, verify_checksums: bool = True) -> CooperativeSkillArtefact:
        """
        Load CSA from .tar.gz archive

        Args:
            archive_path: Path to CSA tarball
            verify_checksums: Whether to verify file integrity

        Returns:
            Loaded CooperativeSkillArtefact

        Raises:
            ValueError: If checksums don't match or required files missing
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 1. Extract archive
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmpdir_path)

            # 2. Verify checksums (if enabled)
            if verify_checksums:
                self._verify_checksums(tmpdir_path)

            # 3. Load manifest
            with open(tmpdir_path / "manifest.json") as f:
                manifest = json.load(f)

            # 4. Load metadata
            metadata = CSAMetadata(**manifest["metadata"])

            # 5. Load roles
            roles = []
            for role_data in manifest["roles"]:
                roles.append(
                    RoleConfig(
                        role_id=role_data["role_id"],
                        role_type=RoleType(role_data["role_type"]),
                        observation_dims=role_data["observation_dims"],
                        action_dims=role_data["action_dims"],
                        requires_coordination=role_data["requires_coordination"],
                    )
                )

            # 6. Load policy adapters
            adapters = []
            roles_dir = tmpdir_path / "roles"
            for role in roles:
                adapter_path = roles_dir / f"{role.role_id}_adapter.pt"
                adapters.append(PolicyAdapter.load(adapter_path))

            # 7. Load coordination encoder
            encoder = CoordinationEncoder.load(tmpdir_path / "coordination_encoder.pt")

            # 8. Load phase machine
            with open(tmpdir_path / "phase_machine.xml") as f:
                phase_machine_xml = f.read()

            # 9. Load safety envelope
            with open(tmpdir_path / "safety_envelope.json") as f:
                safety_data = json.load(f)
            safety_envelope = SafetyEnvelope(**safety_data)

            # 10. Load test suite
            with open(tmpdir_path / "tests" / "test_suite.json") as f:
                test_suite = json.load(f)

            # 11. Construct CSA
            csa = CooperativeSkillArtefact(
                roles=roles,
                policy_adapters=adapters,
                coordination_encoder=encoder,
                phase_machine_xml=phase_machine_xml,
                safety_envelope=safety_envelope,
                metadata=metadata,
                test_suite=test_suite,
            )

            print(f"✓ Loaded CSA: {metadata.skill_name} v{metadata.version}")
            return csa

    @staticmethod
    def _verify_checksums(extract_dir: Path) -> None:
        """Verify file checksums match manifest"""
        checksums_file = extract_dir / "checksums.sha256"
        if not checksums_file.exists():
            raise ValueError("Missing checksums.sha256 file")

        # Read expected checksums
        expected = {}
        with open(checksums_file) as f:
            for line in f:
                checksum, path = line.strip().split("  ", 1)
                expected[path] = checksum

        # Compute actual checksums
        for rel_path, expected_checksum in expected.items():
            file_path = extract_dir / rel_path
            if not file_path.exists():
                raise ValueError(f"Missing file: {rel_path}")

            actual_checksum = CSAPackager._compute_sha256(file_path)
            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Checksum mismatch for {rel_path}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )

        print("✓ All checksums verified")
