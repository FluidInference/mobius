from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from provenance import (
    MATERIAL_INPUTS,
    build_manifest,
    require_disjoint_paths,
    require_commit_sha,
    verified_git_revision,
)


class ProvenanceTests(unittest.TestCase):
    def test_requires_full_commit_sha(self) -> None:
        revision = "a" * 40
        self.assertEqual(require_commit_sha(revision, "revision"), revision)
        for invalid in ("main", "a" * 39, "g" * 40):
            with self.assertRaises(ValueError):
                require_commit_sha(invalid, "revision")

    def test_manifest_hashes_material_inputs_and_release_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_root = root / "model"
            release_root = root / "release"
            for index, relative_path in enumerate(MATERIAL_INPUTS):
                path = model_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"input-{index}".encode())
            artifact = release_root / "Segmentation.mlmodelc" / "model.mil"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"artifact")

            manifest = build_manifest(
                model_root=model_root,
                release_root=release_root,
                upstream_repository="pyannote/speaker-diarization-community-1",
                upstream_revision="a" * 40,
                conversion_revision="b" * 40,
                commands=(("python", "convert-coreml.py"),),
            )

            source_files = manifest["source"]["files"]
            self.assertEqual([entry["path"] for entry in source_files], sorted(MATERIAL_INPUTS))
            self.assertEqual(
                manifest["artifacts"],
                [
                    {
                        "path": "Segmentation.mlmodelc/model.mil",
                        "size": 8,
                        "sha256": hashlib.sha256(b"artifact").hexdigest(),
                    }
                ],
            )

    def test_verified_git_revision_rejects_dirty_or_mismatched_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.email", "provenance-test@example.com"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Provenance Test"],
                cwd=repository,
                check=True,
            )
            tracked = repository / "converter.py"
            tracked.write_text("print('clean')\n", encoding="utf-8")
            subprocess.run(["git", "add", "converter.py"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "--quiet", "-m", "fixture"], cwd=repository, check=True)

            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(verified_git_revision(repository), head)
            self.assertEqual(verified_git_revision(repository, head), head)
            with self.assertRaises(ValueError):
                verified_git_revision(repository, "a" * 40)

            tracked.write_text("print('dirty')\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verified_git_revision(repository)

    def test_release_directories_must_not_overlap(self) -> None:
        root = Path("/tmp/release-layout")
        require_disjoint_paths(root / "work", root / "release")
        for work_dir, release_dir in (
            (root, root),
            (root / "work", root / "work" / "release"),
            (root / "release" / "work", root / "release"),
        ):
            with self.assertRaises(ValueError):
                require_disjoint_paths(work_dir, release_dir)


if __name__ == "__main__":
    unittest.main()
