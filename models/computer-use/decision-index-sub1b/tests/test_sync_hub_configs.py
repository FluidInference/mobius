"""Guard against publishing configs from a different model revision."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "sync-hub-configs.py"
spec = importlib.util.spec_from_file_location("sync_hub_configs", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ConfigReleaseTests(unittest.TestCase):
    def test_each_release_matches_its_conversion_lock(self) -> None:
        self.assertEqual(
            len(module.SOURCES), len({item.hub_repo for item in module.SOURCES})
        )
        for source in module.SOURCES:
            with self.subTest(model=source.name):
                module.verify_lock(source)

    def test_rejects_source_revision_drift(self) -> None:
        for source in module.SOURCES:
            with self.subTest(model=source.name), self.assertRaises(ValueError):
                module.verify_lock(replace(source, source_revision="0" * 40))

    def test_provenance_does_not_claim_automodel_compatibility(self) -> None:
        for source in module.SOURCES:
            with self.subTest(model=source.name):
                note = module.provenance(source).decode()
                self.assertIn(source.source_revision, note)
                self.assertIn(source.sha256, note)
                self.assertIn("does not make", note)


if __name__ == "__main__":
    unittest.main()
