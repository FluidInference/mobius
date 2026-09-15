"""Check real bundled metadata; never create dummy models or synthetic audio."""

import json
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = runpy.run_path(str(ROOT / "verify-demo.py"))


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

    def test_bundled_evidence(self):
        result = TOOLS["verify"](ROOT)
        self.assertEqual(result["metadata_clips_verified"], 12)
        self.assertEqual(result["audio_clips_verified"], 0)

    def test_duplicate_id_rejected(self):
        self.manifest["clips"][1]["id"] = self.manifest["clips"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            TOOLS["validate_manifest"](self.manifest)

    def test_unbalanced_slices_rejected(self):
        self.manifest["clips"][0]["language"] = "mixed"
        with self.assertRaisesRegex(ValueError, "four clips"):
            TOOLS["validate_manifest"](self.manifest)

    def test_path_traversal_rejected(self):
        self.manifest["clips"][0]["id"] = "../../escape"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            TOOLS["validate_manifest"](self.manifest)

    def test_evidence_path_escape_rejected(self):
        self.manifest["clips"][0]["evidence"]["asr"] = "../../elsewhere.json"
        with self.assertRaisesRegex(ValueError, "evidence path"):
            TOOLS["validate_manifest"](self.manifest)

    def test_duration_mismatch_rejected(self):
        self.manifest["total_duration_seconds"] += 1
        with self.assertRaisesRegex(ValueError, "Total duration"):
            TOOLS["validate_manifest"](self.manifest)

    def test_voice_change_rejected(self):
        self.manifest["clips"][0]["voice"] = "af_heart"
        with self.assertRaisesRegex(ValueError, "Voice/speed"):
            TOOLS["validate_manifest"](self.manifest)

    def test_nonfinite_duration_rejected(self):
        self.manifest["clips"][0]["duration_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid duration"):
            TOOLS["validate_manifest"](self.manifest)

    def test_existing_asr_disagreement_preserved(self):
        clip = next(c for c in self.manifest["clips"] if c["id"] == "parallel_mixed_001")
        self.assertEqual(clip["saved_asr"]["rate"], 0)
        self.assertEqual(clip["saved_asr"]["auto_language_diagnostic"]["rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
