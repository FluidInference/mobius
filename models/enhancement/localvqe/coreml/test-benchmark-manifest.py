"""Manifest regressions; uses metadata/path checks, no generated audio/models."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from localvqe_coreml.benchmark_manifest import collect_jobs, require_coverage


class ManifestTests(unittest.TestCase):
    def test_coverage_requires_exact_unique_nonempty_set(self):
        require_coverage(["b", "a"], ["a", "b"])
        for actual, expected in [([], []), (["a"], ["a", "b"]), (["a", "b"], ["a"]),
                                 (["a", "a"], ["a"]), (["a"], ["a", "a"])]:
            with self.assertRaises(ValueError):
                require_coverage(actual, expected)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manifest = Path(self.directory.name) / "manifest.txt"
        self.stems = ["a_doubletalk", "b_doubletalk", "c_nearend-singletalk"]
        self.manifest.write_text("\n".join(self.stems))
        self.microphones = [Path("blind") / stem.split("_", 1)[1] / (stem + "_mic.flac") for stem in self.stems]
        self.scenarios = {"doubletalk": "dt", "nearend-singletalk": "nst"}

    def collect(self, limit=None):
        return collect_jobs(Path("blind"), "renders", self.manifest, self.scenarios, limit)

    def test_missing_loopback_is_fatal(self):
        with patch.object(Path, "glob", return_value=self.microphones), patch.object(
                Path, "is_file", lambda path: not path.name.endswith("_lpb.flac")):
            with self.assertRaisesRegex(ValueError, "loopback"):
                self.collect()

    def test_missing_render_is_fatal(self):
        with patch.object(Path, "glob", return_value=self.microphones), patch.object(
                Path, "is_file", lambda path: not path.name.endswith("_enh.wav")):
            with self.assertRaisesRegex(ValueError, "render"):
                self.collect()

    def test_limit_selects_first_stems_per_scenario(self):
        with patch.object(Path, "glob", return_value=self.microphones), patch.object(Path, "is_file", return_value=True):
            jobs, expected = self.collect(limit=1)
            self.assertEqual([job[1] for job in jobs], ["a_doubletalk", "c_nearend-singletalk"])
            self.assertEqual(expected, self.stems)
            with self.assertRaises(ValueError):
                self.collect(limit=0)

    def test_limit_does_not_hide_missing_source_pair(self):
        with patch.object(Path, "glob", return_value=self.microphones), patch.object(
                Path, "is_file", lambda path: path.name != "b_doubletalk_lpb.flac"):
            with self.assertRaisesRegex(ValueError, "loopback"):
                self.collect(limit=1)


if __name__ == "__main__":
    unittest.main()
