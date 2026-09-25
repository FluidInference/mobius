import json
import tempfile
import unittest
from pathlib import Path

from speed_campaign import (
    ResultSummary,
    generate_candidates,
    load_manifest,
    pareto_frontier,
    write_manifest,
)


class CandidateGenerationTests(unittest.TestCase):
    def test_large_chunk_ladder_is_unique_and_monotonic(self):
        candidates = generate_candidates("large-chunk")

        self.assertEqual([value.chunk_len for value in candidates], [40, 48, 64, 80, 96, 128])
        self.assertEqual(len({value.name for value in candidates}), len(candidates))
        self.assertTrue(all(value.fifo_len == 40 for value in candidates))
        self.assertTrue(all(value.spkcache_len == 264 for value in candidates))
        self.assertTrue(all(value.update_period >= value.chunk_len for value in candidates))
        self.assertEqual([value.audio_per_call for value in candidates], sorted(value.audio_per_call for value in candidates))

    def test_manifest_round_trip(self):
        candidates = generate_candidates("large-chunk")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            write_manifest(path, "large-chunk", candidates)
            payload = json.loads(path.read_text())

            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(load_manifest(path), candidates)

    def test_extended_chunk_ladder_crosses_offline_packed_length(self):
        candidates = generate_candidates("extended-chunk")

        self.assertEqual(candidates[0].chunk_len, 160)
        self.assertEqual(candidates[-1].chunk_len, 448)
        self.assertTrue(any(value.packed_frames < 684 for value in candidates))
        self.assertTrue(any(value.packed_frames > 684 for value in candidates))

    def test_context_frontier_sweeps_ane_and_offline_winners(self):
        candidates = generate_candidates("context-frontier")

        self.assertEqual(len(candidates), 16)
        self.assertEqual({value.chunk_len for value in candidates}, {128, 288})
        self.assertEqual({value.right_context for value in candidates}, {0, 1, 2, 4, 8, 16, 32, 40})


class ParetoTests(unittest.TestCase):
    def test_frontier_keeps_quality_and_speed_extremes(self):
        # These are the measured AMI points from the 2026-08-28 s-family gate.
        s16 = ResultSummary("s16", 16, 26.53, 24.2, 1.2, 1.1, 127, 13, 3, 1.60, 324)
        s24 = ResultSummary("s24", 16, 26.28, 24.1, 1.2, 1.0, 227, 13, 3, 2.24, 332)
        s32 = ResultSummary("s32", 16, 26.21, 24.1, 1.2, 0.9, 292, 13, 3, 2.88, 340)
        dominated = ResultSummary("dominated", 16, 27.0, 24.5, 1.3, 1.2, 100, 12, 4, 3.00, 360)

        frontier = {value.name for value in pareto_frontier([s16, s24, s32, dominated])}

        self.assertEqual(frontier, {"s16", "s24", "s32"})


if __name__ == "__main__":
    unittest.main()
