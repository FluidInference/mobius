"""Fail-closed container checks, independent of NeMo/model downloads."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from fermion_container import FORMAT
from fivevalue_weights import load_fivevalue


class ContainerTests(unittest.TestCase):
    def file(self, entries, payload=b"", fmt=FORMAT):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model.fermion"
        header = json.dumps({"format": fmt, "index": entries}).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
        return path

    def record(self):
        # [-lo, 0, +hi, -hi, +lo], trits 0,1,2,0,2 and high-bit order 0,1,1,0.
        trits = bytes([0 + 3 + 2*9 + 0*27 + 2*81])
        blob = trits + bytes([6]) + np.array([0.25, 0.75], dtype=np.float16).tobytes()
        return {"n": "encoder.test", "k": "five_value", "shape": [1,5], "b": len(blob)}, blob

    def test_exact_five_value_expansion(self):
        record, blob = self.record()
        values = load_fivevalue(self.file([record], blob), expected_count=1)
        np.testing.assert_array_equal(values["encoder.test.weight"].dense_fp16(), [[-0.25,0,0.75,-0.75,0.25]])

    def test_truncated_record_is_rejected(self):
        record, blob = self.record()
        with self.assertRaisesRegex(ValueError, "Truncated"):
            load_fivevalue(self.file([record], blob[:-1]), expected_count=1)

    def test_duplicate_palette_is_rejected(self):
        record, blob = self.record()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_fivevalue(self.file([record, record], blob + blob), expected_count=2)

    def test_trailing_bytes_are_rejected(self):
        record, blob = self.record()
        with self.assertRaisesRegex(ValueError, "Trailing"):
            load_fivevalue(self.file([record], blob+b"x"), expected_count=1)

    def test_wrong_checkpoint_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_fivevalue(self.file([], fmt="different-model"))

    def test_missing_palette_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 264"):
            load_fivevalue(self.file([]))


if __name__ == "__main__":
    unittest.main()
