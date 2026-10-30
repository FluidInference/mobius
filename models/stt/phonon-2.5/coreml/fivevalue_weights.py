# Adapted from FluidInference/mobius PR #111 (e35c1953), Apache-2.0.
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple
import numpy as np
from fermion_container import FORMAT, _five_value
from name_mapping import hf_to_nemo_name

class FiveValue:
    """One five-value record: sign int8 [out, in], is_hi bool [out, in], lo/hi fp16 [out]."""

    def __init__(self, sign: np.ndarray, is_hi: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        self.sign, self.is_hi, self.lo, self.hi = sign, is_hi, lo.astype(np.float16), hi.astype(np.float16)

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.sign.shape)

    def codes(self) -> np.ndarray:
        """Per-row palette index into [-hi, -lo, 0, +lo, +hi]: 2 + sign * (1 + is_hi), uint8 in 0..4."""
        return (2 + self.sign.astype(np.int8) * (1 + self.is_hi.astype(np.int8))).astype(np.uint8)

    def dense_fp16(self) -> np.ndarray:
        mag = np.where(self.is_hi, self.hi[:, None], self.lo[:, None])
        return (self.sign.astype(np.float16) * mag).astype(np.float16)

def load_fivevalue(container: Path, expected_count: int = 264):
    """Read only the palette records, avoiding a second dense 627M-parameter copy."""
    records = {}
    with container.open("rb") as file:
        size = int.from_bytes(file.read(8), "little")
        if not 0 < size < 10_000_000:
            raise ValueError("Invalid header size")
        header = json.loads(file.read(size))
        if header["format"] != FORMAT:
            raise ValueError("Unsupported checkpoint format")
        for entry in header["index"]:
            blob = file.read(entry["b"])
            if len(blob) != entry["b"]:
                raise ValueError("Truncated checkpoint")
            if entry["k"] != "five_value":
                continue
            name = hf_to_nemo_name(entry["n"] + ".weight")
            if name in records:
                raise ValueError(f"Duplicate palette: {name}")
            raw = {}
            _five_value(blob, tuple(entry["shape"]), raw)
            records[name] = FiveValue(raw["sign"], raw["is_hi"], raw["lo"], raw["hi"])
        if file.read(1):
            raise ValueError("Trailing checkpoint data")
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} palette records, got {len(records)}")
    return records
