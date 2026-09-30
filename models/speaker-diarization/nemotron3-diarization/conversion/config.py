"""Streaming latency profiles for Nemotron 3 Diarization (from the model card).

All values are in 80 ms encoder frames; chunk mel frames = (lc + chunk + rc) * 8.
Latency = (chunk_len + right_context) * 80 ms.
"""

import json
from pathlib import Path
from typing import Any

NEMO_CHECKPOINT = "/Users/hanweng/Documents/nemotron3-diar-preview/Nemotron-3-Diarization.nemo"

SUBSAMPLING = 8
FEAT_DIM = 128
EMB_DIM = 512
NUM_SPEAKERS = 8
UPSAMPLE_FACTOR = 8

VARIANT_FIELDS = (
    "spkcache_len",
    "fifo_len",
    "chunk_len",
    "right_context",
    "left_context",
    "update_period",
)

VARIANTS = {
    "offline": dict(spkcache_len=264, fifo_len=40, chunk_len=340, right_context=40, left_context=0, update_period=300),
    "low": dict(spkcache_len=264, fifo_len=264, chunk_len=9, right_context=4, left_context=0, update_period=222),
    "verylow": dict(spkcache_len=264, fifo_len=264, chunk_len=6, right_context=2, left_context=0, update_period=222),
    "ultra": dict(spkcache_len=264, fifo_len=264, chunk_len=3, right_context=1, left_context=0, update_period=222),
    # Speed-push profiles (not in the model card; validated by _check_streaming_parameters):
    # fast: same 1.04s latency as `low` but a 40-frame FIFO -> packed 317 vs 541 frames.
    "fast": dict(spkcache_len=264, fifo_len=40, chunk_len=9, right_context=4, left_context=0, update_period=40),
    # efficient: 4.16s latency, 48-frame chunk amortizes the static state cost per call.
    "efficient": dict(spkcache_len=264, fifo_len=264, chunk_len=48, right_context=4, left_context=0, update_period=222),
}

# Speed-campaign sweeps (all on the `fast` base: spkcache 264, fifo 40 unless swept).
# Sweep 1 — trade right context for chunk at fixed T=317 and 1.04 s latency.
for _c, _r in [(10, 3), (11, 2), (12, 1), (13, 0)]:
    VARIANTS[f"c{_c}r{_r}"] = dict(
        spkcache_len=264, fifo_len=40, chunk_len=_c, right_context=_r, left_context=0, update_period=40)
# Sweep 2 — shrink the speaker cache (83% of fast's packed sequence).
for _sc in [224, 192, 160, 128]:
    VARIANTS[f"sc{_sc}"] = dict(
        spkcache_len=_sc, fifo_len=40, chunk_len=9, right_context=4, left_context=0, update_period=40)
# Sweep 3 — shrink the FIFO further (update period clamped to fifo+chunk).
for _f in [32, 24, 16, 0]:
    VARIANTS[f"f{_f}"] = dict(
        spkcache_len=264, fifo_len=_f, chunk_len=9, right_context=4, left_context=0,
        update_period=min(40, _f + 9))


def chunk_mel_frames(v: dict) -> int:
    return (v["left_context"] + v["chunk_len"] + v["right_context"]) * SUBSAMPLING


def chunk_enc_frames(v: dict) -> int:
    return v["left_context"] + v["chunk_len"] + v["right_context"]


def packed_frames(v: dict) -> int:
    return v["spkcache_len"] + v["fifo_len"] + chunk_enc_frames(v)


def validate_variant(name: str, value: dict[str, Any]) -> dict[str, int]:
    """Validate and normalize one manifest variant before loading NeMo."""
    missing = [field for field in VARIANT_FIELDS if field not in value]
    unknown = sorted(set(value) - set(VARIANT_FIELDS))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ValueError(f"Invalid variant {name!r}: {'; '.join(details)}")

    normalized: dict[str, int] = {}
    for field in VARIANT_FIELDS:
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"Invalid variant {name!r}: {field} must be an integer")
        normalized[field] = raw

    nonnegative = ("spkcache_len", "fifo_len", "right_context", "left_context")
    for field in nonnegative:
        if normalized[field] < 0:
            raise ValueError(f"Invalid variant {name!r}: {field} must be nonnegative")
    for field in ("chunk_len", "update_period"):
        if normalized[field] <= 0:
            raise ValueError(f"Invalid variant {name!r}: {field} must be positive")
    if normalized["spkcache_len"] < NUM_SPEAKERS:
        raise ValueError(
            f"Invalid variant {name!r}: spkcache_len must hold at least {NUM_SPEAKERS} speakers")
    if normalized["spkcache_len"] % NUM_SPEAKERS:
        raise ValueError(
            f"Invalid variant {name!r}: spkcache_len must be divisible by {NUM_SPEAKERS}")
    return normalized


def load_variant_manifest(path: str | Path) -> dict[str, dict[str, int]]:
    """Load variants from a campaign JSON manifest.

    The root may be either ``{"variants": [...]}`` or a direct list. Each list item must
    include a unique ``name`` plus every field in :data:`VARIANT_FIELDS`.
    """
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    items = payload.get("variants") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"Invalid manifest {manifest_path}: expected a variants list")

    variants: dict[str, dict[str, int]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid manifest {manifest_path}: every variant must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid manifest {manifest_path}: invalid variant name {name!r}")
        if name in variants:
            raise ValueError(f"Invalid manifest {manifest_path}: duplicate variant {name!r}")
        variants[name] = validate_variant(name, {key: value for key, value in item.items() if key != "name"})
    return variants

# Combined speed-campaign candidates from screening (2026-08-28):
# turbo1: moderate — chunk 12/rc 1, cache 192, fifo 24 (T=229, advance 0.96s/call)
VARIANTS["turbo1"] = dict(
    spkcache_len=192, fifo_len=24, chunk_len=12, right_context=1, left_context=0, update_period=24)
# turbo2: aggressive — chunk 13/rc 0, cache 160, fifo 16 (T=189, advance 1.04s/call)
VARIANTS["turbo2"] = dict(
    spkcache_len=160, fifo_len=16, chunk_len=13, right_context=0, left_context=0, update_period=16)

# Chunk-size ladder (2026-08-28, "0.72s/call too low"): quality template (fifo 264) and
# speed template (fifo 40) at chunk 16/24/32 = 1.28/1.92/2.56s audio per call.
for _c in [16, 24, 32]:
    VARIANTS[f"q{_c}"] = dict(
        spkcache_len=264, fifo_len=264, chunk_len=_c, right_context=4, left_context=0, update_period=222)
    VARIANTS[f"s{_c}"] = dict(
        spkcache_len=264, fifo_len=40, chunk_len=_c, right_context=4, left_context=0,
        update_period=min(40, 40 + _c))

# Extended chunk tier (resume, 2026-08-28): high-throughput ladder toward batch.
# fifo 40, rc 4; update_period is clamped by NeMo to chunk_len when smaller.
for _c in [64, 96, 128, 160, 192]:
    VARIANTS[f"c{_c}"] = dict(
        spkcache_len=264, fifo_len=40, chunk_len=_c, right_context=4, left_context=0, update_period=40)

# ANE compile-boundary bisection between c160 (mel 1312, compiles) and c192 (mel 1568, fails).
for _c in [168, 176, 184]:
    VARIANTS[f"c{_c}"] = dict(
        spkcache_len=264, fifo_len=40, chunk_len=_c, right_context=4, left_context=0, update_period=40)
# Hypothesis probe: c192 with rc 0 -> mel 1536. If this compiles, the cliff tracks mel length,
# not chunk_len/packed length.
VARIANTS["c192r0"] = dict(
    spkcache_len=264, fifo_len=40, chunk_len=192, right_context=0, left_context=0, update_period=40)

# Split-graph unlock probes: chunk sizes beyond the monolithic ANE cliff.
VARIANTS["c256"] = dict(
    spkcache_len=264, fifo_len=40, chunk_len=256, right_context=4, left_context=0, update_period=40)
