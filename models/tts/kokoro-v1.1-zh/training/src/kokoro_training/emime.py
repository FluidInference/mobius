"""Read-only archive inventory; no extraction, speaker selection, or training."""

from collections import defaultdict
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import tarfile

import soundfile as sf

from .artifacts import sha256

WAV_NAME = re.compile(r"^(M[FM][1-7])_(ENG|MAN)_(\d+)_([01])\.wav$", re.IGNORECASE)


def inventory_emime(archive: Path) -> dict:
    counts = defaultdict(lambda: {"recordings": 0, "seconds": 0.0, "sample_rates": set()})
    documents, errors = [], []
    wav_count, extra_wav_count, matched_bytes = 0, 0, 0
    members_seen = set()
    with tarfile.open(archive, mode="r|bz2") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError("Unexpected unsafe archive member")
            if not member.isfile():
                continue
            if member.name in members_seen:
                raise ValueError("Duplicate archive member")
            members_seen.add(member.name)
            if member.size > 100_000_000:
                raise ValueError("Unexpected member exceeds 100 MB inspection cap")
            if name.suffix.lower() == ".wav":
                wav_count += 1
                match = WAV_NAME.match(name.name)
                # Segmented test copies are separate views, not new training recordings.
                if match is None or "Mandarin_test_set_segmentations" in name.parts:
                    extra_wav_count += 1
                    continue
                speaker, lang, _, microphone = match.groups()
                stream = tar.extractfile(member)
                audio_bytes = stream.read()
                matched_bytes += len(audio_bytes)
                try:
                    info = sf.info(io.BytesIO(audio_bytes))
                except RuntimeError:
                    errors.append({"file": str(name), "reason": "unreadable audio header"})
                    continue
                group = counts[(speaker.upper(), lang.upper(), microphone)]
                group["recordings"] += 1
                group["seconds"] += info.duration
                group["sample_rates"].add(info.samplerate)
                if info.channels != 1:
                    errors.append({"file": str(name), "reason": "unexpected non-mono audio"})
            elif name.suffix.lower() in {".txt", ".pdf"} or "license" in name.name.lower():
                content = tar.extractfile(member).read()
                documents.append(
                    {
                        "file": str(name),
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
    if not counts:
        errors.append({"reason": "no matching EMIME recordings found"})
    return {
        "schema_version": 1,
        "passed": not errors,
        "scope": "archive metadata/header inventory; not QC, data approval, or training readiness",
        "training_ready": False,
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "wav_files": wav_count,
        "extra_or_segmented_wav_files": extra_wav_count,
        "matched_audio_bytes": matched_bytes,
        "groups": [
            {
                "speaker_id": speaker,
                "language": lang,
                "microphone": mic,
                **{k: (sorted(v) if isinstance(v, set) else v) for k, v in stats.items()},
            }
            for (speaker, lang, mic), stats in sorted(counts.items())
        ],
        "documents": documents,
        "errors": errors,
        "limitations": [
            "microphone views must not double usable hours or cross splits",
            "test segmentations overlap source passages",
            "no within-utterance code-switching demonstrated by directory labels",
            "target speaker, accent/listening QC, transcript alignment, rights/consent and splits unresolved",
        ],
    }
