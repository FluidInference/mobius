"""Load and discover CoreML models (.mlmodelc / .mlpackage)."""

from __future__ import annotations

import tempfile
from pathlib import Path


def _is_mlmodelc(path: Path) -> bool:
    return path.is_dir() and path.suffix == ".mlmodelc"


def _is_mlpackage(path: Path) -> bool:
    return path.is_dir() and path.suffix == ".mlpackage"


def compile_mlpackage(mlpackage_path: Path) -> Path:
    """Compile .mlpackage to .mlmodelc using coremltools."""
    import coremltools as ct

    model = ct.models.MLModel(str(mlpackage_path))
    tmp_dir = tempfile.mkdtemp(prefix="coreml_cli_")
    out_path = Path(tmp_dir) / (mlpackage_path.stem + ".mlmodelc")
    model.save(str(out_path))
    if out_path.exists() and _is_mlmodelc(out_path):
        return out_path
    compiled = ct.utils.compile_model(str(mlpackage_path))
    return Path(compiled)


def discover_models(path: Path) -> list[Path]:
    """Given a path, return list of .mlmodelc paths."""
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    if _is_mlmodelc(path):
        return [path]

    if _is_mlpackage(path):
        return [compile_mlpackage(path)]

    if not path.is_dir():
        raise ValueError(f"{path} is not a CoreML model or directory")

    mlmodelc_names: set[str] = set()
    mlmodelc_paths: list[Path] = []
    mlpackage_paths: list[Path] = []

    for entry in sorted(path.iterdir()):
        if _is_mlmodelc(entry):
            mlmodelc_names.add(entry.stem)
            mlmodelc_paths.append(entry)
        elif _is_mlpackage(entry):
            mlpackage_paths.append(entry)

    for pkg in mlpackage_paths:
        if pkg.stem not in mlmodelc_names:
            mlmodelc_paths.append(compile_mlpackage(pkg))

    if not mlmodelc_paths:
        raise FileNotFoundError(f"No CoreML models found in {path}")

    return sorted(mlmodelc_paths, key=lambda p: p.stem)
