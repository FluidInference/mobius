from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPORTER = ROOT / "conversion" / "ls_eend_coreml_export.py"
ARTIFACTS = ROOT / "artifacts" / "coreml"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    checkpoint: Path
    config: Path
    output_stem: str


VARIANTS = (
    VariantSpec(
        name="ami",
        checkpoint=ROOT / "model_checkpoints" / "ls_eend_ami_allspk_model.ckpt",
        config=ROOT / "conf" / "spk_onl_conformer_retention_enc_dec_nonautoreg_ami_infer.yaml",
        output_stem="ls_eend_ami_step",
    ),
    VariantSpec(
        name="callhome",
        checkpoint=ROOT / "model_checkpoints" / "ls_eend_ch_allspk_model.ckpt",
        config=ROOT / "conf" / "spk_onl_conformer_retention_enc_dec_nonautoreg_callhome_infer.yaml",
        output_stem="ls_eend_callhome_step",
    ),
    VariantSpec(
        name="dihard2",
        checkpoint=ROOT / "model_checkpoints" / "ls_eend_dih2_allspk_model.ckpt",
        config=ROOT / "conf" / "spk_onl_conformer_retention_enc_dec_nonautoreg_dihard2_infer.yaml",
        output_stem="ls_eend_dih2_step",
    ),
    VariantSpec(
        name="dihard3",
        checkpoint=ROOT / "model_checkpoints" / "ls_eend_dih3_allspk_model.ckpt",
        config=ROOT / "conf" / "spk_onl_conformer_retention_enc_dec_nonautoreg_dihard3_infer.yaml",
        output_stem="ls_eend_dih3_step",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch export float32 CoreML packages for LS-EEND model variants.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[spec.name for spec in VARIANTS],
        help="Subset of variants to export. Defaults to all.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device used during tracing.")
    parser.add_argument(
        "--deployment-target",
        choices=("macos13", "macos14", "macos15"),
        default="macos15",
    )
    parser.add_argument(
        "--compute-precision",
        choices=("float16", "float32", "mixed_float16"),
        default="float32",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-export even if the package already exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected = {name.lower() for name in args.variants}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    for spec in VARIANTS:
        if spec.name not in selected:
            continue
        output = ARTIFACTS / f"{spec.output_stem}.mlpackage"
        metadata = ARTIFACTS / f"{spec.output_stem}.json"
        if not args.overwrite and output.exists() and metadata.exists():
            print(f"Skipping {spec.name}: {output.name} already exists.")
            continue
        cmd = [
            sys.executable,
            str(EXPORTER),
            "--checkpoint",
            str(spec.checkpoint),
            "--config",
            str(spec.config),
            "--output",
            str(output),
            "--metadata-json",
            str(metadata),
            "--device",
            args.device,
            "--deployment-target",
            args.deployment_target,
            "--compute-precision",
            args.compute_precision,
        ]
        print(f"Exporting {spec.name} -> {output.name}")
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
