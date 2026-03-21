from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

from torch_inference.ls_eend_streaming_infer import build_parser, main as shared_main


def main() -> None:
    parser = build_parser()
    parser.description = "Streaming LS-EEND CoreML inference with optional RTTM scoring."
    args = parser.parse_args()
    if args.coreml_model is None:
        parser.error("--coreml-model is required.")
    shared_main()


if __name__ == "__main__":
    main()
