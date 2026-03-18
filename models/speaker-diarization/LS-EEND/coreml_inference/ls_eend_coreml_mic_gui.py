from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

from torch_inference.ls_eend_mic_gui import LSEENDMicGUI, build_parser, list_devices


def main() -> None:
    parser = build_parser()
    parser.description = "Live microphone GUI for fixed-shape LS-EEND CoreML inference."
    parser.set_defaults(block_seconds=0.1, refresh_seconds=0.1)
    args = parser.parse_args()
    if args.list_devices:
        list_devices()
        return
    if args.coreml_model is None:
        parser.error("--coreml-model is required.")
    app = LSEENDMicGUI(args)
    app.run()


if __name__ == "__main__":
    main()
