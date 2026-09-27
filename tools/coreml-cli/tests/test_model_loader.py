from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from coreml_cli.model_loader import compile_mlpackage, discover_models


class ModelLoaderTests(unittest.TestCase):
    def test_compile_mlpackage_uses_coremltools_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "classifier.mlpackage"
            package.mkdir()
            compiled = root / "classifier.mlmodelc"
            compiled.mkdir()
            calls: list[str] = []

            def compile_model(path: str) -> str:
                calls.append(path)
                return str(compiled)

            coremltools = types.SimpleNamespace(utils=types.SimpleNamespace(compile_model=compile_model))
            with patch.dict(sys.modules, {"coremltools": coremltools}):
                result = compile_mlpackage(package)

            self.assertEqual(result, compiled)
            self.assertEqual(calls, [str(package)])

    def test_discover_models_compiles_a_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "classifier.mlpackage"
            package.mkdir()
            compiled = package.with_suffix(".mlmodelc")

            with patch("coreml_cli.model_loader.compile_mlpackage", return_value=compiled) as compile_package:
                result = discover_models(package)

            self.assertEqual(result, [compiled])
            compile_package.assert_called_once_with(package.resolve())


if __name__ == "__main__":
    unittest.main()
