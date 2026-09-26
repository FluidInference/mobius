"""Merge build/stages/functions/*.mlpackage into one weight-shared multifunction KevStages.mlpackage."""

import argparse
import shutil
from pathlib import Path

import coremltools as ct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions", default="build/stages/functions")
    parser.add_argument("--output", default="build/stages/KevStages.mlpackage")
    args = parser.parse_args()
    functions = Path(args.functions)
    packages = sorted(functions.glob("*.mlpackage"))
    descriptor = ct.utils.MultiFunctionDescriptor()
    for package in packages:
        descriptor.add_function(str(package), src_function_name="main", target_function_name=package.stem)
    descriptor.default_function_name = packages[0].stem
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    ct.utils.save_multifunction(descriptor, str(output))
    shutil.copy(functions / "config.json", output.parent / "config.json")
    print(f"{len(packages)} functions -> {output}")


if __name__ == "__main__":
    main()
