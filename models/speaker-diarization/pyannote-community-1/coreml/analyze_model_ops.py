#!/usr/bin/env python3
"""Analyze CoreML model operations to identify ANE compatibility issues."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import coremltools as ct


def analyze_model_operations(model_path: Path) -> dict[str, Any]:
    """Analyze a CoreML model and return operation statistics."""
    model = ct.models.MLModel(str(model_path))
    spec = model.get_spec()

    # Get the ML Program
    if not hasattr(spec, 'mlProgram'):
        print(f"Model at {model_path} is not an ML Program (might be neural network format)")
        return {}

    ml_program = spec.mlProgram

    # Analyze functions
    op_types = []
    op_details = []
    tensor_shapes = defaultdict(list)
    precision_info = defaultdict(int)

    for function in ml_program.functions.values():
        for block in function.block_specializations.values():
            for op in block.operations:
                op_type = op.type
                op_types.append(op_type)

                # Collect operation details
                op_info = {
                    'type': op_type,
                    'inputs': [str(inp) if hasattr(inp, '__str__') else inp for inp in op.inputs] if hasattr(op, 'inputs') else [],
                    'outputs': [str(out) if hasattr(out, '__str__') else out for out in op.outputs] if hasattr(op, 'outputs') else [],
                }

                # Check for attributes that might affect ANE compatibility
                if hasattr(op, 'attributes'):
                    attrs = {}
                    for attr_name in dir(op.attributes):
                        if not attr_name.startswith('_'):
                            attr_val = getattr(op.attributes, attr_name, None)
                            if attr_val is not None:
                                attrs[attr_name] = str(attr_val)
                    if attrs:
                        op_info['attributes'] = attrs

                op_details.append(op_info)

                # Track tensor shapes for inputs/outputs (skip if not available)
                try:
                    for inp in op.inputs:
                        if hasattr(inp, 'type') and hasattr(inp.type, 'tensorType'):
                            tensor_type = inp.type.tensorType
                            if hasattr(tensor_type, 'shape'):
                                shape = tuple(tensor_type.shape)
                                tensor_shapes[op_type].append(('input', shape))
                            if hasattr(tensor_type, 'dataType'):
                                precision_info[f"{op_type}_input_{tensor_type.dataType}"] += 1
                except (AttributeError, TypeError):
                    pass

                try:
                    for out in op.outputs:
                        if hasattr(out, 'type') and hasattr(out.type, 'tensorType'):
                            tensor_type = out.type.tensorType
                            if hasattr(tensor_type, 'shape'):
                                shape = tuple(tensor_type.shape)
                                tensor_shapes[op_type].append(('output', shape))
                            if hasattr(tensor_type, 'dataType'):
                                precision_info[f"{op_type}_output_{tensor_type.dataType}"] += 1
                except (AttributeError, TypeError):
                    pass

    # Count operation types
    op_counts = Counter(op_types)

    return {
        'total_ops': len(op_types),
        'op_counts': dict(op_counts.most_common()),
        'op_details': op_details,
        'tensor_shapes': dict(tensor_shapes),
        'precision_info': dict(precision_info),
        'unique_op_types': len(op_counts),
    }


def identify_ane_issues(analysis: dict[str, Any]) -> dict[str, list[str]]:
    """Identify potential ANE compatibility issues based on operation analysis."""
    issues = defaultdict(list)

    # Operations that typically don't run well on ANE
    problematic_ops = {
        'reshape', 'transpose', 'squeeze', 'unsqueeze', 'flatten',
        'gather', 'scatter', 'slice_by_index',
        'linear', 'matmul',  # Should be conv2d instead
        'batch_norm',  # Should be layer_norm_ane
        'reduce_mean', 'reduce_sum', 'reduce_prod',  # Often fall back to CPU
    }

    op_counts = analysis.get('op_counts', {})

    for op_type, count in op_counts.items():
        if op_type in problematic_ops:
            issues['problematic_ops'].append(f"{op_type}: {count} occurrences")

    # Check for FP32 operations that could be FP16
    precision_info = analysis.get('precision_info', {})
    for key, count in precision_info.items():
        if 'FLOAT32' in key or '1' in key:  # dataType 1 is FLOAT32
            issues['fp32_ops'].append(f"{key}: {count}")

    # Check for linear/matmul that should be conv2d
    if 'linear' in op_counts or 'matmul' in op_counts:
        linear_count = op_counts.get('linear', 0) + op_counts.get('matmul', 0)
        issues['linear_to_conv2d'].append(
            f"Found {linear_count} linear/matmul ops that should be Conv2d for ANE"
        )

    # Check for excessive memory operations
    memory_ops = ['reshape', 'transpose', 'squeeze', 'unsqueeze']
    memory_op_count = sum(op_counts.get(op, 0) for op in memory_ops)
    if memory_op_count > 10:
        issues['memory_ops'].append(
            f"Found {memory_op_count} memory layout operations (reshape/transpose/etc) "
            "that are expensive on ANE"
        )

    return dict(issues)


def print_analysis(model_path: Path, analysis: dict[str, Any], issues: dict[str, list[str]]) -> None:
    """Print formatted analysis results."""
    print(f"\n{'='*80}")
    print(f"CoreML Model Analysis: {model_path.name}")
    print(f"{'='*80}\n")

    print(f"Total Operations: {analysis['total_ops']}")
    print(f"Unique Operation Types: {analysis['unique_op_types']}\n")

    print("Top 20 Operation Types:")
    print("-" * 80)
    for op_type, count in list(analysis['op_counts'].items())[:20]:
        percentage = (count / analysis['total_ops']) * 100
        print(f"  {op_type:30s}: {count:4d} ({percentage:5.1f}%)")

    if len(analysis['op_counts']) > 20:
        remaining = len(analysis['op_counts']) - 20
        print(f"  ... and {remaining} more operation types")

    print(f"\n{'='*80}")
    print("Potential ANE Issues")
    print(f"{'='*80}\n")

    if not issues:
        print("No obvious ANE compatibility issues detected!")
    else:
        for category, problems in issues.items():
            print(f"{category.replace('_', ' ').title()}:")
            for problem in problems:
                print(f"  - {problem}")
            print()

    print(f"{'='*80}")
    print("Recommendations")
    print(f"{'='*80}\n")

    recommendations = []

    if 'linear_to_conv2d' in issues:
        recommendations.append(
            "1. Replace all nn.Linear layers with nn.Conv2d (kernel_size=1) throughout "
            "the model architecture"
        )

    if 'memory_ops' in issues:
        recommendations.append(
            "2. Minimize transposes and reshapes by maintaining 4D tensor layouts "
            "(batch, channels, height, width) throughout the network"
        )

    if 'problematic_ops' in issues and any('batch_norm' in p for p in issues['problematic_ops']):
        recommendations.append(
            "3. Replace BatchNorm layers with custom LayerNormANE (epsilon=1e-7, "
            "reordered bias/scale)"
        )

    if 'fp32_ops' in issues:
        recommendations.append(
            "4. Review FP32 operations and convert non-sensitive ops to FP16 for better "
            "ANE utilization"
        )

    if recommendations:
        for rec in recommendations:
            print(f"{rec}\n")
    else:
        print("Model appears to be well-optimized for ANE execution!\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CoreML model operations for ANE compatibility"
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the .mlpackage file to analyze"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed operation information"
    )

    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}")
        print("\nAvailable .mlpackage files in current directory:")
        for f in Path.cwd().glob("*.mlpackage"):
            print(f"  {f.name}")
        return

    print(f"Analyzing model: {args.model_path}")
    analysis = analyze_model_operations(args.model_path)

    if not analysis:
        return

    issues = identify_ane_issues(analysis)
    print_analysis(args.model_path, analysis, issues)

    if args.detailed:
        print(f"\n{'='*80}")
        print("Detailed Operation List")
        print(f"{'='*80}\n")

        for i, op in enumerate(analysis['op_details'][:50]):  # Limit to first 50
            print(f"Op {i+1}: {op['type']}")
            if op.get('inputs'):
                print(f"  Inputs: {', '.join(op['inputs'][:3])}")
            if op.get('outputs'):
                print(f"  Outputs: {', '.join(op['outputs'][:3])}")
            if op.get('attributes'):
                print(f"  Attributes: {op['attributes']}")
            print()

        if len(analysis['op_details']) > 50:
            print(f"... and {len(analysis['op_details']) - 50} more operations")


if __name__ == "__main__":
    main()
