#!/usr/bin/env python
"""Run benchmark sweep over multiple timm CNN models.

This script runs torchgeo_bench.py for each model configuration,
with resume mode enabled to skip already-computed results.

Usage:
    # Run all models on all datasets
    python run_timm_sweep.py
    
    # Run specific models
    python run_timm_sweep.py --models resnet18 resnet34 efficientnet_b0
    
    # Run on specific datasets
    python run_timm_sweep.py --datasets m-eurosat m-forestnet
    
    # Custom output file
    python run_timm_sweep.py --output my_results.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

# All timm models we created configs for
ALL_TIMM_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "densenet121",
    "densenet161",
    "vgg16",
    "vgg19",
    "regnetx_002",
    "regnetx_008",
    "regnety_002",
    "regnety_008",
]


def run_benchmark(
    model: str,
    output: str,
    datasets: list[str] | None = None,
    resume: bool = True,
    verbose: bool = False,
    device: str = "cuda:0",
) -> int:
    """Run torchgeo_bench.py for a single model.
    
    Args:
        model: Model config name (e.g., 'resnet18')
        output: Output CSV file path
        datasets: List of dataset names, or None for all
        resume: Enable resume mode
        verbose: Enable verbose output
        device: Device to use
        
    Returns:
        Return code from subprocess
    """
    cmd = [
        "python",
        "torchgeo_bench.py",
        f"model={model}",
        f"output={output}",
        f"device={device}",
        f"verbose={verbose}",
        f"resume={resume}",
    ]
    
    if datasets:
        dataset_str = "[" + ",".join(datasets) + "]"
        cmd.append(f"dataset.names={dataset_str}")
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark sweep over multiple timm CNN models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of model names to run (default: all timm models)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of dataset names (default: all datasets)",
    )
    parser.add_argument(
        "--output",
        default="timm_sweep_results.csv",
        help="Output CSV file (default: timm_sweep_results.csv)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode (recompute everything)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.models:
        models = args.models
        # Validate model names
        invalid = [m for m in models if m not in ALL_TIMM_MODELS]
        if invalid:
            print(f"Warning: Unknown models will be attempted: {invalid}")
    else:
        models = ALL_TIMM_MODELS
    
    print(f"Benchmark Sweep Configuration:")
    print(f"  Models: {len(models)} ({', '.join(models[:3])}" + 
          (f", ... and {len(models)-3} more" if len(models) > 3 else "") + ")")
    print(f"  Datasets: {'all' if not args.datasets else ', '.join(args.datasets)}")
    print(f"  Output: {args.output}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Device: {args.device}")
    print()
    
    # Run benchmark for each model
    failed = []
    for i, model in enumerate(models, 1):
        print(f"\n{'#'*60}")
        print(f"# Model {i}/{len(models)}: {model}")
        print(f"{'#'*60}")
        
        returncode = run_benchmark(
            model=model,
            output=args.output,
            datasets=args.datasets,
            resume=not args.no_resume,
            verbose=args.verbose,
            device=args.device,
        )
        
        if returncode != 0:
            failed.append(model)
            print(f"\n⚠️  Warning: {model} failed with return code {returncode}")
        else:
            print(f"\n✓ {model} completed successfully")
    
    # Summary
    print(f"\n\n{'='*60}")
    print("Sweep Summary")
    print(f"{'='*60}")
    print(f"Total models: {len(models)}")
    print(f"Successful: {len(models) - len(failed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed models: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n✓ All models completed successfully!")
        print(f"Results saved to: {args.output}")
        sys.exit(0)


if __name__ == "__main__":
    main()
