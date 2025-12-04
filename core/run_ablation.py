#!/usr/bin/env python3
"""
Ablation Study: Patch Size vs Resolution Memory Scaling
Based on COMPSCI 602 Project Report 5

Full Factorial Design:
- Resolutions: 1024, 2048, 4096, 8192
- Patch Sizes: 64, 128, 256
- Trials: 3 per configuration
- Metrics: Peak VRAM, Allocation Count, Runtime
"""

import subprocess
import itertools
import argparse
import os
import json
import time
import re
from datetime import datetime
from collections import defaultdict

# === EXPERIMENTAL DESIGN (from Report 5) ===

# Fixed control variables
FIXED_PROMPT = "a detailed futuristic cityscape at sunset, ultra realistic lighting"
FIXED_NEGATIVE = "low quality, blurry, distorted, text artifacts"
FIXED_GUIDANCE = 7.5
FIXED_SEED = 42  # Reproducibility

# Independent variables
RESOLUTIONS = [1024, 2048, 4096, 8192]  # 1x, 2x, 4x, 8x base
PATCH_SIZES = [64, 128, 256]

# Trials per configuration
NUM_TRIALS = 3

# Test mode (smaller grid for validation)
TEST_RESOLUTIONS = [512, 1024]
TEST_PATCH_SIZES = [64, 128]
TEST_TRIALS = 1


def get_scale_for_resolution(target_res, base=1024):
    """Convert target resolution to max_scale parameter."""
    return target_res // base


def run_single_trial(resolution, patch_size, trial_num, seed, dry_run=False):
    """Run a single experimental trial and return metrics."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "measure_memory.py")
    
    cmd = [
        "python", script_path,
        "--resolution", str(resolution),
        "--patch_size", str(patch_size),
        "--seed", str(seed + trial_num),  # Vary seed slightly per trial
        "--prompt", FIXED_PROMPT,
        "--negative_prompt", FIXED_NEGATIVE,
        "--guidance_scale", str(FIXED_GUIDANCE),
    ]
    
    config_str = f"R={resolution}, P={patch_size}, trial={trial_num+1}"
    print(f"\n[TRIAL] {config_str}")
    
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {"status": "dry_run", "resolution": resolution, "patch_size": patch_size, "trial": trial_num}
    
    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    
    # Parse JSON results from output
    metrics = {
        "resolution": resolution,
        "patch_size": patch_size,
        "trial": trial_num,
        "returncode": result.returncode,
    }
    
    # Extract JSON results if present
    json_match = re.search(r'\[JSON_RESULTS\](.*?)\[/JSON_RESULTS\]', result.stdout, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            metrics.update(parsed)
        except json.JSONDecodeError:
            pass
    
    # Determine status
    if "out of memory" in result.stderr.lower() or metrics.get("status") == "OOM":
        metrics["status"] = "OOM"
        print(f"  [OOM] Out of memory at R={resolution}, P={patch_size}")
    elif result.returncode != 0:
        metrics["status"] = "FAILED"
        print(f"  [FAILED] Return code {result.returncode}")
    else:
        metrics["status"] = metrics.get("status", "SUCCESS")
        vram = metrics.get("peak_vram_mb", "N/A")
        runtime = metrics.get("runtime_sec", "N/A")
        print(f"  [SUCCESS] VRAM: {vram} MB, Runtime: {runtime}s")
    
    return metrics


def run_full_ablation(resolutions, patch_sizes, num_trials, seed, dry_run=False):
    """Run full factorial experiment."""
    all_results = []
    total_configs = len(resolutions) * len(patch_sizes)
    total_trials = total_configs * num_trials
    
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: Patch Size vs Resolution Memory Scaling")
    print(f"{'='*70}")
    print(f"Resolutions: {resolutions}")
    print(f"Patch Sizes: {patch_sizes}")
    print(f"Trials per config: {num_trials}")
    print(f"Total trials: {total_trials}")
    print(f"{'='*70}")
    
    trial_count = 0
    
    for resolution, patch_size in itertools.product(resolutions, patch_sizes):
        config_results = []
        
        for trial in range(num_trials):
            trial_count += 1
            print(f"\n[{trial_count}/{total_trials}]", end="")
            
            metrics = run_single_trial(resolution, patch_size, trial, seed, dry_run)
            config_results.append(metrics)
            all_results.append(metrics)
        
        # Summary for this configuration
        if not dry_run:
            successful = [r for r in config_results if r["status"] == "SUCCESS"]
            if successful:
                runtimes = [r["runtime_sec"] for r in successful]
                median_runtime = sorted(runtimes)[len(runtimes)//2]
                print(f"\n  >> Config (R={resolution}, P={patch_size}): {len(successful)}/{num_trials} succeeded, median runtime: {median_runtime:.1f}s")
    
    return all_results


def save_results(results, output_dir="results/ablation"):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"ablation_{timestamp}.json")
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def print_summary(results):
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print("SUMMARY: Peak VRAM vs Resolution (by Patch Size)")
    print(f"{'='*80}")
    print(f"{'Resolution':<12} {'Patch':<8} {'Trials':<8} {'Success':<8} {'Median VRAM':<14} {'Median Runtime':<14}")
    print("-" * 80)
    
    # Group by (resolution, patch_size)
    grouped = defaultdict(list)
    for r in results:
        key = (r["resolution"], r["patch_size"])
        grouped[key].append(r)
    
    for (res, patch), trials in sorted(grouped.items()):
        successful = [t for t in trials if t.get("status") == "SUCCESS"]
        
        if successful:
            runtimes = sorted([t.get("runtime_sec", 0) for t in successful])
            vrams = sorted([t.get("peak_vram_mb", 0) for t in successful])
            median_runtime = runtimes[len(runtimes)//2]
            median_vram = vrams[len(vrams)//2]
            runtime_str = f"{median_runtime:.1f}s"
            vram_str = f"{median_vram:.1f} MB"
        else:
            runtime_str = "N/A"
            vram_str = "N/A"
        
        print(f"{res:<12} {patch:<8} {len(trials):<8} {len(successful):<8} {vram_str:<14} {runtime_str:<14}")
    
    print(f"{'='*80}")
    
    # H1 Analysis hint
    print("\nH1 (Serial-Discard): Check if VRAM is constant across resolutions for each patch size")
    print("H2 (Overhead-Scaling): Check if VRAM increases linearly with N_patches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Size Ablation Study")
    parser.add_argument("--test", action="store_true",
                        help="Run small test (512-1024, fewer trials)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--seed", type=int, default=FIXED_SEED,
                        help="Base random seed")
    parser.add_argument("--resolutions", type=int, nargs="+", default=None,
                        help="Override resolutions")
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=None,
                        help="Override patch sizes")
    parser.add_argument("--trials", type=int, default=None,
                        help="Override number of trials")
    args = parser.parse_args()
    
    # Select configuration
    if args.test:
        resolutions = TEST_RESOLUTIONS
        patch_sizes = TEST_PATCH_SIZES
        num_trials = TEST_TRIALS
        print(">>> TEST MODE <<<")
    else:
        resolutions = RESOLUTIONS
        patch_sizes = PATCH_SIZES
        num_trials = NUM_TRIALS
    
    # Apply overrides
    if args.resolutions:
        resolutions = args.resolutions
    if args.patch_sizes:
        patch_sizes = args.patch_sizes
    if args.trials:
        num_trials = args.trials
    
    # Run ablation
    results = run_full_ablation(resolutions, patch_sizes, num_trials, args.seed, args.dry_run)
    
    if not args.dry_run and results:
        save_results(results)
        print_summary(results)
