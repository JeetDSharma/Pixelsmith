#!/usr/bin/env python3
"""
Ablation Study: Patch Size Memory Scaling for Pixelsmith Cascade
Based on COMPSCI 602 Project Report 5

Experimental Design:
- Fixed cascade: 1024 -> 2048 -> 4096
- Independent variable: Patch Size (64, 128, 256)
- Trials: 3 per patch size (for statistical significance)
- Metrics per step: Peak VRAM, Allocation Count, Runtime, N_patches

Hypotheses:
- H1 (Serial-Discard): Peak VRAM stays ~constant across resolutions
- H2 (Overhead-Scaling): Runtime scales linearly with N_patches
"""

import subprocess
import argparse
import os
import json
import re
from datetime import datetime
from collections import defaultdict

# === EXPERIMENTAL DESIGN ===

# Fixed control variables
FIXED_PROMPT = "a detailed futuristic cityscape at sunset, ultra realistic lighting"
FIXED_NEGATIVE = "low quality, blurry, distorted, text artifacts"
FIXED_GUIDANCE = 7.5
FIXED_SEED = 42  # Reproducibility

# Independent variable: patch sizes to test
PATCH_SIZES = [64, 128, 256]

# Trials per patch size
NUM_TRIALS = 3

# Test mode (quick validation)
TEST_PATCH_SIZES = [128]
TEST_TRIALS = 1


def run_cascade_trial(patch_size, trial_num, seed, dry_run=False):
    """Run a single cascade trial (1024 -> 2048 -> 4096) and return metrics."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "measure_memory.py")
    
    cmd = [
        "python", script_path,
        "--patch_size", str(patch_size),
        "--seed", str(seed + trial_num),  # Vary seed slightly per trial
        "--prompt", FIXED_PROMPT,
        "--negative_prompt", FIXED_NEGATIVE,
        "--guidance_scale", str(FIXED_GUIDANCE),
    ]
    
    config_str = f"P={patch_size}, trial={trial_num+1}"
    print(f"\n[TRIAL] {config_str}", flush=True)
    print(f"  Cascade: 1024 -> 2048 -> 4096", flush=True)
    
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {"status": "dry_run", "patch_size": patch_size, "trial": trial_num}
    
    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    
    # Parse JSON results from output
    metrics = {
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
            print(f"  [WARN] Failed to parse JSON output")
    
    # Determine overall status from cascade results
    steps = metrics.get("steps", [])
    if any(s.get("status") == "OOM" for s in steps):
        metrics["overall_status"] = "OOM"
        failed_step = next(s["step"] for s in steps if s.get("status") == "OOM")
        print(f"  [OOM] Out of memory at {failed_step}", flush=True)
    elif result.returncode != 0 or any(s.get("status") == "FAILED" for s in steps):
        metrics["overall_status"] = "FAILED"
        print(f"  [FAILED] Return code {result.returncode}", flush=True)
    else:
        metrics["overall_status"] = "SUCCESS"
        max_vram = metrics.get("max_peak_vram_mb", "N/A")
        total_runtime = metrics.get("total_runtime_sec", "N/A")
        print(f"  [SUCCESS] Max VRAM: {max_vram} MB, Total: {total_runtime}s", flush=True)
        # Print per-step summary
        for step in steps:
            print(f"    {step['step']}: {step['peak_vram_mb']:.1f} MB, {step['runtime_sec']:.1f}s, N={step['n_patches']}", flush=True)
    
    return metrics


def run_ablation(patch_sizes, num_trials, seed, dry_run=False):
    """Run ablation over patch sizes."""
    all_results = []
    total_trials = len(patch_sizes) * num_trials
    
    print(f"\n{'='*70}")
    print(f"PIXELSMITH ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Cascade: 1024 -> 2048 -> 4096 (fixed)")
    print(f"Patch Sizes: {patch_sizes}")
    print(f"Trials per patch size: {num_trials}")
    print(f"Total trials: {total_trials}")
    print(f"{'='*70}")
    
    trial_count = 0
    
    for patch_size in patch_sizes:
        patch_results = []
        
        for trial in range(num_trials):
            trial_count += 1
            print(f"\n[{trial_count}/{total_trials}]", end="")
            
            metrics = run_cascade_trial(patch_size, trial, seed, dry_run)
            patch_results.append(metrics)
            all_results.append(metrics)
            
            # Incremental save after each trial (crash-safe)
            if not dry_run:
                append_result(metrics)
        
        # Summary for this patch size
        if not dry_run:
            successful = [r for r in patch_results if r.get("overall_status") == "SUCCESS"]
            if successful:
                vrams = [r["max_peak_vram_mb"] for r in successful]
                runtimes = [r["total_runtime_sec"] for r in successful]
                median_vram = sorted(vrams)[len(vrams)//2]
                median_runtime = sorted(runtimes)[len(runtimes)//2]
                print(f"\n  >> Patch={patch_size}: {len(successful)}/{num_trials} succeeded", flush=True)
                print(f"     Median VRAM: {median_vram:.1f} MB, Runtime: {median_runtime:.1f}s", flush=True)
    
    return all_results


def save_results(results, output_dir="results/ablation", filename=None):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ablation_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}", flush=True)
    return filepath


def append_result(result, output_dir="results/ablation", filename="ablation_incremental.json"):
    """Append a single result incrementally (crash-safe)."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Load existing results or start fresh
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append(result)
    
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2)
    
    return filepath


def print_summary(results):
    """Print summary table of results."""
    print(f"\n{'='*90}")
    print("SUMMARY: Pixelsmith Cascade Memory Scaling")
    print(f"{'='*90}")
    
    # Group by patch_size
    grouped = defaultdict(list)
    for r in results:
        grouped[r["patch_size"]].append(r)
    
    for patch_size in sorted(grouped.keys()):
        trials = grouped[patch_size]
        successful = [t for t in trials if t.get("overall_status") == "SUCCESS"]
        
        print(f"\n--- Patch Size: {patch_size} ({len(successful)}/{len(trials)} succeeded) ---")
        
        if not successful:
            print("  No successful trials")
            continue
        
        # Aggregate per-step metrics across trials
        step_metrics = defaultdict(lambda: {"vram": [], "runtime": [], "n_patches": 0})
        for trial in successful:
            for step in trial.get("steps", []):
                step_metrics[step["step"]]["vram"].append(step["peak_vram_mb"])
                step_metrics[step["step"]]["runtime"].append(step["runtime_sec"])
                step_metrics[step["step"]]["n_patches"] = step["n_patches"]
        
        print(f"  {'Step':<15} {'Resolution':<12} {'N_patches':<10} {'VRAM (median)':<15} {'Runtime (median)':<15}")
        print(f"  {'-'*67}")
        
        for step_name in ["base_1024", "refine_2048", "refine_4096"]:
            if step_name in step_metrics:
                m = step_metrics[step_name]
                vram = sorted(m["vram"])[len(m["vram"])//2]
                runtime = sorted(m["runtime"])[len(m["runtime"])//2]
                res = step_name.split("_")[1]
                print(f"  {step_name:<15} {res:<12} {m['n_patches']:<10} {vram:.1f} MB{'':<7} {runtime:.1f}s")
    
    print(f"\n{'='*90}")
    print("HYPOTHESIS CHECK:")
    print("  H1 (Serial-Discard): VRAM should be ~constant across 1024/2048/4096 for same patch size")
    print("  H2 (Overhead-Scaling): Runtime should scale with N_patches")
    print(f"{'='*90}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixelsmith Cascade Ablation Study")
    parser.add_argument("--test", action="store_true",
                        help="Run quick test with single patch size")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--seed", type=int, default=FIXED_SEED,
                        help="Base random seed")
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=None,
                        help="Override patch sizes (e.g., --patch-sizes 64 128 256)")
    parser.add_argument("--trials", type=int, default=None,
                        help="Override number of trials per patch size")
    args = parser.parse_args()
    
    # Select configuration
    if args.test:
        patch_sizes = TEST_PATCH_SIZES
        num_trials = TEST_TRIALS
        print(">>> TEST MODE <<<")
    else:
        patch_sizes = PATCH_SIZES
        num_trials = NUM_TRIALS
    
    # Apply overrides
    if args.patch_sizes:
        patch_sizes = args.patch_sizes
    if args.trials:
        num_trials = args.trials
    
    # Run ablation
    results = run_ablation(patch_sizes, num_trials, args.seed, args.dry_run)
    
    if not args.dry_run and results:
        save_results(results)
        print_summary(results)
