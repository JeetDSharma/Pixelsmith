#!/usr/bin/env python3
"""
Run a single Pixelsmith ablation experiment.
Called by SLURM job arrays for parallel execution.

Usage:
    python run_single_experiment.py --experiment_id 0    # First experiment
    python run_single_experiment.py --experiment_id 8    # Last experiment
"""

import argparse
import json
import os
import sys

# Import existing functions from run_ablation.py
from run_ablation import run_cascade_trial, FIXED_PROMPT, FIXED_NEGATIVE, FIXED_GUIDANCE

# Experiment matrix: (patch_size, trial_num)
# This creates all combinations of 3 patch sizes × 3 trials = 9 experiments
EXPERIMENT_MATRIX = [
    # Patch size 64
    (64, 0), (64, 1), (64, 2),
    # Patch size 128  
    (128, 0), (128, 1), (128, 2),
    # Patch size 256
    (256, 0), (256, 1), (256, 2),
]

def main():
    parser = argparse.ArgumentParser(description="Run single Pixelsmith ablation experiment")
    parser.add_argument("--experiment_id", type=int, required=True, 
                        help=f"Experiment ID (0-{len(EXPERIMENT_MATRIX)-1} for {len(EXPERIMENT_MATRIX)} total experiments)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Base random seed (will be modified by trial number)")
    parser.add_argument("--output_dir", type=str, default="results/ablation",
                        help="Output directory for individual results")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print experiment parameters without running")
    args = parser.parse_args()
    
    # Validate experiment ID
    if args.experiment_id < 0 or args.experiment_id >= len(EXPERIMENT_MATRIX):
        print(f"ERROR: Invalid experiment_id {args.experiment_id}")
        print(f"Valid range: 0 to {len(EXPERIMENT_MATRIX)-1}")
        return 1
    
    # Get experiment parameters
    patch_size, trial_num = EXPERIMENT_MATRIX[args.experiment_id]
    
    print(f"{'='*60}")
    print(f"PIXELSMITH SINGLE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Patch Size: {patch_size}")
    print(f"Trial Number: {trial_num}")
    print(f"Seed: {args.seed + trial_num}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*60}")
    
    if args.dry_run:
        print("[DRY RUN] Would run experiment with above parameters")
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the single experiment (reuses existing run_cascade_trial function!)
    print(f"\nStarting experiment...")
    try:
        metrics = run_cascade_trial(patch_size, trial_num, args.seed, dry_run=False)
        
        # Add experiment metadata
        metrics.update({
            "experiment_id": args.experiment_id,
            "matrix_position": f"patch_{patch_size}_trial_{trial_num}",
        })
        
        # Save individual result file
        result_filename = f"experiment_{args.experiment_id:02d}.json"
        result_path = os.path.join(args.output_dir, result_filename)
        
        with open(result_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nExperiment {args.experiment_id} completed:")
        print(f"  Patch Size: {patch_size}")
        print(f"  Trial: {trial_num}")
        print(f"  Status: {metrics.get('overall_status', 'UNKNOWN')}")
        if metrics.get('overall_status') == 'SUCCESS':
            print(f"  Max VRAM: {metrics.get('max_peak_vram_mb', 0):.1f} MB")
            print(f"  Total Runtime: {metrics.get('total_runtime_sec', 0):.1f} sec")
        print(f"  Results saved: {result_path}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Experiment {args.experiment_id} failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error result
        error_metrics = {
            "experiment_id": args.experiment_id,
            "patch_size": patch_size,
            "trial": trial_num,
            "overall_status": "ERROR", 
            "error_message": str(e),
            "matrix_position": f"patch_{patch_size}_trial_{trial_num}",
        }
        
        result_filename = f"experiment_{args.experiment_id:02d}.json"
        result_path = os.path.join(args.output_dir, result_filename)
        
        with open(result_path, 'w') as f:
            json.dump(error_metrics, f, indent=2)
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
