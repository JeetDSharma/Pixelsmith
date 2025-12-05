#!/usr/bin/env python3
"""
Collect and aggregate results from parallel Pixelsmith ablation experiments.

This script:
1. Finds all experiment_*.json files from parallel runs
2. Aggregates them into the same format as the original run_ablation.py
3. Generates the same summary output and charts data

Usage:
    python collect_parallel_results.py                    # Use default directory
    python collect_parallel_results.py --input results/ablation --output results/summary
"""

import argparse
import json
import os
import glob
from datetime import datetime

# Import existing functions to maintain compatibility
from run_ablation import print_summary, save_results

def collect_results(input_dir="results/ablation", verbose=True):
    """
    Collect all experiment_*.json files and aggregate into single results list.
    
    Returns:
        list: Same format as run_ablation.py's all_results
    """
    pattern = os.path.join(input_dir, "experiment_*.json")
    result_files = sorted(glob.glob(pattern))
    
    if not result_files:
        print(f"ERROR: No experiment files found in {input_dir}")
        print(f"Looking for pattern: {pattern}")
        return []
    
    if verbose:
        print(f"Found {len(result_files)} experiment result files:")
    
    all_results = []
    successful_count = 0
    failed_count = 0
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                all_results.append(result)
            
            # Count status
            status = result.get('overall_status', 'UNKNOWN')
            if status == 'SUCCESS':
                successful_count += 1
            else:
                failed_count += 1
            
            if verbose:
                exp_id = result.get('experiment_id', '?')
                patch_size = result.get('patch_size', '?')
                trial = result.get('trial', '?')
                print(f"  {os.path.basename(file_path)}: exp_{exp_id} patch_{patch_size}_trial_{trial} -> {status}")
                
        except Exception as e:
            print(f"WARNING: Failed to load {file_path}: {e}")
            failed_count += 1
    
    if verbose:
        print(f"\nCollection Summary:")
        print(f"  Total experiments: {len(all_results)}")
        print(f"  Successful: {successful_count}")
        print(f"  Failed: {failed_count}")
    
    return all_results

def validate_completeness(results, expected_experiments=9):
    """
    Check if we have all expected experiments.
    
    Returns:
        dict: Validation report
    """
    expected_matrix = [
        (64, 0), (64, 1), (64, 2),
        (128, 0), (128, 1), (128, 2),
        (256, 0), (256, 1), (256, 2)
    ]
    
    found_experiments = set()
    for result in results:
        patch_size = result.get('patch_size')
        trial = result.get('trial')
        if patch_size is not None and trial is not None:
            found_experiments.add((patch_size, trial))
    
    missing = set(expected_matrix) - found_experiments
    extra = found_experiments - set(expected_matrix)
    
    report = {
        'total_found': len(results),
        'expected_total': expected_experiments,
        'complete': len(missing) == 0,
        'missing_experiments': list(missing),
        'extra_experiments': list(extra)
    }
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Collect parallel ablation results")
    parser.add_argument("--input", type=str, default="results/ablation",
                        help="Input directory containing experiment_*.json files")
    parser.add_argument("--output", type=str, default="results/ablation", 
                        help="Output directory for aggregated results")
    parser.add_argument("--filename", type=str, default=None,
                        help="Output filename (default: auto-generated with timestamp)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate completeness, don't generate summary")
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print("PIXELSMITH PARALLEL RESULTS COLLECTION")
    print(f"{'='*70}")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"{'='*70}")
    
    # Collect all experiment results
    results = collect_results(args.input, verbose=not args.quiet)
    
    if not results:
        print("ERROR: No valid results found. Exiting.")
        return 1
    
    # Validate completeness
    validation = validate_completeness(results)
    print(f"\nValidation Report:")
    print(f"  Expected experiments: {validation['expected_total']}")
    print(f"  Found experiments: {validation['total_found']}")
    print(f"  Complete: {'✅ YES' if validation['complete'] else '❌ NO'}")
    
    if validation['missing_experiments']:
        print(f"  Missing: {validation['missing_experiments']}")
    if validation['extra_experiments']:
        print(f"  Extra: {validation['extra_experiments']}")
    
    if args.validate_only:
        return 0 if validation['complete'] else 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save aggregated results (reuses existing save_results function!)
    if args.filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ablation_parallel_{timestamp}.json"
    else:
        filename = args.filename
    
    output_path = save_results(results, output_dir=args.output, filename=filename)
    print(f"\n✅ Aggregated results saved to: {output_path}")
    
    # Generate summary output (reuses existing print_summary function!)
    print_summary(results)
    
    # Additional parallel-specific summary
    print(f"\n{'='*90}")
    print("PARALLEL EXECUTION SUMMARY")
    print(f"{'='*90}")
    
    # Calculate total runtime if we had run sequentially
    successful_results = [r for r in results if r.get('overall_status') == 'SUCCESS']
    if successful_results:
        total_sequential_time = sum(r.get('total_runtime_sec', 0) for r in successful_results)
        avg_runtime = total_sequential_time / len(successful_results)
        print(f"Sequential runtime (estimated): {total_sequential_time:.1f} seconds ({total_sequential_time/60:.1f} minutes)")
        print(f"Parallel runtime: ~{avg_runtime:.1f} seconds (longest experiment)")
        print(f"Speedup: ~{total_sequential_time/avg_runtime:.1f}x faster")
    
    print(f"Experiment files processed: {len(results)}")
    print(f"Results aggregation: COMPLETE")
    print(f"{'='*90}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
