#!/usr/bin/env python3
"""
Single-run memory measurement for ablation study.
Generates at a specific resolution and reports detailed VRAM metrics.

Usage:
    python measure_memory.py --resolution 2048 --patch_size 128
"""

import os, sys, time, gc, torch, argparse, json
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixelsmith_pipeline import generate_image as pixelsmith_generate


def clear_gpu():
    """Force GPU memory cleanup and reset all stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()  # Reset allocation counters
        torch.cuda.synchronize()


def get_memory_stats():
    """Get current GPU memory statistics including allocation counts."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    torch.cuda.synchronize()
    
    # Get detailed memory stats for allocation count
    stats = torch.cuda.memory_stats()
    
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1e6, 2),
        "current_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 2),
        # Allocation counts (N_alloc) - total malloc calls
        "alloc_count": stats.get("allocation.all.allocated", 0),
        "alloc_count_large": stats.get("allocation.large_pool.allocated", 0),
        "alloc_count_small": stats.get("allocation.small_pool.allocated", 0),
    }


def run_measurement(resolution, patch_size, seed, prompt, negative_prompt, guidance_scale):
    """Run a single generation and measure memory."""
    
    print(f"\n{'='*60}")
    print(f"MEMORY MEASUREMENT")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Patch Size: {patch_size}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")
    
    # Clear GPU before measurement
    clear_gpu()
    baseline = get_memory_stats()
    print(f"Baseline memory: {baseline.get('current_allocated_mb', 0):.1f} MB")
    
    # Run generation
    start_time = time.time()
    
    try:
        img = pixelsmith_generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            h_res=resolution,
            w_res=resolution,
            image=None,  # Direct generation (no upscaling)
            slider=None,
            guidance_scale=guidance_scale,
            seed=seed,
            patch_size=patch_size,
        )
        
        runtime = time.time() - start_time
        status = "SUCCESS"
        
    except torch.cuda.OutOfMemoryError as e:
        runtime = time.time() - start_time
        status = "OOM"
        img = None
        print(f"[OOM ERROR] {e}")
        
    except Exception as e:
        runtime = time.time() - start_time
        status = "FAILED"
        img = None
        print(f"[ERROR] {e}")
    
    # Get final memory stats
    final_stats = get_memory_stats()
    
    # Compile results
    # Independent variables (inputs): resolution, patch_size
    # Dependent variables (measured): peak_vram_mb, alloc_count, runtime_sec
    # Derived (for H2 regression): n_patches = (resolution / patch_size)^2
    results = {
        # Independent variables
        "resolution": resolution,
        "patch_size": patch_size,
        "seed": seed,
        # Dependent variables (measured)
        "status": status,
        "runtime_sec": round(runtime, 2),
        "peak_vram_mb": final_stats.get("peak_allocated_mb", 0),
        "peak_reserved_mb": final_stats.get("peak_reserved_mb", 0),
        "alloc_count": final_stats.get("alloc_count", 0),  # N_alloc - total malloc calls
        "alloc_count_large": final_stats.get("alloc_count_large", 0),
        "alloc_count_small": final_stats.get("alloc_count_small", 0),
        # Derived (for H2 analysis)
        "n_patches": (resolution // patch_size) ** 2 if patch_size > 0 else 0,
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Status: {status}")
    print(f"Runtime: {runtime:.2f} sec")
    print(f"Peak VRAM Allocated: {results['peak_vram_mb']:.2f} MB")
    print(f"Peak VRAM Reserved: {results['peak_reserved_mb']:.2f} MB")
    print(f"Allocation Count (N_alloc): {results['alloc_count']}")
    print(f"  - Large pool: {results['alloc_count_large']}")
    print(f"  - Small pool: {results['alloc_count_small']}")
    print(f"N_patches (derived): {results['n_patches']}")
    print(f"{'='*60}")
    
    # Output JSON for parsing
    print(f"\n[JSON_RESULTS]{json.dumps(results)}[/JSON_RESULTS]")
    
    # Save image if successful
    if img is not None:
        os.makedirs("results/ablation", exist_ok=True)
        img_path = f"results/ablation/measure_R{resolution}_P{patch_size}_S{seed}.png"
        img.save(img_path)
        print(f"Image saved: {img_path}")
    
    # Cleanup
    clear_gpu()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-run memory measurement")
    parser.add_argument("--resolution", type=int, required=True,
                        help="Target resolution (e.g., 1024, 2048, 4096, 8192)")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Patch size for denoising")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--prompt", type=str, 
                        default="a detailed futuristic cityscape at sunset, ultra realistic lighting")
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, blurry, distorted, text artifacts")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()
    
    results = run_measurement(
        resolution=args.resolution,
        patch_size=args.patch_size,
        seed=args.seed,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
    )
