#!/usr/bin/env python3
"""
Single-run memory measurement for ablation study.
Generates at a specific resolution and reports detailed VRAM metrics.

Usage:
    python measure_memory.py --resolution 2048 --patch_size 128
"""

import os, sys, time, gc, torch, argparse, json, threading
import numpy as np
from PIL import Image
from datetime import datetime

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


def get_gpu_info():
    """Get GPU environment info for reproducibility."""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_vram_mb": round(props.total_memory / 1e6, 2),
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


class MemoryTracer:
    """Background thread to sample GPU memory over time for sawtooth plots."""
    
    def __init__(self, interval=0.5):
        self.interval = interval
        self.trace = []  # [(time_sec, allocated_mb, reserved_mb), ...]
        self.running = False
        self.thread = None
        self.start_time = None
    
    def _sample_loop(self):
        while self.running:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                t = time.time() - self.start_time
                allocated = torch.cuda.memory_allocated() / 1e6
                reserved = torch.cuda.memory_reserved() / 1e6
                self.trace.append({
                    "time_sec": round(t, 2),
                    "allocated_mb": round(allocated, 2),
                    "reserved_mb": round(reserved, 2),
                })
            time.sleep(self.interval)
    
    def start(self):
        self.trace = []
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.trace


def run_measurement(resolution, patch_size, seed, prompt, negative_prompt, guidance_scale):
    """Run a single generation and measure memory."""
    
    print(f"\n{'='*60}")
    print(f"MEMORY MEASUREMENT")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Patch Size: {patch_size}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")
    
    # Get GPU info for reproducibility
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('gpu_name', 'N/A')} ({gpu_info.get('gpu_total_vram_mb', 0):.0f} MB)")
    
    # Clear GPU before measurement
    clear_gpu()
    baseline = get_memory_stats()
    print(f"Baseline memory: {baseline.get('current_allocated_mb', 0):.1f} MB")
    
    # Start memory tracer for sawtooth plot
    tracer = MemoryTracer(interval=0.5)
    tracer.start()
    
    # Run generation
    timestamp = datetime.now().isoformat()
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
    
    # Stop memory tracer and get trace
    memory_trace = tracer.stop()
    
    # Get final memory stats
    final_stats = get_memory_stats()
    
    # Compile results
    # Independent variables (inputs): resolution, patch_size
    # Dependent variables (measured): peak_vram_mb, alloc_count, runtime_sec
    # Derived (for H2 regression): n_patches = (resolution / patch_size)^2
    results = {
        # Metadata
        "timestamp": timestamp,
        # GPU environment (reproducibility)
        **gpu_info,
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
        # Memory trace for sawtooth plots
        "memory_trace": memory_trace,
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
    
    # Save image if successful (use project root for consistency)
    if img is not None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "results", "ablation")
        zoomed_dir = os.path.join(project_root, "results", "zoomed")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(zoomed_dir, exist_ok=True)
        
        # Save full image
        img_path = os.path.join(output_dir, f"measure_R{resolution}_P{patch_size}_S{seed}.png")
        img.save(img_path)
        print(f"Image saved: {img_path}")
        
        # Save center crop (512x512) for detail comparison
        crop_size = 512
        cx, cy = img.width // 2, img.height // 2
        crop = img.crop((cx - crop_size//2, cy - crop_size//2, 
                         cx + crop_size//2, cy + crop_size//2))
        crop_path = os.path.join(zoomed_dir, f"zoom_R{resolution}_P{patch_size}_S{seed}.png")
        crop.save(crop_path)
        print(f"Zoomed crop saved: {crop_path}")
        
        # Record file sizes (quality/complexity proxy)
        results["image_file_size_kb"] = round(os.path.getsize(img_path) / 1024, 2)
        results["crop_file_size_kb"] = round(os.path.getsize(crop_path) / 1024, 2)
    
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
