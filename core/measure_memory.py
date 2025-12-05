#!/usr/bin/env python3
"""
Cascaded memory measurement for Pixelsmith ablation study.
Implements the proper two-step framework: Base (1024) -> 2K -> 4K

Usage:
    python measure_memory.py --patch_size 128
    python measure_memory.py --patch_size 256
"""

import os, sys, time, gc, torch, argparse, json, threading, traceback
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


def run_single_step(step_name, input_image, target_resolution, patch_size, seed, 
                    prompt, negative_prompt, guidance_scale, slider):
    """
    Run a single refinement step and measure memory.
    
    Args:
        step_name: Label for this step (e.g., "base", "2k", "4k")
        input_image: PIL Image from previous step (None for base generation)
        target_resolution: Target resolution for this step
        patch_size: Patch size for denoising
        ... other generation params
    
    Returns:
        (output_image, results_dict)
    """
    print(f"\n{'='*60}")
    print(f"STEP: {step_name.upper()}")
    print(f"Input: {'None (text-to-image)' if input_image is None else f'{input_image.size[0]}x{input_image.size[1]}'}")
    print(f"Target: {target_resolution}x{target_resolution}")
    print(f"Patch Size: {patch_size}")
    print(f"Slider: {slider}")
    print(f"{'='*60}")
    
    # Clear GPU and reset stats before this step
    clear_gpu()
    baseline = get_memory_stats()
    print(f"Baseline memory: {baseline.get('current_allocated_mb', 0):.1f} MB")
    
    # Start memory tracer
    tracer = MemoryTracer(interval=0.25)
    tracer.start()
    
    timestamp = datetime.now().isoformat()
    start_time = time.time()
    
    try:
        img = pixelsmith_generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            h_res=target_resolution,
            w_res=target_resolution,
            image=input_image,  # None for base, PIL Image for refinement
            slider=slider,
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
        print(f"[ERROR] {type(e).__name__}: {e}")
        print("[TRACEBACK]")
        traceback.print_exc()
        print("[/TRACEBACK]")
    
    # Stop tracer and get memory stats
    memory_trace = tracer.stop()
    final_stats = get_memory_stats()
    
    # Compile step results
    results = {
        "step": step_name,
        "timestamp": timestamp,
        "input_resolution": input_image.size[0] if input_image else 0,
        "target_resolution": target_resolution,
        "patch_size": patch_size,
        "slider": slider,
        "seed": seed,
        "status": status,
        "runtime_sec": round(runtime, 2),
        "peak_vram_mb": final_stats.get("peak_allocated_mb", 0),
        "peak_reserved_mb": final_stats.get("peak_reserved_mb", 0),
        "alloc_count": final_stats.get("alloc_count", 0),
        "alloc_count_large": final_stats.get("alloc_count_large", 0),
        "alloc_count_small": final_stats.get("alloc_count_small", 0),
        # Patch size is in latent space; SDXL VAE has 8x downsampling
        "latent_resolution": target_resolution // 8,
        "n_patches": ((target_resolution // 8) // patch_size) ** 2 if patch_size > 0 else 0,
        "memory_trace": memory_trace,
    }
    
    # Print step results
    print(f"\nStep '{step_name}' complete:")
    print(f"  Status: {status}")
    print(f"  Runtime: {runtime:.2f} sec")
    print(f"  Peak VRAM: {results['peak_vram_mb']:.2f} MB")
    print(f"  N_patches: {results['n_patches']}")
    
    return img, results


def run_cascade(patch_size, seed, prompt, negative_prompt, guidance_scale):
    """
    Run the full Pixelsmith cascade: Base (1024) -> 2K -> 4K
    Measures memory independently for each step.
    """
    print(f"\n{'#'*60}")
    print(f"PIXELSMITH CASCADE MEASUREMENT")
    print(f"Patch Size: {patch_size}")
    print(f"Seed: {seed}")
    print(f"Pipeline: 1024 -> 2048 -> 4096")
    print(f"{'#'*60}")
    
    # Slider progression map (matches generate_image.py)
    slider_map = {
        1024: None,  # base generation
        2048: 22,
        4096: 34,
    }
    
    # Get GPU info
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('gpu_name', 'N/A')} ({gpu_info.get('gpu_total_vram_mb', 0):.0f} MB)")
    
    all_results = {
        "cascade_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "patch_size": patch_size,
        "seed": seed,
        **gpu_info,
        "steps": [],
    }
    
    # Step 1: Base generation at 1024x1024
    base_img, base_results = run_single_step(
        step_name="base_1024",
        input_image=None,  # Text-to-image
        target_resolution=1024,
        patch_size=patch_size,
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        slider=slider_map[1024],
    )
    all_results["steps"].append(base_results)
    
    if base_img is None:
        print("\n[ABORT] Base generation failed, cannot continue cascade.")
        print(f"\n[JSON_RESULTS]{json.dumps(all_results)}[/JSON_RESULTS]")
        return all_results
    
    # Step 2: Refine to 2048x2048
    img_2k, results_2k = run_single_step(
        step_name="refine_2048",
        input_image=base_img,
        target_resolution=2048,
        patch_size=patch_size,
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        slider=slider_map[2048],
    )
    all_results["steps"].append(results_2k)
    
    if img_2k is None:
        print("\n[ABORT] 2K refinement failed, cannot continue cascade.")
        print(f"\n[JSON_RESULTS]{json.dumps(all_results)}[/JSON_RESULTS]")
        return all_results
    
    # Step 3: Refine to 4096x4096
    img_4k, results_4k = run_single_step(
        step_name="refine_4096",
        input_image=img_2k,
        target_resolution=4096,
        patch_size=patch_size,
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        slider=slider_map[4096],
    )
    all_results["steps"].append(results_4k)
    
    # Summary
    print(f"\n{'#'*60}")
    print("CASCADE SUMMARY")
    print(f"{'#'*60}")
    total_runtime = sum(s["runtime_sec"] for s in all_results["steps"])
    max_vram = max(s["peak_vram_mb"] for s in all_results["steps"])
    print(f"Total Runtime: {total_runtime:.2f} sec")
    print(f"Max Peak VRAM (across steps): {max_vram:.2f} MB")
    for step in all_results["steps"]:
        print(f"  {step['step']}: {step['status']} | {step['runtime_sec']:.1f}s | {step['peak_vram_mb']:.1f} MB | N={step['n_patches']}")
    
    all_results["total_runtime_sec"] = round(total_runtime, 2)
    all_results["max_peak_vram_mb"] = max_vram
    
    # Save final image
    if img_4k is not None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "results", "ablation")
        zoomed_dir = os.path.join(project_root, "results", "zoomed")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(zoomed_dir, exist_ok=True)
        
        # Save full 4K image
        img_path = os.path.join(output_dir, f"cascade_P{patch_size}_S{seed}_4096.png")
        img_4k.save(img_path)
        print(f"\n4K image saved: {img_path}")
        
        # Save center crop for detail comparison
        crop_size = 512
        cx, cy = img_4k.width // 2, img_4k.height // 2
        crop = img_4k.crop((cx - crop_size//2, cy - crop_size//2,
                            cx + crop_size//2, cy + crop_size//2))
        crop_path = os.path.join(zoomed_dir, f"cascade_P{patch_size}_S{seed}_crop.png")
        crop.save(crop_path)
        print(f"Zoomed crop saved: {crop_path}")
        
        all_results["image_file_size_kb"] = round(os.path.getsize(img_path) / 1024, 2)
        all_results["crop_file_size_kb"] = round(os.path.getsize(crop_path) / 1024, 2)
    
    # Output JSON for parsing
    print(f"\n[JSON_RESULTS]{json.dumps(all_results)}[/JSON_RESULTS]")
    
    # Cleanup
    clear_gpu()
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascaded Pixelsmith memory measurement")
    parser.add_argument("--patch_size", type=int, required=True,
                        help="Patch size for denoising (e.g., 64, 128, 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--prompt", type=str, 
                        default="a detailed futuristic cityscape at sunset, ultra realistic lighting")
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, blurry, distorted, text artifacts")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()
    
    results = run_cascade(
        patch_size=args.patch_size,
        seed=args.seed,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
    )
