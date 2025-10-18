# core/generate_image.py
# Step 1: Single-run image generation with metrics and zoomed crops
# Compatible with RTX 2080 Ti on Unity HPC Cluster

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import time
import uuid
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300

# import Pixelsmith wrapper (already defined in your repo)
from pixelsmith_pipeline import generate_image as pixelsmith_generate


def setup_device():
    """Detect and prepare CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    return gpu_name


def _make_dirs(base_dir="results"):
    """Ensure result directories exist."""
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "zoomed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    return base_dir


def _save_zoom_crops(image, save_prefix, zoom_specs):
    """Crop and save zoomed-in regions."""
    arr = np.array(image)
    h, w, _ = arr.shape
    saved_paths = []

    for name, (y_frac, x_frac, frac_h, frac_w) in zoom_specs.items():
        y1 = int(h * y_frac)
        y2 = int(y1 + h * frac_h)
        x1 = int(w * x_frac)
        x2 = int(x1 + w * frac_w)
        crop = arr[y1:y2, x1:x2]
        crop_img = Image.fromarray(crop)
        path = f"{save_prefix}_zoom_{name}.png"
        crop_img.save(path)
        saved_paths.append(path)

    return saved_paths


def generate_with_metrics(
    prompt,
    negative_prompt,
    h_res=1024,
    w_res=1024,
    slider=None,
    guidance_scale=7.5,
    image=None,
    seed=None,
    output_dir="results",
    save_images=True,
):
    """
    Run a single Pixelsmith generation and record metrics.
    """
    _make_dirs(output_dir)
    gpu_name = setup_device()

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    run_id = str(uuid.uuid4())[:8]
    base_name = f"{output_dir}/images/run_{run_id}"

    # Start timing and clear memory
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    im = pixelsmith_generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        h_res=h_res,
        w_res=w_res,
        image=image,
        slider=slider,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    runtime = round(time.time() - start, 2)
    max_mem_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # Save main image
    image_path = None
    if save_images:
        image_path = f"{base_name}.png"
        im.save(image_path)

        # Define zoom crops (fractions of image)
        zoom_specs = {
            "center": (0.45, 0.45, 0.1, 0.1),
            "top_left": (0.05, 0.05, 0.08, 0.08),
            "random": (
                random.uniform(0.2, 0.7),
                random.uniform(0.2, 0.7),
                0.08,
                0.08,
            ),
        }
        _save_zoom_crops(im, base_name, zoom_specs)

    gc.collect()
    torch.cuda.empty_cache()

    metrics = {
        "run_id": run_id,
        "prompt": prompt,
        "resolution": f"{h_res}x{w_res}",
        "slider": slider,
        "guidance_scale": guidance_scale,
        "runtime_sec": runtime,
        "gpu_memory_gb": max_mem_gb,
        "gpu_name": gpu_name,
        "seed": seed,
        "image_path": image_path,
    }

    return im, metrics


if __name__ == "__main__":
    # Minimal test run (safe for cluster)
    prompt = "a detailed futuristic cityscape at sunset, ultra realistic lighting"
    negative_prompt = "low quality, blurry, distorted, text artifacts"
    image, metrics = generate_with_metrics(
        prompt=prompt,
        negative_prompt=negative_prompt,
        h_res=1024,
        w_res=1024,
        guidance_scale=7.5,
        slider=None,
    )

    print("Run complete. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
