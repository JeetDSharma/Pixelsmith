import os, sys, time, uuid, gc, torch, argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

# --- path fix so imports work via sbatch ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixelsmith_pipeline import generate_image as pixelsmith_generate


def setup_dirs():
    os.makedirs("results/images", exist_ok=True)
    os.makedirs("results/zoomed", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)


def measure_gpu_memory():
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return round(mem, 3)


def save_zoom(image, zoom_factor, region, label, run_id):
    y1, y2, x1, x2 = region
    arr = np.array(image)
    h, w = arr.shape[:2]
    # clamp to image bounds
    y1 = min(y1 * zoom_factor, h)
    y2 = min(y2 * zoom_factor, h)
    x1 = min(x1 * zoom_factor, w)
    x2 = min(x2 * zoom_factor, w)
    if y2 <= y1 or x2 <= x1:
        return None  # skip if region is invalid
    zoomed = arr[y1:y2, x1:x2]
    zoom_path = f"results/zoomed/{run_id}_{label}_{zoom_factor}x.png"
    Image.fromarray(zoomed).save(zoom_path)
    return zoom_path


def generate_progressive(prompt, negative_prompt, h_res, w_res, guidance_scale, max_scale, seed, patch_size=128):
    start_total = time.time()
    run_id = uuid.uuid4().hex[:8]
    stage_metrics = []

    # fixed zoom regions (y1, y2, x1, x2) for 1024x1024 base
    zoom_region_1 = (90, 290, 450, 600)
    zoom_region_2 = (180, 210, 520, 550)

    print(f"Run ID: {run_id}")
    print(f"Prompt: {prompt}")
    print(f"Max scale: {max_scale}x")

    # define slider progression pattern
    slider_map = {
        1: None,   # base generation
        2: 22,
        4: 34,
        8: 42,
        16: 48
    }

    image = None
    current_scale = 1

    while current_scale <= max_scale:
        stage_start = time.time()
        scaled_h = int(h_res * current_scale)
        scaled_w = int(w_res * current_scale)
        slider = slider_map.get(current_scale, 48)

        print(f"\n--- Generating {current_scale}x ({scaled_h}x{scaled_w}) ---")
        print(f"Slider: {slider}, Guidance: {guidance_scale}")

        img = pixelsmith_generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            h_res=scaled_h,
            w_res=scaled_w,
            image=image,
            slider=slider,
            guidance_scale=guidance_scale,
            seed=seed,
            patch_size=patch_size,
        )

        runtime = round(time.time() - stage_start, 2)
        mem = measure_gpu_memory()

        out_path = f"results/images/run_{run_id}_{current_scale}x.png"
        img.save(out_path)

        # zoom crops
        z1 = save_zoom(img, current_scale, zoom_region_1, "zoomA", run_id)
        z2 = save_zoom(img, current_scale, zoom_region_2, "zoomB", run_id)

        stage_metrics.append({
            "run_id": run_id,
            "scale": f"{current_scale}x",
            "resolution": f"{scaled_h}x{scaled_w}",
            "slider": slider,
            "guidance_scale": guidance_scale,
            "patch_size": patch_size,
            "runtime_sec": runtime,
            "gpu_memory_gb": mem,
            "image_path": out_path,
            "zoomA": z1,
            "zoomB": z2,
        })

        # prepare next iteration
        image = img
        current_scale *= 2
        gc.collect()
        torch.cuda.empty_cache()

    total_runtime = round(time.time() - start_total, 2)
    print(f"\nRun complete. Total runtime: {total_runtime} s")

    # log metrics per run
    log_path = f"results/logs/metrics_{run_id}.txt"
    with open(log_path, "w") as f:
        for s in stage_metrics:
            f.write(str(s) + "\n")

    print(f"Metrics saved to {log_path}")
    return stage_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str,
                        default="a detailed futuristic cityscape at sunset, ultra realistic lighting")
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, blurry, distorted, text artifacts")
    parser.add_argument("--h_res", type=int, default=1024)
    parser.add_argument("--w_res", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_scale", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Patch size for iterative denoising (default: 128)")
    args = parser.parse_args()

    setup_dirs()
    metrics = generate_progressive(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        h_res=args.h_res,
        w_res=args.w_res,
        guidance_scale=args.guidance_scale,
        max_scale=args.max_scale,
        seed=args.seed,
        patch_size=args.patch_size,
    )

    print("\nFinal Metrics Summary:")
    for m in metrics:
        print(m)
