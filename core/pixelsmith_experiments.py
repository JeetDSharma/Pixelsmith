# core/pixelsmith_experiments.py
# Experiment controller: parameter grid -> runs -> aggregated CSV + JSON logs.

import os, sys, json, uuid, argparse, itertools, datetime
import pandas as pd

# stable imports regardless of where sbatch starts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PROJ_DIR)

from core.generate_image import setup_dirs, generate_progressive  # uses your existing implementation


def read_prompts(path: str | None):
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return lines
    # default small set
    return [
        "a detailed futuristic cityscape at sunset, ultra realistic lighting",
        "a macro photograph of a dew-covered leaf with bokeh lights, photorealistic",
        "an astronaut riding a horse on mars, cinematic, highly detailed",
        "a medieval village at dawn with fog and lanterns, 8k detail",
        "a Japanese zen garden with raked sand and bonsai, soft light",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str,
                        default="blurry, overexposed, underexposed, low quality, noise, distorted edges, text artifacts")
    parser.add_argument("--base_resolutions", type=str, default="1024")  # comma-separated: "1024,2048"
    parser.add_argument("--guidance_scales", type=str, default="6.0,7.5,9.0")
    parser.add_argument("--seeds", type=str, default="1643170768,2456789123")
    parser.add_argument("--max_scale", type=int, default=8)  # 1x->2x->... up to this
    parser.add_argument("--outdir", type=str, default=os.path.join(BASE_DIR, "results"))
    parser.add_argument("--csv", type=str, default=os.path.join(BASE_DIR, "results", "pixelsmith_metrics.csv"))
    args = parser.parse_args()

    setup_dirs()  # ensures results/{images,zoomed,logs}

    prompts = read_prompts(args.prompts_file)
    neg = args.negative_prompt
    base_res_list = [int(x) for x in args.base_resolutions.split(",") if x]
    guidance_list = [float(x) for x in args.guidance_scales.split(",") if x]
    seed_list = [int(x) for x in args.seeds.split(",") if x]

    # grid
    grid = list(itertools.product(prompts, base_res_list, guidance_list, seed_list))

    all_records = []
    run_batch_id = uuid.uuid4().hex[:8]
    started = datetime.datetime.utcnow().isoformat() + "Z"

    print(f"Batch ID: {run_batch_id}")
    print(f"Total runs: {len(grid)}")

    for idx, (prompt, base_res, guidance, seed) in enumerate(grid, 1):
        print(f"\n[{idx}/{len(grid)}] prompt_id={hash(prompt)%10000} base={base_res} guidance={guidance} seed={seed}")

        # generate full progressive chain for this config
        stage_metrics = generate_progressive(
            prompt=prompt,
            negative_prompt=neg,
            h_res=base_res,
            w_res=base_res,
            guidance_scale=guidance,
            max_scale=args.max_scale,
            seed=seed,
        )

        # enrich and collect
        for sm in stage_metrics:
            rec = {
                "batch_id": run_batch_id,
                "prompt": prompt,
                "prompt_id": hash(prompt) % 10_000,
                "negative_prompt": neg,
                "base_res": f"{base_res}x{base_res}",
                "max_scale": args.max_scale,
                "guidance_scale": guidance,
                "seed": seed,
                # stage fields from generate_progressive()
                **sm,  # includes: run_id, scale, resolution, slider, runtime_sec, gpu_memory_gb, paths
            }
            all_records.append(rec)

        # also dump one JSON per (prompt,base,guidance,seed)
        json_name = f"results/logs/run_{stage_metrics[0]['run_id']}_meta.json"
        with open(os.path.join(BASE_DIR, json_name), "w") as jf:
            json.dump({
                "batch_id": run_batch_id,
                "started": started,
                "prompt": prompt,
                "negative_prompt": neg,
                "base_res": base_res,
                "guidance_scale": guidance,
                "seed": seed,
                "max_scale": args.max_scale,
                "stages": stage_metrics,
            }, jf, indent=2)

    # write aggregated CSV (append if exists)
    df = pd.DataFrame(all_records)
    csv_path = args.csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.isfile(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_all = pd.concat([df_existing, df], ignore_index=True)
        df_all.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

    print(f"\nSaved aggregated metrics to: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
