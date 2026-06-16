"""
Measure upscaler throughput (sec/frame, fps) and peak VRAM on the local GPU.

Deterministic conv upscalers (ESRGAN family) loaded via spandrel, the same path
ComfyUI uses. Speed and VRAM are content-independent for these networks, so a
synthetic 720p input is sufficient for the throughput/VRAM layer. Quality on real
AI-generated footage is a separate axis (see docs/methodology.md).

Run inside a torch+CUDA env that supports your GPU (on Blackwell: torch cu128).
Edit UPSCALE_DIR / MODELS to point at your model files.

License: PolyForm Noncommercial 1.0.0 (see LICENSE.md).
"""
import time, json, os
import torch

UPSCALE_DIR = r"C:\pinokio\api\comfy.git\app\models\upscale_models"
MODELS = ["RealESRGAN_x4plus.pth", "4x-UltraSharp.pth", "RealESRGAN_x4plus_anime_6B.pth"]
H, W = 720, 1280
ITERS = 5


def load_model(path):
    from spandrel import ModelLoader
    d = ModelLoader().load_from_file(path)
    return d.model.train(False).cuda(), int(getattr(d, "scale", 4))


def bench(path, dtype):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, scale = load_model(path)
    model = model.to(dtype)
    x = torch.rand(1, 3, H, W, device="cuda", dtype=dtype)
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()
    out_h, out_w = int(y.shape[-2]), int(y.shape[-1])
    ts = []
    with torch.no_grad():
        for _ in range(ITERS):
            torch.cuda.synchronize(); t0 = time.time()
            y = model(x)
            torch.cuda.synchronize(); ts.append(time.time() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, x, y
    torch.cuda.empty_cache()
    return dict(model=os.path.basename(path), scale=scale, dtype=str(dtype).split(".")[-1],
                in_res=f"{W}x{H}", out_res=f"{out_w}x{out_h}",
                sec_per_frame=round(med, 4), fps=round(1.0 / med, 2),
                peak_vram_gb=round(peak, 2))


def main():
    results = []
    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0),
          "| cc", torch.cuda.get_device_capability(0))
    for mdl in MODELS:
        p = os.path.join(UPSCALE_DIR, mdl)
        if not os.path.exists(p):
            print("skip missing", mdl); continue
        for dt in [torch.float32, torch.float16]:
            try:
                r = bench(p, dt); results.append(r); print(json.dumps(r))
            except Exception as e:
                print("ERR", mdl, str(dt), repr(e)[:200])
    print("RESULTS_JSON " + json.dumps(results))


if __name__ == "__main__":
    main()
