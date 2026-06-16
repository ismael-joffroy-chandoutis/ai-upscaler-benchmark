# Recipes: exact settings per model

Precise, copy-pasteable settings for each upscaler, for ~720p AI-generated input. Settings
tagged **[measured]** are the exact configs run on the reference RTX 5090 (see
[measurements-5090.json](../data/measurements-5090.json)); **[recommended]** are the
documented/community settings for tools not yet timed locally. Where a knob controls
fidelity-vs-invention, the value to start at is given.

Reference machine: RTX 5090 (32 GB), Ryzen 9 9950X, 96 GB RAM, 8 TB Samsung 990 Pro (~14,000 MB/s).

---

## VIDEO

### SeedVR2 (ByteDance, open, Apache-2.0) — local quality leader
ComfyUI node `numz/ComfyUI-SeedVR2_VideoUpscaler` (v2.5.24), or its `inference_cli.py`.
Set `PYTHONUTF8=1`. Weights auto-download to `models/SEEDVR2`. `--blocks_to_swap` requires
`--dit_offload_device cpu`.

- **Cinema 4K, fits 9 GB [measured 0.13 fps]:**
  `inference_cli.py OUT -i clip.mp4 --dit_model seedvr2_ema_3b_fp8_e4m3fn.safetensors --resolution 2160 --batch_size 1 --vae_encode_tiled --vae_decode_tiled --blocks_to_swap 16 --dit_offload_device cpu --attention_mode sageattn_3`
- **Volume 1080p, best speed [measured 0.35 fps / 25 GB]:**
  `--dit_model seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors --resolution 1080 --batch_size 5 --blocks_to_swap 16 --dit_offload_device cpu --attention_mode sageattn_3`
- **Quality stills/hero:** 7B fp16, `--attention_mode sageattn_3`.
- **Do NOT** use `--batch_size 9` (maxes 32 GB at 1080p, OOMs at 4K). Big batches are slower
  here, not faster: VAE-decode batching dominates.
- **Anti over-sharpen** (its documented failure on clean 720p AI): prefer 3B, keep batch small,
  do not stack sharpening.

### FlashVSR (FlashVSR_plus, open) — fast/streaming
`run.py` in the FlashVSR_plus app venv (torch 2.9). Needs **>= 21 input frames**.

- **x2 720p->1440p [measured 0.80 fps / 31 GB]:**
  `run.py OUT -i clip.mp4 -s 2 -m tiny -a sage -t bf16`
- **x4 / 4K: not usable on 32 GB** in this build — OOM without tiling, and `--tiled-dit` / `full`
  hit a repo bug (`UnboundLocalError: temp_name`). Stay at x2, or use SeedVR2 for 4K.

### Topaz Video (classic models, local) — featherweight, fast
Bundled `tvai` ffmpeg. Set `TVAI_MODEL_DIR` and `TVAI_MODEL_DATA_DIR` to
`C:\ProgramData\Topaz Labs LLC\Topaz Video\models`. Note: Topaz's ffmpeg has **no libx264** — output
with `h264_nvenc`, `prores_ks`, a PNG sequence, or `-f null -` to benchmark.

- **Proteus (face-safe, no hallucination) [measured 15 fps@1080p / 7.3 fps@4K, ~1.8-2.7 GB]:**
  `ffmpeg -i clip.mp4 -vf "tvai_up=model=prob-4:scale=0:w=3840:h=2160:device=0" -c:v h264_nvenc out.mp4`
  (set `w/h` to 1920x1080 for 1080p). Keep sliders gentle on synthetic input.
- **Gaia `ghq-5`** for stylized/animated, **Iris `iris-3`** for compression/face artifacts. Same
  featherweight class.
- **Starlight Precise 2.5** (diffusion, de-plastics faces) runs in the **GUI via Neuroserver**, not
  this CLI. Local NVIDIA, 12 GB min; vendor cites ~12 fps@1080p->4K (re-measure).

### Real-ESRGAN / community ESRGAN (local, cheap-fast)
ComfyUI "Upscale Image (using Model)" or VapourSynth `vs-mlrt` (TensorRT for real-time).

- **[measured] full-frame x4 720p->2880p:** RealESRGAN_x4plus / 4x-UltraSharp = 1.71 fps fp16,
  0.88 fps fp32 (peak 7.7 / 15.4 GB). Lighter `realesr-animevideov3` / anime_6B = 5.3 fps fp16.
- **Always run fp16** (2x speed, half VRAM, free). For video, add a temporal pass
  (`vs_temporalfix`) or it amplifies AI flicker. Commercial: use BSD/MIT models (Real-ESRGAN,
  Nomos), **not** 4x-UltraSharp/AnimeSharp (CC-BY-NC).

### API video (pick the model explicitly)
- **Topaz via fal / Replicate** — choose the variant: **`Starlight Precise 2.5`** for AI footage
  (quality), **`Proteus`** for face-safe, **`Gaia 2`** for half price. $0.01/s <=720p, $0.02/s
  ->1080p, $0.08/s >1080p (x2 at 60fps; Gaia 2 = half).
- **WaveSpeed SeedVR2** — flat per-second, 5 s min: $0.02/s 720p, $0.03/s 1080p, $0.05/s 4K
  (~$3/min of 4K, cheapest verified).
- **fal SeedVR2** — $0.001/MP (MP = w*h*frames); great at 1080p, ~4x pricier at 4K.

---

## IMAGE

### SUPIR (local diffusion, max detail) — non-commercial
kijai ComfyUI-SUPIR, PyTorch cu128.

- **[recommended] faithful on AI frames:** model **v0F**, `s_churn` ~0, `cfg` ~3-4, raise
  control/structure guidance, tight positive prompt, **tiled VAE on** (fp8 VAE off — it artifacts).
  Fits 32 GB; minutes per 4K still. Use v0Q only when you want maximum invented detail on a hero
  still and identity drift is acceptable.

### ControlNet Tile (SDXL or Flux) — most controllable, faithful
SDXL Tile + Ultimate SD Upscale, or Flux-tile (`jasperai/Flux.1-dev-Controlnet-Upscaler`, Flux is
non-commercial).

- **[recommended] clean AI artifacts without re-drawing:** **denoise 0.2-0.4**, ControlNet
  **strength < 0.7**, tile 1024. Higher denoise drifts faces/composition; higher CN gives edge
  artifacts. Flux-tile gives the least-plastic skin.

### Clarity Upscaler (local, open SD1.5) — tunable "Magnific-like"
- **[recommended] faithful:** `creativity` 0.2-0.35, high `resemblance`, tile strength < 0.7,
  low denoise. Push `creativity` up and it stylizes (reads as more synthetic on AI input).

### CCSR-v2 (local diffusion) — seed-stable, best for frame sequences
- **[recommended]** single-step mode for speed; it is low-invention by design (least flicker
  across frames). Good when you upscale AI-video frames as stills.

### One-step diffusion (OSEDiff / AdcSR / InvSR) — cheap-fast
- **[recommended]** 1 step. **InvSR**: pick the start timestep to dial invention (later start =
  less invented detail, safer on synthetic). OSEDiff/AdcSR are fixed 1-step, faithful, fast.

### GAN / transformer stills (local, deterministic)
- **[measured class]** Real-ESRGAN / 4x models via TensorRT ~real-time (see ESRGAN above), ~1.4 GB.
  **Real-CUGAN** conservative mode + denoise level for anime. **HAT `Real_HAT_GAN_SRx4`** /
  **DRCT** for max deterministic fidelity (Apache/MIT, commercial-clean, seconds/image).
- **AuraSR v2** (built for generated images): use the **overlapped** method
  (`upscale_4x_overlapped`) to avoid tile seams (~2x time).

### Topaz Gigapixel (stills) — local + API
- **Faithful:** non-generative **High Fidelity 3** or **Standard** (no identity invention).
- **Soft AI fill:** generative **Redefine / Recover** hallucinate detail (great on soft AI stills,
  risky on faces, do not use frame-by-frame on video).

### API image (pick the model / mode)
- **Magnific** — **Precision** mode for faithful; **Creative 30-50%** for reinvention (past 50%
  artifacts). ~$0.16 / 4K still.
- **fal SeedVR2 image** — $0.001/MP (~$0.83 / 100 4K stills), cheapest sane-quality.
- **Recraft Crisp** (deterministic) — $0.004/image on fal; **Topaz Gigapixel API** $8-12/100.

---

*Last updated 2026-06-16.*
