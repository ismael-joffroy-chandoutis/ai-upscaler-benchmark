# Changelog

This benchmark is maintained continuously. Each entry records what changed, what was
re-verified, and against which sources. Dates are ISO (YYYY-MM-DD).

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).
Versioning is date-anchored: a `MINOR` bump means new tools or a re-run of the
methodology, a `PATCH` bump means corrections and re-verification of existing entries.

## [0.2.4] - 2026-06-16

### Changed
- Corrected the reference NVMe to ~14,000 MB/s (Samsung 990 Pro 8TB), not 1400. The
  frame-sequence I/O analysis is updated: on this rig the GPU is always the limit, never the disk.
- De-cluttered the public docs: the "Gaps / what to measure next" sections are now tight "Scope
  and limitations" notes, and internal-process phrasing was removed. This is a comparative, not a
  to-do list.
- Named the model on every "Topaz via fal" recommendation (Starlight Precise 2.5 / Proteus / Gaia 2).

### Added
- `docs/recipes.md`: exact, copy-pasteable settings per model (measured configs + recommended
  params) for both video and image.

## [0.2.3] - 2026-06-16

### Added (Topaz classic tier measured on the RTX 5090)
- Topaz Video AI runs headless on NOMAD via its bundled tvai ffmpeg CLI (offline, from a cached
  auth.tpz, despite cancelled subscriptions). Measured (Proteus prob-4, `-f null`, 100-frame clip,
  steady-state): 15 fps at 1080p, 7.3 fps at 4K, peak VRAM ~1.8-2.7 GB. Iris/Gaia same featherweight
  class.
- Confirms the community ~4.7-7 fps@4x and quantifies the classic-vs-diffusion gap: classic is
  20-100x faster and 4-15x lighter than SeedVR2/FlashVSR, but only sharpens.
- Topaz Starlight (diffusion) is not measurable via the tvai ffmpeg CLI (needs Neuroserver).

## [0.2.2] - 2026-06-16

### Added (FlashVSR measured on the RTX 5090)
- FlashVSR (FlashVSR_plus build, isolated venv torch 2.9+cu128, SageAttention) via its `run.py`
  CLI on a 25-frame 720p clip: tiny x2 (720p->1440p) = 0.80 fps / ~31 GB (works, near the ceiling).
- x4 (4K-class) is blocked on 32 GB: OOM without tiling, and the tiled-dit / full paths hit a
  FlashVSR_plus repo bug (UnboundLocalError temp_name). The marketed ~17 fps on A100 does not hold
  on the 5090 at these settings. FlashVSR needs >= 21 input frames (streaming window).

### Still pending
- Remaining diffusion tools on the 5090: SUPIR, CCSR, Clarity; SeedVR2 7B at 4K and fp16 variants.

## [0.2.1] - 2026-06-16

### Added (SeedVR2 diffusion tier MEASURED on the RTX 5090)
- SeedVR2 via the node's `inference_cli.py` on NOMAD (torch 2.7+cu128, SageAttention 3, 9-frame
  720p clip): 3B fp8 1080p batch 9 = 0.11 fps / 31.4 GB; 3B fp8 4K batch 1 (tiled + block-swap +
  cpu offload) = 0.13 fps / 9 GB; 7B fp8 1080p batch 5 (offload) = 0.35 fps / 25.5 GB; 3B fp8 4K
  batch 9 = OOM.
- Key finding: SeedVR2 throughput on the 5090 is dominated by VAE-decode batching, not model size
  or resolution. Small batch + VAE tiling + block-swap (`--dit_offload_device cpu`) is both faster
  and lower-VRAM than naive large batches; 32 GB binds at large temporal batches.
- Env note (NOMAD ComfyUI venv): diffusers 0.27.2 was already broken against huggingface_hub 1.4.1;
  upgraded to diffusers 0.38.0 (a repair) and added opencv-python-headless, rotary_embedding_torch,
  omegaconf, peft, gguf for the SeedVR2 CLI. Torch left untouched (2.7.0+cu128).

### Still pending
- Remaining diffusion tools (SUPIR, CCSR, Clarity, FlashVSR) on the 5090; SeedVR2 7B at 4K.

## [0.2.0] - 2026-06-16

### Added (first on-device RTX 5090 measurements)
- Measured speed and peak VRAM for the deterministic tier on the reference 5090 (NOMAD, torch
  2.7+cu128, driver 596.21), via `scripts/bench_upscale.py`. RealESRGAN x4plus and 4x-UltraSharp:
  1.71 fps fp16 / 0.88 fps fp32 at full-frame x4 of 720p (5120x2880), peak VRAM 7.7 / 15.4 GB;
  anime_6B 5.3 fps fp16. Full results in `data/measurements-5090.json`.
- Finding: fp16 is a free 2x speed-and-VRAM win; the eager full-frame number is far below the
  third-party TensorRT 12.7 fps figure (which is at a 4x-smaller output).

### Still pending
- Diffusion-tier measurements (SeedVR2, SUPIR, CCSR, Clarity): nodes are installed in the NOMAD
  ComfyUI, but weights need downloading (several GB each). SDXL base is already present for SUPIR.

## [0.1.1] - 2026-06-16

### Changed (adversarial verification pass)
- Confirmed **Astra 2** and **Starlight Precise 2.5** are real Topaz products (the earlier
  "name not found" reading was a search miss); Precise 2.5 runs **locally** on the 5090 with a
  vendor-cited ~12 fps 1080p-to-4K figure.
- License corrections: **VEnhancer = Apache-2.0** (commercial OK), **Real-CUGAN = MIT**
  (commercial OK), **AdcSR = Apache-2.0** (code), **HunyuanVideo 1.5 = Tencent Hunyuan Community
  License** (not Apache).
- Struck a fabricated "Magnific 4.6/5 on AI images" stat; triangulated Magnific standalone tiers
  ($39 / $99 / $299).
- Fixed Hallucination-Score figures (MUSIQ rho -0.23, not -0.16) and removed invented proxy names
  (DINO-HS / Qwen-HS) in favour of the paper's DINO/CLIP-feature proxies.
- Pinned FlashVSR to OpenImagingLab / Shanghai AI Lab on a Wan2.1-T2V-1.3B base.

### Added
- NVIDIA RTX VSR (free Blackwell-native ComfyUI node, the exact 720p-to-4K case), open frame
  interpolation (RIFE / GIMM-VFI), Upscayl, chaiNNer, SPAN/DAT, DiT4SR, a SUPIR frame-by-frame
  note, and a bicubic/Lanczos baseline floor.

### Still pending
- On-device RTX 5090 measurements (the one genuinely missing layer).
- Publication to GitHub (awaiting go).

## [0.1.0] - 2026-06-16

### Added
- Initial release. Video and image upscaler landscape as of June 2026.
- Methodology for benchmarking upscalers on synthetic (AI-generated) input with no
  high-resolution ground truth.
- Machine-readable tool registry (`data/`) so tables can be regenerated from data.

### Notes
- Every quantitative claim carries a source and a `last_verified` date in the registry.
- Entries marked `UNVERIFIED` are kept visible on purpose rather than dropped, so the
  reader sees the edge of what is confirmed.
