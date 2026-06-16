# Video upscalers for AI-generated footage (June 2026)

Scope: you generate video at roughly 720p with models like LTX-2.3, Wan 2.2, Veo 3.1,
Seedance 2.0 and Kling, then you want to push it to 1080p or 4K. This page compares the
realistic options across local (single RTX 5090) and hosted-API paths, on three axes at
once: quality, cost and speed. There is no single winner. The recommendation depends on
the shot and the budget, so the page is built around use-cases, not a leaderboard.

Reference machine for every local claim: RTX 5090 (32 GB GDDR7, Blackwell), Ryzen 9
9950X, 96 GB RAM, 8 TB NVMe at ~1400 MB/s.

> **Verification status (v0.1.1).** Research pass plus an adversarial verification pass have
> both run; the corrections that pass surfaced are folded in (notably: Astra 2 and Starlight
> Precise 2.5 are confirmed real products, and Precise 2.5 runs locally). Hard numbers (fps,
> VRAM, prices) are still single-source unless a primary link is given, and nothing was measured
> on the reference 5090 yet (one figure aside). On-device 5090 measurement is the next milestone.

---

## 1. The one thing that changes everything: synthetic input

Every upscaler on this page was trained on *real* footage degraded by *real* pipelines
(compression, sensor noise, interlacing). Your input is different. A 720p Veo or Wan frame
has no real grain, but it carries warping, temporal flicker, waxy skin and over-smoothed
gradients. Two consequences run through every verdict below:

1. **GAN/CNN upscalers sharpen what is already there.** They have no temporal model and no
   semantic prior, so they crisp up plastic skin and amplify flicker rather than fix them.
2. **Diffusion upscalers regenerate detail.** They can *replace* AI mush with plausible
   texture, which is the better match for synthetic input, but they add their own drift and
   cost, and with no ground-truth 4K reference you are trading one synthetic look for
   another.

The sharpest single fact in the whole study: SeedVR2's own model card, the strongest open
local tool, **explicitly warns that it "tends to over-sharpen lightly degraded content like
720p AI-generated videos"** ([HF card](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)).
Your exact use-case is its documented failure mode. The fix is not a different tool, it is
dialing restoration strength down and finishing with a gentle non-generative pass. This is
the texture of the whole field right now: the tools are good, but they are tuned for real
footage, and synthetic input needs a lighter hand.

---

## 2. Recommendations by use-case (the Pareto answer)

| Use-case | Best LOCAL (5090) | Best API / cloud | Why |
|---|---|---|---|
| **Cinema 4K, quality-max** (hero shots) | **SeedVR2-7B FP16**, low restoration strength, finish with a light deterministic 1080p→4K pass | **Topaz via fal** (Starlight Precise) or **Magnific Creative** | Real detail rebuild + temporal coherence. Accept overnight render for a handful of shots. |
| **Cinema 1080p** | **SeedVR2** (comfortable in 32 GB, ~4x faster than 4K) or **FlashVSR Full** | **SeedVR2 on fal** (~$3/min 1080p) | 1080p keeps diffusion in VRAM with no tiling, fast enough to iterate. |
| **Short-drama vertical, volume** | **FlashVSR Tiny** (one-step streaming, temporal, fast) or **SeedVR2-3B FP8** | **WaveSpeed SeedVR2** (~$3/min 4K, cheapest verified) or **Topaz Astra** cloud | Throughput + temporal stability + lowest verified 4K price. |
| **Cheap batch / previews** | **Real-ESRGAN Compact** or **Real-CUGAN** via TensorRT **+ `vs_temporalfix`** | **fal RealESRGAN** (~$2.4/min 1080p) | Near real-time and almost free, but temporally naive: the temporal patch is mandatory on AI footage. |
| **Face-safe, zero hallucination** | **Topaz Proteus (Manual)**, gentle sliders | **Topaz via fal**, Proteus model | Hand-tuned, deterministic, will not invent a new face. Your prior field rule still holds here. |

**The short version.** For local cinema, **SeedVR2** is the quality engine and **FlashVSR**
is the speed/temporal engine, both open and both fitting the 5090. **Topaz** is the best
turnkey option: Proteus Manual for face-safe cleanup, and **Astra** (their cloud product,
explicitly tuned for Sora/Veo/Kling/Seedance) for GenAI-native volume. For 4K at volume the
RTX 5090 cannot keep up with diffusion (see speed section), so cloud earns its place there
while the local card handles 1080p diffusion and the occasional 4K hero shot overnight.

---

## 3. Local, open-weights tools

### SeedVR2 (ByteDance) — the local quality leader

The user's half-remembered "SideVR" is this: **SeedVR2 by ByteDance** (the Seed team), not
Alibaba. One-step diffusion video restoration, Apache-2.0, which makes it clean for a public
benchmark and for commercial use.

- **Repo / weights:** [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR),
  [SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B),
  [SeedVR2-7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B). ComfyUI node:
  [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
  (v2.5.24, Dec 2025), with FP16 / FP8 / GGUF quants and Blackwell SageAttention 3 support.
- **Academic standing:** SeedVR2 accepted to ICLR 2026; SeedVR was a CVPR 2025 Highlight.
  Paper [arXiv 2506.05301](https://arxiv.org/html/2506.05301v1).
- **Fits the 5090:** yes. 7B FP16 runs on 24 GB+, so 32 GB has headroom; the node author
  recommends FP16 for high-end cards like the 5090. BlockSwap + VAE tiling reach 4K.
- **Speed (now MEASURED on the reference 5090, v0.2.1):** the surprise is that **VAE-decode
  batching dominates, not model size or resolution**. 3B fp8 1080p batch 9 = **0.11 fps at ~31 GB**
  (near the ceiling, VAE decode ~73s of 85s), but 3B fp8 4K batch 1 with tiling + offload =
  **0.13 fps at ~9 GB**, and 7B fp8 1080p batch 5 + offload = **0.35 fps at ~25 GB**. Batch 9 at 4K
  **OOMs**. The efficient path is small batch + VAE tiling + block-swap (`--dit_offload_device cpu`),
  not large batches. Still minutes of compute per second of footage: overnight for volume, not
  real-time. Full data in [data/measurements-5090.json](../data/measurements-5090.json).
- **Synthetic-input verdict:** highest detail rebuild *and* it was benchmarked on AIGC28, an
  AI-generated test set, beating STAR/VEnhancer/MGLD-VSR there. But the over-sharpen warning
  above is real. Use the 7B at low strength for cinema faces, 3B FP8 for volume.
- **Pareto:** quality-tier (7B); the 3B FP8 path leans balanced.

### FlashVSR — the only open tool on the balanced frontier

First one-step *streaming* diffusion VSR: sliding-window so memory is bounded regardless of
clip length, distilled from a Wan2.1 text-to-video base. This is the bridge between diffusion
quality and GAN speed, and it is the standout open tool for your dual use-case.

- **Paper [arXiv 2510.12747](https://arxiv.org/abs/2510.12747)** (Oct 2025). ComfyUI:
  [1038lab/ComfyUI-FlashVSR](https://github.com/1038lab/ComfyUI-FlashVSR) (GPL-3.0) and the
  [Ultra_Fast fork](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) with explicit
  RTX 50-series support and no custom-kernel compile.
- **Speed: now MEASURED on the 5090 (FlashVSR_plus build), and it disappoints.** Tiny x2
  (720p to 1440p, 25-frame clip) = **0.80 fps at ~31 GB** (near the ceiling). x4 (4K-class) is
  **blocked on 32 GB**: OOM without tiling, and the tiled-dit / full paths hit a repo bug
  (`UnboundLocalError: temp_name`). The marketed "~17 fps on A100" does not translate to anything
  near real-time here. Note it needs >=21 input frames (streaming window). Data in
  [data/measurements-5090.json](../data/measurements-5090.json).
- **Modes:** Full (quality, tile for 4K) / Tiny (fast) / Tiny-Long (low-VRAM, long clips).
- **Synthetic-input verdict:** its prior is literally the text-to-video domain, and the
  streaming design fights AI flicker. More conservative than SeedVR2 (preserves motion blur,
  invents less), which is the safer choice when you fear hallucination and the weaker one
  when the source genuinely lacks detail to rebuild.
- **Pareto:** balanced. The only diffusion tool here that earns it.
- **Affiliation note:** sources disagree on who owns FlashVSR. It is from **OpenImagingLab /
  Shanghai AI Lab**, built on Alibaba's Wan2.1 base; one scout mislabeled it pure Alibaba.
  Flagged below.

### The classic diffusion trio (quality-tier, each gated)

- **VEnhancer** (Vchitect / Shanghai AI Lab, [repo](https://github.com/Vchitect/VEnhancer)):
  true space-time enhancement (spatial SR + frame interpolation + refinement, prompt-aware),
  purpose-built for AI video. Gated by hardware: the repo states "A100 80G required" and caps
  output at ~2K, so it does **not** fit the 5090 for a 4K target. License **Apache-2.0**
  (commercial use explicitly permitted, verification-corrected from "unverified").
- **Upscale-A-Video** (NTU S-Lab, [repo](https://github.com/sczhou/Upscale-A-Video)): strong
  flow-guided temporal stability, explicitly good on AI-generated video. **License is NTU
  S-Lab 1.0, non-commercial**, which blocks the short-drama studio. Auteur/festival use only.
- **MGLD-VSR** ([repo](https://github.com/IanYeung/MGLD-VSR)): motion-guided latent diffusion,
  best-in-class anti-flicker, conservative detail (good for AI input). License unverified
  (built on Stable Diffusion code, likely OpenRAIL constraints).

### GAN / transformer chains (cheap-fast, temporally naive)

Run frame-by-frame through ComfyUI, VapourSynth (`vs-mlrt` + TensorRT) or chaiNNer. Fast and
nearly free, but no temporal model: any flicker in the source is preserved or amplified. On
AI footage, pair with the temporal patch.

| Tool | License | Note on AI-gen 720p |
|---|---|---|
| **Real-ESRGAN** (`realesr-general`, `animevideov3`, `x4plus`) | BSD-3 | Real-time at 1080p via TensorRT. Sharpens waxy skin, amplifies flicker. Low-to-medium. |
| **Real-CUGAN** (bilibili) | MIT | Excellent on anime/2D/stylized, wrong tool for photoreal humans. |
| **RealViformer** | MIT | Recurrent transformer, some temporal propagation, low-artifact, conservative. Medium. |
| **RealBasicVSR** | Apache-2.0 | Recurrent, calms flicker and grain, but softens detail. Medium. |
| **4x-UltraSharp / 4x-AnimeSharp** | **CC-BY-NC-SA (non-commercial)** | Popular but a license trap for the studio. Trained on JPEG restoration, over-textures smooth AI gradients. Low. |
| **4xNomos2** | CC-BY-4.0 (commercial OK) | Photographic, still per-frame. Low. |

**Temporal patch:** [`pifroggi/vs_temporalfix`](https://github.com/pifroggi/vs_temporalfix),
a VapourSynth filter that adds temporal coherence to AI upscales. This is the standard way to
make a GAN chain watchable on synthetic footage. License unverified.

### Also on the table (added in verification)

- **NVIDIA RTX Video Super Resolution (RTX VSR)** as a ComfyUI node / Python wheel, shipped
  Feb 2026, runs natively on Blackwell (your 5090), markets the exact 720p to 4K AI-video case,
  NVFP4 path cuts VRAM ~60%, free. A per-frame-class consumer SR worth testing first for the
  cheap-fast 4K corner ([RTX VSR node](https://www.runcomfy.com/comfyui-nodes/Nvidia_RTX_Nodes_ComfyUI/rtx-video-super-resolution)).
  Verify per-frame quality on synthetic faces before trusting it.
- **Open frame interpolation (FPS):** **RIFE / Practical-RIFE** is the free workhorse and
  **GIMM-VFI** the 2025 quality leader, both open and ComfyUI-friendly, a better-documented path
  than Topaz's Chronos/Apollo/Aion (the "Aion = 2026 flagship" label is third-party-sourced, not
  vendor-confirmed). On synthetic footage, interpolate *before* heavy upscaling, at low ratios,
  or warped frames produce morph artifacts.
- **SUPIR, frame-by-frame:** the heavyweight image upscaler (see the
  [image page](./image-upscalers.md)) can run per-frame on video, but it has no temporal lock and
  flickers, so it is a stills/hero-frame tool, not a sequence tool.
- **Baseline floor:** as a benchmark control, a plain **Lanczos/bicubic + light sharpen** (ffmpeg)
  is the do-nothing reference every comparison should include.

### Watch list (too new to recommend)

Stream-DiffVSR ([arXiv 2512.23709](https://arxiv.org/abs/2512.23709), claims 0.328 s/frame at
720p on a 4090) and InstaVSR are the trajectory the field is on (auto-regressive / lightweight
diffusion VSR), but lack mature ComfyUI integration and verified runtime. Track, do not deploy.

---

## 4. Topaz (turnkey, proprietary)

As of mid-2026 the desktop app is **Topaz Video** (rebranded from "Video AI", current build
~1.6.1, [version history](https://www.videohelp.com/software/Topaz-Video-AI/version-history)),
**subscription-only** since Oct 2025. Three surfaces that must not be conflated:

1. **Topaz Video (desktop):** local CUDA rendering on your 5090, unlimited, plus 25 cloud
   credits/mo. Pricing: Personal $299/yr, Pro $699/yr
   ([pricing](https://www.topazlabs.com/pricing)).
2. **Astra ([astra.app](https://www.topazlabs.com/astra)):** cloud-only, explicitly tuned for
   GenAI footage (Sora, Veo, Kling, Seedance, Runway, Midjourney), with a Creative/Precision
   slider and automatic per-scene detection. This is the surface Topaz points AI-video users
   to. Max 4K.
3. **API / third-party cloud:** fal.ai, Replicate, WaveSpeed, plus a first-party Topaz Cloud
   API and a Premiere integration (Expansion release, May 2026).

**Models that matter for synthetic input:**

- **Proteus (Manual):** classic, non-diffusion, hand-tuned. Your face-safe workflow. It beats
  the older Starlight on faces without hallucinating, as long as you keep sliders gentle. Push
  hard and you get halos and plastic skin. Fast (~45-61 fps at 1080p, scaling down to ~5-7 fps
  at 4x to 4K, see speed section).
- **Starlight (diffusion):** local variants Mini, Sharp, Fast 2, HQ, and **Precise 2.5**.
  Precise 2.5 is the first Topaz diffusion video model with full temporal consistency and is
  explicitly built to de-plastic synthetic faces and skin. This **supersedes your old "avoid
  Starlight on faces" rule** for this specific model. Verification correction: Precise 2.5
  **does run locally** on the 5090 (Windows + NVIDIA, 12 GB VRAM min, 16-24 recommended), and
  Topaz's own docs cite it "maxing out at roughly 12 fps on a top-tier RTX 5090" for 1080p to 4K
  ([Precise 2.5 model page](https://www.topazlabs.com/models/starlight-precise-2-5-video)). Note
  that figure sits oddly against the ~0.2 fps an earlier community thread reported for Starlight
  Mini, so treat both as needing an on-device re-measure.
- **Astra 2 (cloud):** confirmed real ([topazlabs.com/models/astra-2](https://www.topazlabs.com/models/astra-2),
  April 2026 Next-Gen release). The Creative-mode generative video upscaler, adds a prompt field
  plus Creativity (1-5) and Sharpness (1-5) sliders, positioned for AI footage from Seedance 2.0,
  Kling and Runway. The names in your brief were right.
- **Gaia 2 / Gaia CG:** best classic pick for stylized/animated AI output, and half-price on
  fal.

**Pricing and AI-gen suitability:** desktop is the only path giving you local, unlimited,
offline runs (Precise 2.5 for quality, Proteus Manual for face-safe). Astra is the highest-fit
surface for pure synthetic volume, scene-aware, no local 0.2-fps wall, but cloud-only and
capped at 4K. Topaz is fully proprietary, so reproducibility means pinning the app version and
model, not inspecting weights.

---

## 5. The Chinese / Asian ecosystem, disentangled

Forums conflate three different things. For a public benchmark, get this right:

- **Dedicated VSR that takes any video in:** the strongest is **ByteDance SeedVR2** (above).
  ByteDance, not Alibaba. This is the misattribution to fix.
- **Generation models that refine/upscale their *own* output:** Alibaba **Wan 2.2** (low-noise
  expert used as a low-denoise refine pass; the popular "Wan 2.2 AIO Upscale" is a *community*
  ComfyUI workflow pairing Wan refine with 4x-UltraSharp, not a first-party Alibaba upscaler),
  Tencent **HunyuanVideo 1.5** (a built-in cascaded 720p→1080p SR stage; licensed under the
  Tencent Hunyuan Community License, not Apache, so it carries acceptable-use and large-user
  restrictions), Kuaishou **Kling**
  (built-in HD upscale; Kling 3.0 generates native 4K). All three shine on their *own*
  generations and impose their own look on foreign footage.
- **The GAN root:** Tencent ARC's **Real-ESRGAN** and **GFPGAN**, the genealogy of half the
  "AI upscale" buttons in consumer apps.

Two corrections for the record:
- **Adobe VideoGigaGAN is Adobe Research, never shipped.** Not Alibaba, not in Premiere. A
  tech demo ([Adobe Research](https://research.adobe.com/publication/videogigagan-towards-detail-rich-video-super-resolution/)).
- **RealBasicVSR / BasicVSR++ and VEnhancer are NTU S-Lab and Shanghai AI Lab**, not Alibaba
  DAMO. No first-party Alibaba DAMO general-purpose VSR model could be confirmed.

---

## 6. Hosted-API options and price

Two billing models, very different break-even. **Per-second-by-output-resolution** (Topaz on
fal/Replicate, WaveSpeed) is flat regardless of frame count. **Per-megapixel** (fal SeedVR2,
fal RealESRGAN, where MP = width x height x frames) scales ~4x going from 1080p to 4K.

`$/min of 4K` is for a 5s 720p→4K@24fps clip extrapolated to 60s. Verified prices only.

| Service | Model | Billing | $/min 4K | $/min 1080p | AI-gen fit | Notes |
|---|---|---|---|---|---|---|
| **WaveSpeed** | SeedVR2 | $/s by res, 5s min | **$3.00** | $1.80 | high | Cheapest verified 4K. Up to 10 min/clip. Ownership/license not quoted. |
| **fal** / **Replicate** | Topaz | $0.08/s >1080p, x2@60fps, Gaia2 half | **$4.80** ($2.40 Gaia2) | $1.20 | high | Topaz house quality, watermark-free. Variant list includes Starlight Precise 1/2/2.5. |
| **fal** | RealESRGAN | $0.0008/MP | ~$9.6 | ~$2.4 | low | Per-frame GAN, no temporal. Previews only. |
| **fal** | SeedVR2 | $0.001/MP | ~$12 | ~$3 | high | Per-MP makes 4K expensive; great value at 1080p. |
| **Topaz Cloud API** | Starlight/Astra | credits ($0.08-0.12) | unverified (video) | unverified | high | First-party, opaque per-video pricing. |
| **Krea** | 7 models incl. Topaz | plan units | unverified | unverified | med-high | You already have access; derive cost from your own account. |
| **Magnific / Freepik** | Creative / Precision | credits (100-500+/clip) | unverified | unverified | high (Creative) | Most "hallucinatory", best at hiding AI mush, identity-drift risk. |
| **Runway / Luma** | native upscale | credits | ~$1.2 (Runway, unverified) | n/a | medium | Designed for their *own* clips; external 720p upload may be unsupported. |

**Verified 4K price ranking (cheapest first):** WaveSpeed SeedVR2 **$3.00** < Topaz via
fal/Replicate **$4.80** (or **$2.40** with Gaia 2) < fal RealESRGAN ~$9.6 < fal SeedVR2 ~$12.

---

## 7. Speed and the 5090 reality

Upscaling is three different workloads on the same card, and they bottleneck on different
things.

**GAN/CNN: fast at 1080p, a batch job at 4K.** Topaz classic models on a 5090, verified from
[community benchmarks](https://community.topazlabs.com/t/video-ai-7-0-x-user-benchmarking-results/91056)
(1080p input): Proteus 57-61 fps at 1x, but **4.7-7.0 fps at 4x**. The "4x the pixels" tax is
real: 720p→4K is ~9x the pixels of 720p, ~4x the pixels of 1080p, and per-frame time tracks
output pixel count. At 4x, CPU/RAM/IO co-limit the GPU (two 5090 users differ 4.4 vs 6.9 fps
on the same model).

**Measured here (reference 5090, v0.2.0):** open ESRGAN-family models in ComfyUI eager mode,
full-frame x4 of 720p (output 5120x2880, torch 2.7+cu128): RealESRGAN x4plus and 4x-UltraSharp =
**1.71 fps fp16, 0.88 fps fp32** (peak VRAM 7.7 / 15.4 GB); the lighter anime_6B = **5.3 fps fp16**.
**Topaz classic Proteus** (local tvai ffmpeg CLI, steady-state) = **15 fps at 1080p, 7.3 fps at 4K**,
peak VRAM only **~1.8-2.7 GB**, confirming the community ~4.7-7 fps@4x. The classic tier is
featherweight and near-interactive; the diffusion tier below is 20-100x slower (Topaz Starlight
diffusion needs Neuroserver, not measurable via this CLI). Per-frame GAN has no temporal model, so
add `vs_temporalfix` for video. Data in
[data/measurements-5090.json](../data/measurements-5090.json).

**Diffusion VSR: not real-time, by one to two orders of magnitude.**

- Topaz Starlight Mini (local): **~0.2 fps** at 4x, Sharp ~1.8 fps
  ([forum](https://community.topazlabs.com/t/low-fps-on-rtx-5090-with-starlight-mini/98436)).
  A 15-min source reported at ~36 hours (an early build). Overnight, hero-shot tool only. This
  conflicts sharply with Topaz's own ~12 fps claim for the heavier Precise 2.5 at 1080p to 4K:
  the two figures are an order of magnitude apart in the "wrong" direction, so both need an
  on-device re-measure before either is trusted.
- SeedVR2 7B: minutes per second of 4K (no clean 5090 number; 2080 Ti GGUF anchor ~6-8 min per
  minute of 1080p).
- FlashVSR (FlashVSR_plus): **measured** tiny x2 (720p to 1440p) = 0.80 fps near the 32 GB ceiling;
  x4/4K **blocked** on 32 GB (OOM + repo bugs). The A100 ~17 fps figure does not hold here.

**VRAM:** 32 GB is generous for upscaling (unlike generation). What forces tiling is output
resolution and clip length, not weights. The OOM triad is long clip + high res + big temporal
batch. The 96 GB system RAM is a genuine asset for BlockSwap/offload. **FP4 is stranded:** the
silicon does 3,352 TOPS FP4 but ComfyUI's FP4 loaders were
[still broken in Jan 2026](https://github.com/Comfy-Org/ComfyUI/issues/11864) (upcast to fp16 →
OOM). Plan throughput on FP8/fp16.

**Hidden IO wall:** diffusion-VSR tools often want PNG/EXR frame sequences. 4K 16-bit PNG is
~15-40 MB/frame, and you pay it three times (write, read for upscaler, read for encoder). A
fast 1080p GAN pass writing 4K frames can saturate your 1400 MB/s NVMe (a SATA/older-NVMe tier,
not Gen4/5). Mitigation: pipe frames in-memory (ComfyUI VHS / ffmpeg stdin-stdout), keep scratch
on its own fast volume, reserve EXR for genuine HDR/linear work.

**Planning rule:** classic GAN 4x→4K ≈ 5-7 fps (a few times slower than real-time); local
diffusion VSR→4K ≈ 0.2-2 fps (overnight per 10-15 min). 4K diffusion locally = hero shots only;
volume 4K must use a GAN model or the cloud.

---

## 8. Local vs API break-even

At your volume (30 min to 3 h of upscaled video per month):

- **Classic / GAN upscaling:** local desktop wins hard. Unlimited local rendering on the 5090
  versus, say, fal RealESRGAN at ~$2.4/min of 1080p ($72-432/mo at your volume) or Topaz on
  fal at $1.20/min 1080p. Local is effectively electricity only.
- **Diffusion 4K:** the calculus flips. Local SeedVR2/Starlight is "free" but 0.2-2 fps makes
  3 h of 4K diffusion physically impossible in a month on one 5090. The cheapest verified cloud
  4K is WaveSpeed SeedVR2 at $3/min ($90-540/mo for 30 min-3 h). So the rational split is: local
  card does 1080p diffusion and overnight 4K hero shots, cloud does 4K at volume.
- **SeedVR2 is open**, so any time the API path is SeedVR2-based, self-hosting on the 5090
  dominates the economics for anything you can afford to render overnight.

---

## 9. How this should be benchmarked

The full protocol lives in [methodology.md](./methodology.md). The headline, because it is
what separates this from a YouTube comparison:

- **Full-reference metrics (PSNR/SSIM/LPIPS) mostly do not apply.** With no ground-truth 4K,
  they can only measure fidelity to the flawed 720p source, where a do-nothing bicubic upscale
  "wins". Use them as a drift anchor, never as a quality score.
- **No-reference metrics are the primary quality signal but have a fatal failure mode.** MUSIQ
  correlates *negatively* with human hallucination judgments
  ([arXiv 2507.14367](https://arxiv.org/pdf/2507.14367)): it rewards confidently-wrong invented
  detail. Never rank on it alone.
- **Score a triad per clip:** (A) no-reference quality (MANIQA, CLIP-IQA, Q-Align, DOVER for
  video), (B) fidelity and hallucination anti-cheat (VMAF vs VMAF-NEG gap, DINO-HS hallucination
  proxy, ArcFace identity drift on faces), (C) temporal consistency (VBench flicker/consistency,
  RAFT warping error). A tool that tops quality but fails the anti-cheat is hallucinating: flag
  it, do not crown it.

---

## 10. Contradictions resolved by the verification pass

- **"Astra 2" is real.** Confirmed product page ([astra-2](https://www.topazlabs.com/models/astra-2)),
  April 2026 Next-Gen release, a Creative-mode generative video upscaler. The earlier "no version
  string found" reading was a search miss. Your original name was correct.
- **Starlight Precise 2.5 runs locally** on the 5090 (not cloud-only), and its ~12 fps 1080p-to-4K
  figure is vendor-corroborated, not single-blog. So the local quality option is Precise 2.5, not
  just Mini. The ~12 fps vs ~0.2 fps Mini gap is now the open question (flagged in the speed
  section), pending on-device measurement.
- **VEnhancer is Apache-2.0**, commercial use permitted (was wrongly flagged "license unverified").
- **FlashVSR** is **OpenImagingLab / Shanghai AI Lab**, distilled from **Wan2.1-T2V-1.3B** (not a
  "3B" base). ComfyUI wrappers are GPL-3.0; the Wan2.1 base weights carry their own terms, so the
  commercial-redistribution license is the one thing left to confirm before billing clients.
- **HunyuanVideo 1.5** is under the **Tencent Hunyuan Community License**, not Apache-2.0.

Still genuinely open: **Topaz Astra / Studio pricing** (third-party figures conflict, re-pull from
topazlabs.com), and **every 5090 diffusion fps** except the two community Starlight figures.

---

## 11. Gaps (what to measure next)

- **No RTX 5090 figures** for SeedVR2, FlashVSR, Starlight Precise 2.5 at 4K. Every diffusion
  number here is A100, H100 or 4090. A logged ComfyUI run on the reference 5090 (sec/frame, peak
  VRAM, at 720p→1080p and 720p→4K) is the single biggest missing piece, and exactly what this
  repo should fill.
- **Licenses to confirm before commercial use:** VEnhancer, MGLD-VSR, `vs_temporalfix`, FlashVSR
  weights, Wan 2.6/2.7.
- **API unknowns:** max clip length / resolution on fal and Replicate endpoints; Topaz first-party
  video pricing; Krea and Magnific per-clip cost; whether Runway/Luma upscalers accept external
  (non-native) uploads.
- **DAM-VSR** ([arXiv 2507.01012](https://arxiv.org/pdf/2507.01012)) claims SOTA on real + AIGC
  data; affiliation and code license unconfirmed. Worth a look.

---

## Sources

Topaz: [pricing](https://www.topazlabs.com/pricing) · [Astra](https://www.topazlabs.com/astra) ·
[Starlight docs](https://docs.topazlabs.com/topaz-video/project-starlight-series) ·
[5090 Starlight thread](https://community.topazlabs.com/t/low-fps-on-rtx-5090-with-starlight-mini/98436) ·
[community 5090 benchmarks](https://community.topazlabs.com/t/video-ai-7-0-x-user-benchmarking-results/91056) ·
[Expansion release](https://www.topazlabs.com/news/the-expansion-release).
SeedVR2: [repo](https://github.com/ByteDance-Seed/SeedVR) ·
[7B card / over-sharpen warning](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) ·
[paper](https://arxiv.org/html/2506.05301v1) ·
[ComfyUI node](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) ·
[4090/H100 benchmarks](https://www.weirdwonderfulai.art/resources/seedvr2-upscaler-for-video-and-images/).
FlashVSR: [paper](https://arxiv.org/abs/2510.12747) ·
[ComfyUI node](https://github.com/1038lab/ComfyUI-FlashVSR).
Others: [VEnhancer](https://github.com/Vchitect/VEnhancer) ·
[Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video) ·
[MGLD-VSR](https://github.com/IanYeung/MGLD-VSR) ·
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) ·
[vs_temporalfix](https://github.com/pifroggi/vs_temporalfix) ·
[OpenModelDB](https://openmodeldb.info/).
APIs: [fal Topaz](https://fal.ai/models/fal-ai/topaz/upscale/video) ·
[fal SeedVR2](https://fal.ai/models/fal-ai/seedvr/upscale/video/api) ·
[Replicate Topaz](https://replicate.com/topazlabs/video-upscale) ·
[WaveSpeed SeedVR2](https://wavespeed.ai/models/wavespeed-ai/seedvr2/video) ·
[Krea pricing](https://www.krea.ai/pricing) ·
[Magnific video](https://www.magnific.com/magnific-video-upscaler).
Hardware: [RTX 5090 specs](https://www.spheron.network/blog/nvidia-rtx-5090-specs/) ·
[FP4 loader bug](https://github.com/Comfy-Org/ComfyUI/issues/11864).
Method: [Hallucination Score](https://arxiv.org/pdf/2507.14367) ·
[VMAF](https://github.com/Netflix/vmaf) · [VBench](https://github.com/Vchitect/VBench) ·
[IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) · [DOVER](https://github.com/VQAssessment/DOVER).

*Last updated 2026-06-16. Research pass only (no adversarial second pass, no on-device 5090
measurement yet).*
