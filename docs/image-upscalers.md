# Image upscalers for AI-generated stills and frames (June 2026)

Scope: upscaling still images, and individual frames, that came out of an AI generator at
roughly 720p, pushed to 1080p or 4K. Same two use-cases as the video page: cinema-grade
quality on one side, commercial-volume throughput on the other. Same three axes: quality,
cost, speed. Reference machine for local claims: RTX 5090 (32 GB), Ryzen 9 9950X, 96 GB RAM,
8 TB NVMe.

> **Read this first if your real input is video.** None of the still-image tools here
> temporally stabilize. Run them frame by frame and fine detail will shimmer and crawl from
> frame to frame. Topaz states the principle plainly, and it is the exact reason their *video*
> product exists. These image tools are for hero stills, posters, key art, storyboards, and
> the AI-generation step *before* animation. For moving sequences, use the
> [video page](./video-upscalers.md).

> **Verification status (v0.1.1).** Research pass plus an adversarial verification pass have both
> run. Corrections are folded in (one fabricated Magnific benchmark stat was struck, several
> licenses were corrected). Hard numbers are still single-source unless a link is given, and
> nothing was measured on the reference 5090 yet beyond one TensorRT figure. On-device 5090
> measurement is the next milestone.

---

## 1. The trilemma, and the over-sharpening trap

For AI-generated 720p there is no ground-truth 4K original, so you are not "restoring" an
image, you are inventing detail for one that was already invented. Three goals pull against
each other and cannot be collapsed into one score:

- **Fidelity** (stay true to the source): measured by PSNR/SSIM, but these reward the
  *blurriest* output and need a reference you do not have.
- **Perceptual realism** ("looks like a real photo"): measured by LPIPS/DISTS and the
  no-reference metrics, but these are gameable by sharpening.
- **Generative detail** (invent plausible texture): what diffusion upscalers do best, and
  exactly what destroys identity, text and structure when pushed.

The trap that decides the whole field: the popular no-reference metrics (NIQE, MUSIQ,
CLIP-IQA+) **systematically over-reward over-sharpened output**
([arXiv 2504.18524](https://arxiv.org/pdf/2504.18524)). Optimizing toward MUSIQ produces
"structured artifacts similar to an adversarial attack". So the practical rule for choosing an
image upscaler for synthetic input is not "which looks sharpest", it is "which adds plausible
detail without inventing a new face, mangling text, or crunching the existing AI texture into
hard-edged AI texture". The fidelity-vs-creativity knob (denoise strength, control scale, CFG)
is the real control surface, per tool.

---

## 2. Recommendations by use-case (the Pareto answer)

| Use-case | Best LOCAL (5090) | Best API / cloud | Why |
|---|---|---|---|
| **Cinema hero still, quality-max** | **SUPIR** (v0F model, low s_churn), or **Flux-tile** (jasperai) | **Magnific Precision**, **Topaz Gigapixel API**, or **Replicate SUPIR** | Best photoreal detail rebuild on a single soft AI still. Both SUPIR and Flux-dev are non-commercial: cinema/auteur use, not the studio. |
| **Faithful, zero new hallucination** | **ControlNet Tile (SDXL)** at low denoise, or **HAT / DRCT** (commercial-clean) | **Topaz Gigapixel API** ($8-12/100), **Recraft Crisp** ($0.40/100), **Ideogram** ($6/100) | Adds resolution without inventing identity. Tile at denoise 0.2-0.4 cleans AI artifacts without re-drawing. |
| **Per-frame video (image tool on a sequence)** | **CCSR-v2** (seed-stable, less flicker) or one-step **OSEDiff/AdcSR** | **fal SeedVR2 image** ($0.83/100) | If you must use an image tool on frames, pick the most temporally stable. Better: use a real video tool. |
| **Cheap batch stills** | **Real-ESRGAN / Real-CUGAN** via TensorRT (12.7 fps verified on 5090), commercial-clean models only | **fal SeedVR2** ($0.83/100) or **Recraft Crisp** ($0.40/100) | Sub-second, nearly free locally. Avoid the non-commercial community models (UltraSharp). |
| **Anime / stylized** | **Real-CUGAN** (conservative mode) | **Ideogram** (style-preserving) | Built for line art, tunable denoise, will not photoreal-ify a 2D look. |

**The short version.** For a cinema still, **SUPIR** is the local detail king (mind the
non-commercial license) and **Magnific Precision** or **Topaz Gigapixel** are the turnkey
equivalents. For anything commercial in the short-drama studio, stay on **commercial-clean**
tools (Real-ESRGAN BSD, SwinIR/HAT Apache, DRCT MIT, SeeSR Apache, AuraSR v2 Apache, Topaz,
Recraft, Krea) and avoid the license traps (SUPIR, Flux-dev, UltraSharp, InvSR, StableSR are
all non-commercial). The cheapest sane-quality hosted path by a wide margin is **SeedVR2 on
fal** at about $0.83 per 100 4K stills.

---

## 3. Local, diffusion-based (the quality tier)

These hallucinate detail from a generative prior, which on synthetic input can replace AI mush
with believable texture (great for one hero still) or invent a different face and flicker across
frames (bad for sequences). The whole game is the creativity knob.

### SUPIR — the heavyweight

The reference for max-detail photoreal restoration. Repo
[Fanghua-Yu/SUPIR](https://github.com/Fanghua-Yu/SUPIR), practical front-end
[kijai/ComfyUI-SUPIR](https://github.com/kijai/ComfyUI-SUPIR). SDXL-based.

- **License: non-commercial.** The code README adds a strict non-commercial restriction that
  overrides the nominal MIT ([license conflict #51](https://github.com/Fanghua-Yu/SUPIR/issues/51));
  the kijai wrapper confirms "non-commercial use only". A real blocker for the studio. Cinema
  case only, or license it.
- **Fits the 5090** comfortably (up to ~3072x3072 on 24 GB without fp8; fp8 U-Net + tiled VAE
  for more). Needs PyTorch built for Blackwell (cu128).
- **Speed:** slow, minutes per 4K still (many EDM diffusion steps). Community 4090 figures
  (~111 s at 2x, ~606 s at 4x) circulate but could not be anchored to a primary source.
- **Models:** v0Q (max quality), v0F (light degradation, hallucinates less, the safer pick for
  AI frames).
- **Hallucination control:** v0F model, low s_churn, modest cfg, raise structure guidance, tight
  positive prompt. **High for single hero stills, low-to-medium for frame sequences** (no
  temporal lock, identity drift).

### Clarity Upscaler — the open Magnific

[philz1337x/clarity-upscaler](https://github.com/philz1337x/clarity-upscaler), AGPL-3.0,
SD1.5/SDXL + ControlNet Tile + tiled diffusion. Much faster than SUPIR. The `creativity` and
`resemblance` sliders are the explicit fidelity dial. Low creativity + high resemblance + tile
strength under 0.7 keeps it faithful; push creativity and it stylizes, which can read as *more*
synthetic on AI input. License caveat: AGPL plus the base checkpoint license governs output.
**Medium** on AI-gen, controllable and fast.

### CCSR-v2 — stability-first

[csslc/CCSR](https://github.com/csslc/CCSR), TIP 2026. Engineered for content consistency and
**stability across noise seeds**, where SUPIR/StableSR/SeeSR are unstable. That seed-stability is
the single most useful property for *frame sequences*: it cleans without aggressively
re-inventing, so it flickers less. **Medium-high** for AI frames. License: verify (SD-based).

### ControlNet Tile (SDXL and Flux) — the most controllable fidelity dial

The workhorse "infinite resolution" path. Detail comes from the base model, structure is locked
by the tile control. At low denoise (0.2-0.4) and control strength under 0.7 it **cleans AI
artifacts without re-inventing**, the best behavior for synthetic input.

- **SDXL Tile + Ultimate SD Upscale:** trivial on the 5090, fast, cheap. License flexible
  (depends on the SDXL checkpoint you pick). Pareto: cheap-fast.
- **Flux tile** ([jasperai/Flux.1-dev-Controlnet-Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler)):
  cleanest, least-plastic skin of the tile options. FLUX.1-dev fits the 32 GB card in fp16
  (~24 GB) or fp8/GGUF for big tiles. **License: FLUX.1-dev is non-commercial**; commercial use
  needs a BFL license. Pareto: balanced to quality-tier.

### SeeSR / StableSR — the SD2-era foundations

- **SeeSR** ([cswry/SeeSR](https://github.com/cswry/SeeSR)): **Apache-2.0** (commercial-clean),
  semantics-aware (tag/prompt conditioning preserves object identity). A solid permissive
  baseline. Medium.
- **StableSR** ([IceClear/StableSR](https://github.com/IceClear/StableSR)): the foundational
  diffusion-prior SR, now surpassed by SUPIR/CCSR. **License: NTU S-Lab 1.0, non-commercial.**

### The one-step diffusion frontier (cheap-fast, the 2024-2026 wave)

These distill the diffusion prior into 1-5 steps, collapsing SUPIR's minutes into sub-second to
seconds per image, and the low step count means *less flicker* than multi-step SUPIR. This is the
throughput answer and the most promising path for per-frame work.

| Tool | Base | Steps | License | Note |
|---|---|---|---|---|
| **OSEDiff** ([repo](https://github.com/cswry/OSEDiff)) | SD2.1-base | 1 | Apache-2.0 code (weights caveat) | ~0.1 s/iter on A100, deployed commercially (OPPO). Fast, faithful, modest texture ceiling. |
| **AdcSR** ([repo](https://github.com/Guaishou74851/AdcSR), CVPR 2025) | distilled OSEDiff | 1 | **Apache-2.0** (code) | Up to 9.3x over previous one-step diffusion SR methods. The throughput champion. |
| **DiT4SR** ([repo](https://github.com/Adam-duan/DiT4SR), ICCV 2025) | SD3.5-medium | multi | **Pi-Lab 1.0** (commercial: verify) | Modern SD3.5 fidelity challenger to SUPIR. Weights released (`dit4sr_f`/`dit4sr_q`), no ComfyUI node yet (scripting). |
| **StrSR / FluxSR** (2025-26 papers) | DiT / FLUX | 1 | n/a | StrSR code out ([repo](https://github.com/jkwang28/StrSR)); FluxSR weights NOT released and FLUX.1-dev base is non-commercial. Watch. |
| **InvSR** ([repo](https://github.com/zsyOAOA/InvSR), CVPR 2025) | SD-Turbo | 1-5 | **NTU S-Lab (non-commercial)** | Pick start timestep = real creativity dial. Studio-blocked by license. |
| **SinSR** | ResShift distill | 1 | verify | Fast, quality below OSEDiff/AdcSR. |
| **FiDeSR**, **DiT one-step SR** (2026 papers) | various | 1 | n/a | High-fidelity one-step, but no released ComfyUI node confirmed yet. Watch H2 2026. |

---

## 4. Local, fast and deterministic (the cheap-fast tier)

Feed-forward CNN/GAN/transformer networks, one pass per frame, no sampling loop. Sub-second to
low-single-digit fps on the 5090, 1-4 GB VRAM, essentially free at your volume. They **cannot
invent** plausible new photoreal texture, which is both their virtue (faithful, deterministic,
reproducible, no hallucination on top of hallucination) and their limit (they sharpen the
existing waxy/plastic look into crunchy waxy/plastic). High suitability for synthetic line
art/anime/graphics, low-to-medium for synthetic photoreal.

| Tool | License | Best for AI-gen 720p | Note |
|---|---|---|---|
| **Real-ESRGAN** (`x4plus`, `anime_6B`, `animevideov3`, `general-x4v3`) | **BSD-3** (clean) | medium (anime: high) | The baseline. Crisp, oil-paint smear on fine texture. ncnn-vulkan build runs anywhere. |
| **Real-CUGAN** (bilibili) | **MIT** (LICENSE is plain MIT, commercial OK) | high (anime), low (photo) | Best anime line preservation, tunable denoise + conservative mode. |
| **SPAN / DAT** (community) | varies (read each) | medium | SPAN = the speed king, DAT = current community quality favorite. Named ESRGAN successors r/StableDiffusion expects. |
| **SwinIR** | **Apache-2.0** (clean) | medium | Gentle, faithful, less crunchy than ESRGAN, slower (~1 s+/frame). |
| **HAT** (`Real_HAT_GAN_SRx4`) | **Apache-2.0** (clean) | medium | Highest fidelity of the deterministic family, slowest, tile for 4K. |
| **DRCT** | **MIT** (clean) | medium | HAT-class fidelity, more efficient. Most public weights are classical (bicubic), verify a real-world variant. |
| **Aura-SR v2** ([fal](https://huggingface.co/fal/AuraSR-v2)) | **Apache-2.0** (v2 weights) | **high** | GigaGAN, the one deterministic tool *built for generated images*. Use the overlapped method to avoid tile seams (~2x time). Can over-invent on faces. A 2024 release, not 2026. |
| **Community ESRGAN** (4x-UltraSharp, Siax, AnimeSharp, Nomos) | **heterogeneous, read each** | high (clean/anime), low-med (photo) | The most tunable corner. **4x-UltraSharp and 4x-AnimeSharp are CC-BY-NC-SA, non-commercial.** Run via TensorRT for 5090 throughput. |

**Measured on the reference RTX 5090** (v0.2.0, full data in
[data/measurements-5090.json](../data/measurements-5090.json)): full-frame eager-mode x4 of a 720p
frame (output 5120x2880, torch 2.7+cu128, driver 596.21), **RealESRGAN x4plus and 4x-UltraSharp run
at 1.71 fps fp16 / 0.88 fps fp32**, peak VRAM **7.7 GB fp16 / 15.4 GB fp32**; the lighter anime_6B
hits **5.3 fps fp16**. fp16 is a free 2x speed-and-VRAM win. (A third-party TensorRT figure of
12.7 fps is at a much smaller output, 512 to 2048, so not comparable to this 4K-class full-frame
number.) These are throughput/VRAM figures on synthetic input; quality on real AI footage is a
separate axis.

**Runners and the free Blackwell path (added in verification):**
- **NVIDIA RTX Video Super Resolution (RTX VSR)** ships as a ComfyUI node (Feb 2026), runs
  natively on your Blackwell 5090, targets exactly 720p to 4K, NVFP4 cuts VRAM ~60%, free. Test it
  first for the cheap-fast 4K corner; verify quality on synthetic faces
  ([node](https://www.runcomfy.com/comfyui-nodes/Nvidia_RTX_Nodes_ComfyUI/rtx-video-super-resolution)).
- **Upscayl** (AGPL-3.0, [tool](https://www.tooljunction.io/ai-tools/upscayl)) is the turnkey
  desktop wrapper of the OpenModelDB/ESRGAN corpus (ncnn-Vulkan). **chaiNNer**
  ([repo](https://github.com/chaiNNer-org/chaiNNer)) is the node-based batch runner and the natural
  reproducibility backbone (pin a model, pin a chain) for this benchmark.

---

## 5. Topaz stills

As of mid-2026 the line is subscription-only, "Photo AI" is now **Topaz Photo**, and there is a
cloud platform (**Topaz Image / Image Web + API**) and a bundle (**Topaz Studio**). Models split
into **non-generative** (Standard, High Fidelity 2/3, Low Res 2, Text & Shapes, Art & CGI) and
**generative** (Redefine, Recover 2/3, Bloom, Wonder, Standard Max, Face Recovery; note Standard
Max is a *generative* model per Topaz docs, despite the name).

- **Gigapixel:** best-in-class *non-generative* detail. On AI stills, the non-generative models
  firm up soft 720p without inventing identity. The generative models (Redefine, Bloom) hallucinate
  ("Bloom re-opened a subject's closed eyes at minimum creativity"), so do not use them on
  sequences. Pricing: Personal $149/yr, Pro $499/yr.
- **Photo (ex-Photo AI):** same upscalers plus integrated denoise/sharpen and Autopilot batch,
  useful because AI frames often need de-artifacting before upscale. Personal $199/yr, Pro $599/yr.
- **Image / Image Web + API:** the cloud offload when a generative model does not fit your 32 GB,
  or to API-batch hundreds of stills. Per-credit pricing (1 credit ~= output-MP allowance).
- **Naming corrections:** the brief's "Lines" model does not appear in current Gigapixel docs
  (likely Text & Shapes or Art & CGI). "Starlight Precise 2.5" and "Astra" are **video** models,
  not stills.

Topaz stills are **per-frame with no temporal model**, so for motion route to Topaz Video. For
single frames, posters and storyboards they are the per-frame quality ceiling.

---

## 6. Hosted APIs and apps, with price

The key mental model: **most "magic" upscalers are hosted wrappers of the open models above.**
Clarity (on fal/Replicate) is open SD1.5 + ControlNet-tile; Aura-SR is an open GigaGAN repro;
ESRGAN is free. You pay a convenience/scale premium, not for secret weights. The genuinely
proprietary IP worth a premium: Magnific, Topaz Gigapixel/Bloom/Wonder, Crystal (faces), Recraft
Crisp, Ideogram native, and SeedVR2 (open but conveniently hosted).

`$/100` is for a 720p to 4K still (~8.3 MP output). Verified prices.

| Tool / endpoint | Underlying model (openness) | $/100 (4K still) | AI-gen fit | Pareto |
|---|---|---|---|---|
| **fal SeedVR2 image** | SeedVR2 (open) | **~$0.83** | high | cheap-fast |
| **Recraft Crisp** (via fal) | proprietary deterministic | **$0.40** | med-high | cheap-fast |
| **Ideogram Upscale** (native) | proprietary, style-preserving | **$6** | med-high | cheap-fast |
| **Topaz Gigapixel API** | proprietary GAN | **$8-12** | high | balanced |
| **fal Crystal** | proprietary face model | **~$13** | high (faces) | balanced |
| **Recraft Creative** | proprietary generative | **~$13.50** | medium | balanced |
| **Magnific** (in-Freepik) | proprietary | **~$16** | high (Precision) | quality-tier |
| **Ideogram to Topaz** | Topaz (marked up) | **$24** | high | balanced |
| **fal Clarity** | open SD1.5 + CN-tile | **~$25** | med-high | balanced |
| **Topaz Bloom / Wonder API** | proprietary generative | **$40-60** | medium | quality-tier |
| **fal Flux Vision** | Flux-based | **~$83** | medium | quality-tier |
| **Replicate SUPIR** | open SUPIR | **$5.40-35** (deployment-dependent) | high (restore) | quality-tier |
| **Replicate Clarity** | open Clarity | **~$3.30** | med-high | cheap-fast |
| **Krea Upscale** | Topaz under the hood | **flat fee** ($9/$35/$70 mo) | med-high | balanced |

**Flat-fee note:** Krea routes to Topaz and charges a flat monthly compute budget (commercial
license on all paid tiers, from $9/mo; Free-tier commercial rights are ambiguous on the page). At
your moderate volume the $35 Pro or $70 Max fee is effectively the whole cost, which makes it
attractive if you do not need to know exactly which model ran (it hides the graph, a downside for
a reproducible benchmark).

**Pricing notes (verification):** Magnific standalone tiers triangulate to **Pro $39 / Premium
$99 / Business $299 per month** (2,500 / 6,500 / 20,000 credits, no rollover, ~17% off annual).
Recraft Crisp is **platform-dependent**: $0.004/image on fal but ~$0.044/run on Eachlabs, so cite
the platform with the price. An earlier "Magnific scored 4.6/5 on AI images" figure was a
fabrication from a secondary page and has been struck.

---

## 7. How this should be benchmarked

Full protocol in [methodology.md](./methodology.md). The image-specific headline:

- **Two strictly separated test sets.** Set A: native AI-gen 720p frames, scored with
  no-reference + preservation metrics only (no ground truth exists). Set B: real 4K downscaled to
  720p then upscaled, the *only* place full-reference metrics (PSNR/SSIM/DISTS) are valid. Never
  mix B's numbers into A's ranking.
- **Never rank on a single no-reference metric.** Use an ensemble (MUSIQ + MANIQA + TOPIQ) with an
  **anti-over-sharpening sentinel** (high-frequency energy ratio + halo detection), because those
  metrics reward the over-sharpening that AI upscalers produce.
- **Guard identity and text.** ArcFace cosine similarity on faces (source vs upscale) catches
  generative face drift; OCR character-error-rate catches mangled text. These are the anti-cheat
  for the "looks great, wrong face" failure.
- All of this runs through one toolbox, [`pyiqa`](https://github.com/chaofengc/IQA-PyTorch),
  which is **PolyForm Noncommercial**: fine for the public research repo, audit before using its
  scores inside a commercial product.

---

## 8. Open contradictions and licensing map

- **SeedVR2 appears in three places** (video, this page, and the deterministic-image slice that
  correctly flags it as diffusion, not deterministic). It is one model: ByteDance, open, diffusion,
  works on stills and video. Treated fully on the [video page](./video-upscalers.md).
- **AuraSR v2 is a 2024 release**, not 2025/2026 as the brief implied. No v3 found.
- **Commercial-license map** (the most actionable output for the studio):
  - **Clean (commercial OK):** Real-ESRGAN (BSD), Real-CUGAN (MIT), SwinIR (Apache), HAT (Apache),
    DRCT (MIT, and a real-world Real-DRCT-GAN variant exists on HF), SeeSR (Apache), SeedVR2
    (Apache), AuraSR v2 weights (Apache), AdcSR (Apache code), Topaz (paid), Recraft, Krea (paid),
    Ideogram, Magnific (paid), 4xNomos2 (CC-BY).
  - **Non-commercial (cinema/auteur only, or license it):** SUPIR, FLUX.1-dev tile, FluxSR, InvSR,
    StableSR, Upscale-A-Video, 4x-UltraSharp, 4x-AnimeSharp, pyiqa toolbox.
  - **Verify before shipping:** CCSR-v2, SinSR, DiT4SR (Pi-Lab 1.0), most community ESRGAN
    checkpoints, OSEDiff weights (Apache code but SD2.1-base weights have their own terms).

---

## 9. Gaps (what to measure next)

- **No first-party RTX 5090 benchmark exists for any diffusion image tool** (SUPIR, Clarity, CCSR,
  Flux-tile, one-step models). All 5090 framings are inferred from 4090/A100. The one verified 5090
  figure is the TensorRT ESRGAN number (12.7 fps). Measuring sec/image and peak VRAM on the
  reference card, at 720p to 1080p and 720p to 4K, is the biggest missing piece.
- **SUPIR's exact 4090/5090 timings** are widely repeated but unanchored to a primary source.
- **The 2026 one-step SOTA** (FiDeSR, DiT one-step SR) is paper-confirmed but not yet packaged as
  usable ComfyUI nodes. Recheck H2 2026. (A dedicated scout for this was lost to the session limit
  and should be re-run.)
- **Pricing to re-pull:** Magnific standalone vs in-Freepik tiers (sources differ ~5x, the page
  blocked scraping); fal SUPIR endpoint existence; Krea's full 7-model list; Recraft Creative API
  price.
- **No public head-to-head on synthetic AI-gen input** exists for these tools; every SR benchmark
  uses real-photo degradation. The "amplifies generative artifacts" claim is reasoned from
  training-domain mismatch, not measured. A controlled A/B on your own LTX-2.3 / Wan 2.2 frames is
  the only way to settle it for your footage.

---

## Sources

Local diffusion: [SUPIR](https://github.com/kijai/ComfyUI-SUPIR) ·
[SUPIR license #51](https://github.com/Fanghua-Yu/SUPIR/issues/51) ·
[Clarity](https://github.com/philz1337x/clarity-upscaler) · [CCSR](https://github.com/csslc/CCSR) ·
[SeeSR](https://github.com/cswry/SeeSR) · [StableSR](https://github.com/IceClear/StableSR) ·
[Flux-tile](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler) ·
[OSEDiff](https://github.com/cswry/OSEDiff) · [AdcSR](https://github.com/Guaishou74851/AdcSR) ·
[InvSR](https://github.com/zsyOAOA/InvSR).
Deterministic: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) ·
[Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) ·
[SwinIR](https://github.com/JingyunLiang/SwinIR) · [HAT](https://github.com/XPixelGroup/HAT) ·
[DRCT](https://github.com/ming053l/DRCT) · [AuraSR v2](https://huggingface.co/fal/AuraSR-v2) ·
[OpenModelDB](https://openmodeldb.info/) ·
[5090 TensorRT benchmark](https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt).
Topaz: [Gigapixel](https://www.topazlabs.com/topaz-gigapixel) ·
[pricing](https://www.topazlabs.com/pricing) ·
[Next-Gen launch](https://www.prnewswire.com/news-releases/topaz-labs-announces-its-largest-single-release-of-ai-models-in-company-history-with-next-gen-launch-302756375.html).
APIs: [Magnific](https://www.tooljunction.io/ai-tools/magnific-ai) ·
[Krea](https://www.krea.ai/pricing) · [Topaz API](https://www.topazlabs.com/api) ·
[fal SeedVR2 image](https://fal.ai/models/fal-ai/seedvr/upscale/image) ·
[fal Clarity](https://fal.ai/models/fal-ai/clarity-upscaler) ·
[Replicate SUPIR](https://replicate.com/shanginn/supir) ·
[Ideogram API](https://ideogram.ai/features/api-pricing) ·
[Recraft Crisp](https://fal.ai/models/fal-ai/recraft/upscale/crisp).
Method: [over-sharpening trap](https://arxiv.org/pdf/2504.18524) ·
[IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) ·
[rethinking SR evaluation / RQI](https://arxiv.org/html/2503.13074v3).

*Last updated 2026-06-16. Research pass only (no adversarial second pass, no on-device 5090
measurement yet).*
