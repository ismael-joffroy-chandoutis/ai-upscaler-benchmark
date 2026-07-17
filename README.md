**English** · [Français](README.fr.md)

# AI Upscaler Benchmark, for generative video and image

A continuously updated, source-cited comparison of video and image upscalers for
**AI-generated footage**, across local (RTX 5090 class) and API solutions, on the
quality / cost / speed trade-off.

**Scope.** You generate at roughly 720p with models like LTX-2.3, Wan 2.2, Veo 3.1,
Seedance 2.0 and Kling, and you want to push to 1080p or 4K. Every recommendation is
framed for two distinct use-cases: cinema-grade quality on one side, commercial-volume
throughput on the other. There is no single winner; the answer depends on the shot and
the budget.

**Reference machine** for local figures: NVIDIA RTX 5090 (32 GB GDDR7, Blackwell),
AMD Ryzen 9 9900X, 96 GB DDR5, 8 TB NVMe (Samsung 990 Pro) at ~14,000 MB/s.

> **Status.** A living comparative, sourced June 2026 and cross-checked. Four tiers are **measured on the reference RTX 5090**: ESRGAN, Topaz classic (Proteus 15 fps@1080p / 7.3 fps@4K), SeedVR2 and FlashVSR ([data/measurements-5090.json](./data/measurements-5090.json)). Other speed figures are sourced and labelled with their GPU. See the [CHANGELOG](./CHANGELOG.md) for revisions.

---

## The one insight that organizes everything

Your input is **synthetic**. A 720p AI frame has no real grain, but it carries warping,
temporal flicker, waxy skin and over-smoothing. Almost every upscaler was trained on *real*
degraded footage, so:

- **GAN/CNN upscalers sharpen what is already there**, crisping plastic skin and amplifying
  flicker.
- **Diffusion upscalers regenerate detail**, which fits synthetic input better, but they can
  invent a new face and drift across frames.

The sharpest single fact: the strongest open local video tool, **SeedVR2**, carries an
explicit warning on its own model card that it over-sharpens 720p AI-generated video. Your
exact use-case is its documented failure mode. The fix is not a different tool, it is a
lighter hand (lower restoration strength) plus the right evaluation, which is why this repo
weights its methodology as heavily as its tool list.

---

## Headline picks (June 2026)

**Video** (full detail in [docs/video-upscalers.md](./docs/video-upscalers.md)):

| Use-case | Local pick | API pick |
|---|---|---|
| Cinema 4K, quality-max | SeedVR2-7B (low strength) | Topaz via fal (Starlight Precise) |
| Short-drama volume | FlashVSR Tiny / SeedVR2-3B | WaveSpeed SeedVR2 (~$3/min 4K) |
| Cheap previews | Real-ESRGAN + `vs_temporalfix` | fal RealESRGAN |
| Face-safe, no hallucination | Topaz Proteus (Manual) | Topaz via fal (Proteus) |

**Image** (full detail in [docs/image-upscalers.md](./docs/image-upscalers.md)):

| Use-case | Local pick | API pick |
|---|---|---|
| Cinema hero still | SUPIR (v0F) or Flux-tile | Magnific Precision / Topaz Gigapixel |
| Faithful, no invention | ControlNet Tile (SDXL), HAT, DRCT | Topaz Gigapixel ($8-12/100) |
| Cheap batch | Real-ESRGAN (measured 1.7 fps fp16, x4 720p, on the 5090) | fal SeedVR2 (~$0.83/100) |

**Exact settings for every model** (copy-pasteable): [docs/recipes.md](./docs/recipes.md).
**Price, value, and what to buy** (local vs API, the Topaz sub-vs-per-use call, provider ranking):
[docs/economics.md](./docs/economics.md).

**The local-vs-cloud rule of thumb at moderate volume (30 min-3 h/month):** classic/GAN
upscaling is far cheaper local (unlimited on the 5090). Diffusion 4K is too slow locally for
volume (0.2-2 fps), so the rational split is local card for 1080p diffusion plus overnight 4K
hero shots, cloud for 4K at volume.

---

## Repository layout

```
README.md                  this file (English)
README.fr.md                this file (French)
docs/
  video-upscalers.md       the video benchmark
  image-upscalers.md       the image benchmark
  methodology.md           how to benchmark upscalers on synthetic input (reproducible)
  recipes.md               exact, copy-pasteable settings per model
  economics.md             price x quality, local-vs-API break-even, the Topaz buy decision
data/
  registry-schema.json     JSON Schema for a tool record
  tools.json               the machine-readable tool registry (tables derive from this)
CHANGELOG.md               dated record of what changed and was re-verified
CITATION.cff               how to cite
LICENSE.md                 CC BY-NC-ND 4.0 (text) + PolyForm NC 1.0.0 (code)
```

The benchmark data lives in `data/tools.json`, not in the prose. The Markdown tables are a
human-readable view of that registry. This is deliberate: keeping it up to date means editing
a small JSON record (and its `last_verified` date), not rewriting paragraphs.

---

## How to keep this current

1. Edit or add a record in `data/tools.json` following `data/registry-schema.json`. Set
   `last_verified` to today and cite a primary source for every price or hard number.
2. Note the change in `CHANGELOG.md` with the date and the source checked.
3. Anything you cannot confirm against a primary source: mark it `UNVERIFIED` and keep it
   visible rather than dropping it. The edge of what is confirmed is itself information.
4. When the reference 5090 measurements land, replace the inferred speed numbers and bump the
   minor version.

---

## Methodology in one paragraph

There is no ground-truth 4K for AI-generated 720p, so standard full-reference metrics
(PSNR/SSIM/LPIPS) mostly do not apply, and the popular no-reference metrics (MUSIQ, NIQE,
CLIP-IQA+) **reward the over-sharpening and hallucinated detail that AI upscalers produce**.
The protocol therefore scores a per-clip triad (no-reference quality, fidelity/hallucination
anti-cheat, temporal consistency) and never ranks on a single number. Full reproducible
protocol in [docs/methodology.md](./docs/methodology.md).

---

## License and citation

Text, tables and data: **CC BY-NC-ND 4.0**. Code and schema: **PolyForm Noncommercial 1.0.0**.
See [LICENSE.md](./LICENSE.md). To cite, see [CITATION.cff](./CITATION.cff).

Maintained by Ismaël Joffroy Chandoutis. Contributions of verified measurements (especially
real RTX 5090 figures) and corrections to flagged items are welcome via issues.

By [Ismaël Joffroy Chandoutis](https://ismaeljoffroychandoutis.com).
