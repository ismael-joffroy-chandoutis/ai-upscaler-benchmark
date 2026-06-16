# Economics: price, value, and what to actually buy

The recommendation is not "which tool is best" but "what is the cheapest way to get the quality
you need, at your volume." Prices verified June 2026 (links at the bottom); they move, so re-check
before committing money. Assumed volume: **30 min to 3 h of upscaled output per month**, two
use-cases (cinema-grade hero work, and short-drama commercial volume). Hardware already owned: a 3-GPU render fleet, **2x RTX 5090 (32 GB) + 1x RTX 4090 (24 GB)**.
Per-tool speeds in this repo are measured on a single 5090; the fleet multiplies overnight
throughput, which changes the local-vs-cloud math for 4K diffusion (section 2).

---

## 1. The four cost models, and how each scales

| Model | Examples | You pay for | Wins when |
|---|---|---|---|
| **Own hardware** | the RTX 5090 you already have | electricity only (~$0.10-0.15/GPU-hour) | high volume of *fast* models; anything you can run overnight |
| **Subscription** | Topaz Video $299-699/yr, Astra $19-299/mo | a fixed monthly fee, capped or unlimited | steady, predictable, high-frequency use of one vendor |
| **Per-use / serverless** | fal, Replicate, WaveSpeed (per second or per megapixel) | only what you render | bursty, low-to-moderate volume, no lock-in |
| **Credits** | Astra, Topaz API, Magnific, Krea | a pre-bought balance that expires | a known monthly batch size |

The single fact that overrides all of this: **owning the 5090 only helps for models the card can
actually keep up with.** Throughput, measured on the card, decides feasibility before price does
(next section).

---

## 2. The throughput ceiling (why local is not always an option)

From the [measured numbers](../data/measurements-5090.json), cost per minute of finished output
*locally* (electricity at ~$0.12/GPU-hour) and whether it is even feasible at this volume:

| Local path | Measured speed | GPU-time per min of output | Local cost/min | Feasible at 30min-3h/mo? |
|---|---|---|---|---|
| Topaz classic (Proteus) 4K | 7.3 fps | ~3.3 min | ~$0.01 | **Yes, trivially** |
| ESRGAN x4 (fp16) | 1.7 fps | ~14 min | ~$0.03 | **Yes** |
| SeedVR2 diffusion **1080p** | 0.35 fps | ~1.1 h | ~$0.14 | Yes up to ~1 h/mo (overnight batches) |
| SeedVR2 diffusion **4K** | 0.13 fps | ~3 h | ~$0.37 | **No** (3 h of compute per *minute* of 4K) |
| FlashVSR x2 (1440p) | 0.80 fps | ~30 min | ~$0.06 | Yes, modest volume |

**On one 5090, 4K diffusion is throughput-infeasible at volume:** one minute of 4K SeedVR2 is
~3 GPU-hours; a single card's monthly budget buys ~1 minute. **But there are three cards.**

**Fleet capacity (2x 5090 + 1x 4090, overnight).** The 4090 runs at ~0.6-0.7x a 5090 on these
models, and its 24 GB caps the heaviest configs (SeedVR2 7B fp16, FlashVSR full), but it handles
3B fp8 + tiling, classic and ESRGAN fine. Treat the fleet as ~2.6x a single 5090. Run the towers
overnight, say 8 h/night x ~25 nights = **~600 GPU-hours/month** of batch capacity. At ~3 GPU-hours
per minute of 4K SeedVR2, that is **~200 minutes (~3 h) of 4K diffusion per month, rendered locally
overnight** for the price of electricity. So **your entire 30 min-3 h/month 4K-diffusion budget fits
on the fleet** (~$8-46 of electricity for the full 3 h, vs $90-540 on the cheapest cloud). The cloud
flips from *mandatory* to *optional overflow*. The classic/GAN tier is free on any one card, 1080p
diffusion is comfortable, and 4K diffusion is now an overnight farm job. That is what drives the
recommendations below.

---

## 3. Service-provider price ranking (verified)

**Video, USD per minute of finished 4K** (5 s 720p->4K@24fps, extrapolated):

| Service | Model | $/min 4K | $/min 1080p | Notes |
|---|---|---|---|---|
| **WaveSpeed** | SeedVR2 | **$3.00** | $1.80 | cheapest verified 4K; flat per-second, 5 s min, up to 10 min/clip |
| **Topaz via fal / Replicate** | Gaia 2 | **$2.40** | $0.60 | half-price Topaz; good for stylized/animated |
| Topaz via fal / Replicate | Starlight/Proteus | $4.80 | $1.20 | the Topaz "look", per clip, no sub |
| fal | RealESRGAN | ~$9.6 | ~$2.4 | per-MP; previews only (no temporal) |
| fal | SeedVR2 | ~$12 | ~$3 | per-MP punishes 4K; great value at 1080p |

**Image, USD per 100 finished 4K stills:**

| Service | Model | $/100 | Notes |
|---|---|---|---|
| **Recraft Crisp** | deterministic | **$0.40** | cheapest; clean, no hallucination |
| **fal SeedVR2** | SeedVR2 | **$0.83** | best $/quality; up to 10K |
| Ideogram | native | $6 | style-preserving 2x |
| Topaz Gigapixel API | Gigapixel | $8-12 | faithful premium |
| fal Crystal | face model | $13 | portraits |
| Magnific | Precision/Creative | $16 | premium reinvention |
| fal Clarity | open SD1.5 | $25 | you pay for hosting an open model |
| fal Flux Vision | Flux | $83 | most expensive, rarely worth it |

**Cheapest credible provider:** video 4K = **WaveSpeed** (SeedVR2, open model, $3/min); image =
**fal** (SeedVR2 $0.83/100, or Recraft Crisp $0.40 for deterministic).

---

## 4. Local vs cloud break-even, at your volume

Monthly cost to render your whole 30 min-3 h budget on the cheapest cloud path vs locally:

| Workload | Cheapest cloud / month (30 min -> 3 h) | Local |
|---|---|---|
| Classic / GAN upscaling (1080p or 4K) | fal RealESRGAN ~$72 - $432 | ~$1 electricity. **Local wins by 100x.** |
| Diffusion **1080p** | WaveSpeed/fal SeedVR2 ~$54 - $324 | feasible local up to ~1 h/mo (free-ish), cloud above |
| Diffusion **4K** | WaveSpeed SeedVR2 **$90 - $540** | **~$8-46 electricity on the 3-GPU fleet, overnight** (~200 min/mo capacity). Local wins ~10x; cloud = overflow. |

**The rational split, with the fleet:** run essentially everything **locally** — classic/GAN and
1080p diffusion on any single card, and **4K diffusion as an overnight batch spread across the three
towers**. Use the cloud (WaveSpeed) only for **overflow**: a hard deadline, or when a tower is tied
up (e.g. one 5090 busy generating). Reserve a premium API (Topaz-via-fal, Magnific) for the
occasional hero shot where the specific look justifies it. Net effect: at your volume the monthly
cloud bill should be near zero, replaced by a few dollars of electricity and a render queue.

---

## 5. The Topaz decision: buy annual, monthly, or pay per use?

The numbers (June 2026):

- **Topaz Video desktop:** $59/mo, or $299/yr (Personal), or **$699/yr Pro** — unlimited *local*
  rendering + 25 cloud credits/mo. Subscription-only (no perpetual).
- **Astra cloud** (GenAI-native, Creative + Precision): $19/mo (400 cr), $99/mo (1400 cr), $299/mo.
- **Topaz via fal / Replicate:** no sub, $4.80/min of 4K ($2.40 with Gaia 2).

The reasoning for your situation:

1. **Topaz classic (Proteus/Iris/Gaia) already runs offline on NOMAD** from a cached license, free.
   So you do **not** need a subscription for face-safe classic upscaling.
2. **Topaz Starlight diffusion** (the part that needs a live sub + Neuroserver) is the only reason
   to pay Topaz, and 4K Starlight is slow locally anyway, so you would route it to **Astra cloud**
   or pay per clip.
3. **An annual $699 Pro sub pays off only above ~$58/mo of Topaz-specific use.** At 30 min-3 h/mo,
   routing 4K to WaveSpeed ($3/min, cheaper and open) and bulk to your free local stack, you will
   not hit that. The annual lock-in is dead weight.

**Verdict: do not buy an annual Topaz subscription.**
- Default: **pay per clip via fal/Replicate** ($2.40-4.80/min 4K) when you specifically want the
  Topaz look or face-safe Starlight, and run classic locally for free.
- Only if Starlight diffusion becomes a *daily* habit: take **Astra Standard ($19/mo)** or the
  **monthly Topaz Video ($59, cancel anytime)** — never the annual, until your usage is proven and
  steady for several months.

---

## 6. What I recommend

**Local stack (your owned 5090, marginal cost = electricity):**
- Cheap/fast bulk: Real-ESRGAN / community ESRGAN (fp16, commercial-clean models) + `vs_temporalfix`.
- Face-safe / deterministic: Topaz **Proteus** (classic, offline, free).
- Quality detail rebuild: **SeedVR2** (3B for volume, 7B for hero), small batch + tiling + offload.
- Fast streaming: **FlashVSR** x2.
- Stills hero (cinema, non-commercial OK): **SUPIR** v0F.

**API stack (pay per use, no lock-in):**
- 4K video at volume: **WaveSpeed SeedVR2** ($3/min) — cheapest, open, license-clean.
- The Topaz look / face-safe per clip: **Topaz via fal** (Gaia 2 for cheap, Starlight Precise for quality).
- Image bulk: **fal SeedVR2** ($0.83/100) or **Recraft Crisp** ($0.40).
- Premium image hero: **Magnific Precision** or **Topaz Gigapixel API**.

**By use-case:**
- **Cinema (few hero shots):** local SeedVR2/SUPIR for quality + Topaz-via-fal Starlight for the look. Cost: near zero local + a few dollars of API.
- **Short-drama (commercial volume):** local for 1080p + bulk; **4K diffusion as an overnight batch
  across the fleet** (2x 5090 + 4090), with WaveSpeed only for overflow or deadlines. Stay on
  commercial-clean models. No Topaz sub.

**Running the fleet.** Treat the three towers as a render queue: split a night's clips across them,
put the heavy configs (SeedVR2 7B fp16, FlashVSR full) on the **32 GB 5090s** and the lighter ones
(3B fp8 + tiling, classic Topaz, ESRGAN) on the **24 GB 4090**. Keep one 5090 free if you are also
generating. A simple per-clip job script over SSH (one queue, three workers) turns "4K diffusion is
too slow locally" into "it is done by morning." That is the whole point of owning the hardware: the
marginal cost of an extra night of 4K is electricity, not a cloud invoice.

---

## 7. Will the pricing practices change?

Likely, and in your favour:
- **Per-use / serverless is displacing subscriptions** for bursty, low-volume creators. Paying only
  for what you render is winning; annual locks make less sense for an auteur cadence.
- **Open models commoditize the local tier.** SeedVR2 (Apache-2.0) and FlashVSR remove the need to
  pay for a local upscaler at all, which pressures Topaz's and Magnific's premiums downward.
- **Vendors retreat to the diffusion / cloud tier**, where they still lead (Topaz Starlight/Astra,
  Magnific Creative). Expect their value to concentrate there, and their classic-tier moat to erode.
- **Expect 4K diffusion API prices to keep falling** as open models and competition (fal, Replicate,
  WaveSpeed and Chinese clouds) drive each other down. The WaveSpeed $3/min floor today is unlikely
  to be the floor for long.

Practical consequence: **commit to nothing annual.** Own your hardware, lean on open local models,
and treat the cloud as a metered utility you can switch the moment a cheaper provider appears.

---

## Sources

[Topaz pricing](https://www.topazlabs.com/pricing) ·
[fal Topaz video](https://fal.ai/models/fal-ai/topaz/upscale/video) ·
[fal SeedVR2](https://fal.ai/models/fal-ai/seedvr/upscale/video/api) ·
[WaveSpeed SeedVR2](https://wavespeed.ai/models/wavespeed-ai/seedvr2/video) ·
[fal SeedVR2 image](https://fal.ai/models/fal-ai/seedvr/upscale/image) ·
[fal Recraft Crisp](https://fal.ai/models/fal-ai/recraft/upscale/crisp) ·
[Magnific](https://www.magnific.com/) ·
on-device throughput: [data/measurements-5090.json](../data/measurements-5090.json).

*Prices verified June 2026. They move; re-check before any purchase. Last updated 2026-06-16.*
