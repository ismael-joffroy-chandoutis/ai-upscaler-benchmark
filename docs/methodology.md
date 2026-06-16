# Methodology: benchmarking upscalers on synthetic input

How to score an upscaler when the input is AI-generated and there is no high-resolution
ground truth. This is the load-bearing part of the benchmark: a wrong metric choice ranks the
wrong tool. The protocol is the same in spirit for video and image, with video adding a
temporal axis.

---

## 1. Why the standard playbook breaks

Two facts dominate:

1. **No ground truth.** The 720p AI frame *is* the source of truth; there is no real 4K
   original it was downscaled from. This disqualifies full-reference metrics (PSNR, SSIM,
   LPIPS, DISTS) for the real task. A do-nothing bicubic upscale "wins" on them because it
   adds nothing, which is backwards for an upscaler. They re-enter only through a deliberate
   downscale-then-upscale test (Set B below).
2. **The input is synthetic and already degraded** (flicker, waxy skin, warping,
   over-smoothing). No-reference metrics were trained on *naturally* degraded photos, and the
   popular ones **systematically over-reward over-sharpening and hallucinated detail**
   ([arXiv 2504.18524](https://arxiv.org/pdf/2504.18524)). MUSIQ even correlates *negatively*
   with human hallucination judgments (ρ ≈ -0.23 on the paper's combined set; LPIPS +0.17,
   DISTS +0.02, [arXiv 2507.14367](https://arxiv.org/pdf/2507.14367)): it rates confidently-wrong
   detail as good. Rank on it alone and you crown the worst hallucinator.

The consequence: report a **vector, not a number**. Quality, fidelity/hallucination, and (for
video) temporal consistency are separate axes that trade against each other.

---

## 2. The two test sets (keep them strictly separate)

- **Set A, the real task (no ground truth).** 30-50 native AI-gen 720p frames (and, for video,
  12-20 clips of 4-6 s), spanning each generator you use (LTX-2.3, Wan 2.2, Veo 3.1, Seedance
  2.0, Kling) and the content that breaks upscalers: close-up face/skin, hands, fast motion and
  pans (flicker, boiling), fine repeating texture (fabric, foliage, text, the hallucination
  bait), low-light/grain, flat gradient sky (banding). Score with **no-reference + preservation
  metrics only**.
- **Set B, the synthetic FR sanity check (constructed ground truth).** Take clean 4K reals
  (DIV2K or your own), downscale to 720p, upscale. Now full-reference metrics are valid. Use it
  only to isolate fidelity/sharpening behavior, and **never mix Set B's FR numbers into Set A's
  ranking**.

Commit the frame set (or hash-pin it with generation prompts/seeds) so the benchmark is
re-runnable.

---

## 3. Metrics by axis

All run through one toolbox: [`pyiqa` / IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)
(Chaofeng Chen, NTU S-Lab). **License: PolyForm Noncommercial 1.0.0 + NTU S-Lab.** Fine for
this public research repo; audit each metric's own weight license before using its scores in a
commercial product. Metrics are tiny next to the upscalers (sub-GB to ~2 GB VRAM), so they are
never the bottleneck on the 5090.

### A. No-reference quality (primary quality signal, but never alone)

| Metric | Catches | Caveat on synthetic input |
|---|---|---|
| **MANIQA** | GAN/SR restoration artifacts (its training domain) | Best NR pick *for SR artifacts*, still no notion of "correct". |
| **MUSIQ** | sharpness, "looks HD" | Rewards hallucinated detail. Never use alone. |
| **CLIP-IQA / CLIP-IQA+** | broad quality; probe "real photo vs CGI" via antonym prompts | A cheap plasticity/AI-look proxy. Shares the sharpening bias. |
| **TOPIQ (nr)** | semantic-guided NR | Good ensemble member. |
| **NIQE / BRISQUE** | gross blur, blockiness | The most over-sharpening-fooled. Tripwire only. |
| **Q-Align** (LMM) | most human-aligned single scorer; native video mode | Heavy (~7B), fits 32 GB quantized. Tie-breaker. |
| **FGResQ** (2025) | fine-grained restoration ranking | Closest to a metric *designed* for the narrow-quality, restoration regime. |

Use MUSIQ + MANIQA + TOPIQ as an **ensemble** (report mean and spread, the spread is itself a
flicker signal across frames), never as a lone judge.

For **video**, add a no-reference *video* metric: **DOVER**
([repo](https://github.com/VQAssessment/DOVER), aesthetic + technical branches, <2 GB VRAM),
and ideally an AIGC-specialized scorer (**AIGV-Assessor**, trained on AI-generated video).
Both were trained on real user video, so treat their "technical" branch with suspicion on
plastic AI output.

### B. Fidelity and hallucination (the anti-cheat axis)

Run output-4K against source-720p (downscale the 4K back, or bicubic-up the source; log which).
This measures *how far the tool drifted from what the source showed*, not quality. High
no-reference quality + high drift = hallucination.

- **VMAF and VMAF-NEG** ([Netflix](https://github.com/Netflix/vmaf), one ffmpeg line). Report
  **both**: a large `VMAF - VMAF-NEG` gap means the tool is winning by enhancement/hallucination,
  not fidelity. VMAF-NEG subtracts the sharpening/contrast boost that upscalers use to game
  metrics.
- **Hallucination Score proxy** (the paper's efficient **DINO/DINOv2- and CLIP-feature-based**
  proxies, [arXiv 2507.14367](https://arxiv.org/pdf/2507.14367)). The purpose-built "did it invent
  detail?" detector, reported to track human judgment far better than LPIPS/DISTS/MUSIQ (verify the
  exact correlation against the paper's table before quoting it). Run the open proxy locally;
  reserve a GPT-judge pass for cinema-tier clips only.
- **ArcFace identity drift** on faces: 512-d embedding cosine similarity, source-face vs
  output-face and frame-to-frame. The guard against an upscaler quietly changing the actor's
  face. Always on for the short-drama (talking-head) case.
- **DISTS** (preferred over LPIPS for texture-rich SR) and **RQI** (a no-ground-truth-aware
  pairwise comparator, [arXiv 2503.13074](https://arxiv.org/html/2503.13074v3)) for A/B drift on
  the same frame.

### C. Temporal consistency (video only)

- **VBench** temporal dimensions ([repo](https://github.com/Vchitect/VBench), Apache-2.0):
  `temporal_flickering`, `motion_smoothness`, `subject_consistency` (morphing),
  `background_consistency` (boiling). Built to score generated video, so directly relevant.
- **RAFT warping error / tOF**: optical flow between frame t and t+1, warp forward, measure the
  residual not explained by motion. The most reproducible, model-light flicker measure. Run flow
  on source *and* output: a tool that adds boiling shows higher warping error than its input.
- **Frame-std of the no-reference scores**: free flicker proxy (quality bouncing frame-to-frame).

### D. Preservation guards (both video and image)

- **Text/edge:** OCR character-error-rate (PaddleOCR/Tesseract) on source vs upscale crops, plus
  edge-map SSIM for ringing/halo. AI upscalers notoriously mangle small text.
- **Faces:** ArcFace IDS (above).

---

## 4. The anti-over-sharpening sentinel (mandatory)

Because NIQE/MUSIQ/CLIP-IQA+ reliably overpredict over-sharpened output, add a sentinel:
measure **high-frequency energy gain** (Laplacian-variance ratio of upscale to source) and
**halo/overshoot near edges**. Flag any candidate whose no-reference gains correlate with HF
spikes: that is the adversarial-artifact signature, not real quality. Report no-reference scores
*alongside* the sentinel, never standalone.

---

## 5. How to read the result (no single winner)

- Report a per-tool vector: **(quality, fidelity-drift, temporal)** for video,
  **(quality, fidelity-drift, preservation)** for image.
- **Disqualify metric-gaming.** If a tool tops no-reference quality but has low Hallucination
  Score, a large VMAF gap, or ArcFace drift, it is hallucinating. Flag it, do not rank it first.
- **Two verdicts per tool**, matching the two use-cases: *cinema-tier* maximizes quality subject
  to hallucination/identity/temporal staying clean (fidelity gates quality); *volume-tier*
  maximizes quality-per-second-per-dollar with a looser hallucination gate.
- **Always include a human pass** (2AFC pairwise on >=20 pairs). Every cited study agrees
  automatic metrics are advisory on synthetic content; the human spot-check is the calibration
  ground truth, logged as 1-5 MOS so you can track metric-vs-human correlation over time.

---

## 6. What to log per run (reproducibility)

clip/frame id + sha256; generator + prompt + seed; tool name + exact version/commit;
model/variant + tiling/offload settings; target res + scale factor; every metric name + version
(pin `pyiqa`, the VBench commit, the VMAF model `vmaf_v0.6.1` and `_neg`); raw per-frame arrays
(not just means); per-clip mean and std; wall-clock seconds, fps achieved, peak VRAM
(`nvidia-smi --query-gpu=memory.used`), GPU name and driver, fp8/fp16 mode; cost (local ~$0 plus
electricity, or API $/min). Commit a `metrics.lock` of pinned versions and a one-command
`run.sh`.

---

## 7. Licensing note for the commercial pipeline

`pyiqa`, DOVER, and several metric weights are non-commercial (PolyForm-NC / NTU S-Lab). They
are fine for this public research benchmark. VMAF (BSD) and VBench (Apache-2.0) are
commercially clean. Using the non-commercial scorers *inside* the paid short-drama workflow
needs a commercial license or a clean substitute. Audit before shipping.

---

## Sources

[Hallucination Score](https://arxiv.org/pdf/2507.14367) ·
[over-sharpening trap](https://arxiv.org/pdf/2504.18524) ·
[rethinking SR evaluation / RQI](https://arxiv.org/html/2503.13074v3) ·
[VMAF](https://github.com/Netflix/vmaf) ·
[VMAF-NEG](https://streaminglearningcenter.com/blogs/netflix-addresses-vmaf-hackability-with-new-model.html) ·
[VBench](https://github.com/Vchitect/VBench) · [DOVER](https://github.com/VQAssessment/DOVER) ·
[IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) · [Q-Align](https://github.com/Q-Future/Q-Align) ·
[AIGV-Assessor](https://arxiv.org/pdf/2411.17221) · [FGResQ](https://arxiv.org/html/2508.14475v3) ·
[ArcFace ID-drift](https://arxiv.org/html/2510.14255v3).

*Last updated 2026-06-16.*
