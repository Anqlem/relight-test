# Relighting architecture: changing sun direction / time of day without breaking the building

*Mikhail Shmelev · MyArchitectAI test task · November 2026*

**Short answer.** Changing the *time of day* is solved well enough to ship. Your own Relight presets kept geometry and materials on every test image and produced a convincing night. Moving the *sun direction* is not ready. The only tool that actually moved the shadows (a FLUX.2 Klein LoRA) also changed facade colours, added objects, failed on the interior, and sits on a non-commercial base model. I would treat direction control as R&D, not ship it yet.

## 1. Market scan

| Option | How light is controlled | Cost / licence | Tried? |
| --- | --- | --- | --- |
| Sun Direction LoRA on FLUX.2 Klein 9B (HF Space) | drag a sun on a sphere: real direction + elevation | free Space; base model **non-commercial** | yes |
| Qwen-Image-Edit-2509 Relight LoRA | text prompt | Apache 2.0; free Space / Replicate | yes |
| MyArchitectAI Relight | presets (golden hour, night, overcast…) | subscription | yes (baseline) |
| IC-Light v1 / v2 | light source or background; v1 runs locally (SD1.5) | open / fal API | no |
| fal `image-editing/time-of-day` | text, time of day | $0.04 / image | no |
| Bria FIBO Edit Relight | text, direction + time | API, commercially licensed output | no |
| Magnific Relight | prompt, reference image or light map | €0.10 / operation | no |
| LTX-2.3 Relight IC-LoRA | sun direction, for **video** | open weights | no (relevant for image-to-video) |
| General editors (Gemini / Nano Banana, Flux Kontext, GPT-image) | free text | API | no |

## 2. Test setup

Three Unsplash photos, chosen so that a wrong result is easy to spot: a flat facade where balcony shadows show the sun direction, a corner building with one lit side and one shaded side, and an interior where window-blind stripes on the wall show the sun angle. Before running anything I wrote down a target for each image (for example, "sun from the left instead of the right") so I could not grade the results against whatever came out. Each option got the same three targets: by sphere for Sun Direction, by prompt for Qwen. MyArchitectAI got presets, because it has no direction control: *golden hour* and *night*.

All before/after images are in `results/grids/`. Under every result there is a heatmap of where the light got brighter (red) or darker (blue).

## 3. Results

| Option | Light change (auto) | Geometry | Materials | Matches target | Shadows plausible | Frame kept | Time / image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sun Direction | **0.73** | 4.7 | **2.0** | **5.0** | 5.0 | 98–100% | ~21 s |
| Qwen Relight | 0.03 | 4.0 | 5.0 | **1.0** | 1.0 | **67–94%** | ~11 s (6 s warm) |
| MyArchitectAI · night | 0.71 | 5.0 | 5.0 | 4.3 | 4.0 | 100% | ~16 s |
| MyArchitectAI · golden hour | 0.04 | 4.7 | 4.0 | 5.0* | 5.0 | 98–100% | ~16 s |

Scores are 1–5 by eye against a fixed rubric (`results/runs.csv`). *Control case: see "What went wrong".

- **Sun Direction is the only option that moved the sun.** On the facade, balcony shadows flipped from left to right. But the warm orange facade turned pale peach, a traffic light appeared on the corner image, and on the interior the sun simply disappeared instead of moving. The LoRA was trained on exteriors only.
- **Qwen Relight barely changes the light.** It acts as a colour/contrast grade and the shadows stay where they were, whatever the prompt says. It also silently zoomed in 1.22× on two images, cutting off a third of the frame.
- **MyArchitectAI night is the most usable result.** Geometry and materials are intact and the lit windows look natural. Two physical errors remain: on the facade, the tree shadow stays on the wall at night; in the interior, the wall lamp stays off and the image reads as blue hour.

## 4. How I decided which result is better

I used two automatic checks, plus scores by eye against a written rubric. *Structure*: Canny edge precision/recall against the input, as in my previous benchmark. *Light change*: one minus the correlation of heavily blurred luminance. A colour grade scores about 0, and moved shadows score well above 0.

Both checks turned out to be **filters for obvious failures, not judges**:

1. **Edge metrics confuse shadows with geometry.** When the light really changes, shadow edges disappear and appear elsewhere, so a correct relight loses edge precision (Sun Direction: about 0.6).
2. **Edge recall collapses at night** (0.13–0.25), because edges become invisible, not because the building broke.
3. **Light change cannot tell "moved" from "removed".** The Sun Direction interior got the highest score (0.88), yet it is the worst result, because the sun just vanished.
4. **Silent reframing breaks every metric.** Qwen's zoom first looked like a lighting change. I added automatic alignment (ORB features + similarity transform) before scoring, which also measures how much of the frame was lost.

The final ranking therefore comes from the rubric scores. The metrics are there to catch "nothing changed", "building redrawn" and "frame cropped".

## 5. What went wrong

- **My first comparison was unfair.** All three photos were shot at golden hour, so the *golden hour* preset had almost nothing to change (light change 0.01–0.05). I kept it as a control and added *night* as a real test.
- **My first scores were inconsistent.** I gave Qwen 5/5 for "light changed" while my own note said "almost nothing changed". I then wrote a rubric and rescored every option with it.
- **Free infrastructure failed twice.** The ZeroGPU daily quota ran out after two images as an anonymous user. The official Qwen Relight Space crashed on start-up, so I used another Space with the same LoRA.

## 6. Time and cost per image

Measured on free HF Spaces (shared GPUs, including queue): 6–23 s. MyArchitectAI took about 16 s, faster than the ~30 s stated on the product page. Hosted APIs: FLUX.2 Klein 9B edit with a custom LoRA on fal costs $0.02 per megapixel, so about **$0.04** for a 1 MP image; fal time-of-day costs $0.04. Own GPU: an L40S on RunPod costs $0.79–1.09 per hour. At an *assumed* 5 s per image, that is **≈ $0.001–0.002** per image. That speed is an estimate, not a measurement.

## 7. What I would ship, and how

1. **Now: keep time of day on presets** (current Relight). Add two cheap fixes found here: remove cast shadows from off-frame objects for night/overcast, and switch on interior light fixtures for night.
2. **Next, as R&D: sun direction from geometry, not from a prompt.** Do not ship the Sun Direction LoRA as-is. Its base (Klein 9B) is non-commercial, it shifts materials, and it fails on interiors. Every tool here *paints* shadows. None of them knows where the balconies actually are, which is why shadows stay put, vanish, or belong to an off-frame tree at night. The idea I would test first is a two-step pipeline. (a) Estimate depth and surface normals from the image (Depth Anything, which I already used in my ControlNet benchmark), then compute a shading and cast-shadow map for the requested sun position. (b) Let a diffusion model turn that map into the final light, while a Canny ControlNet holds the edges; in my benchmark, Canny held geometry best (edge F1 0.83 vs 0.47 for Depth). The shadow direction then comes from geometry and is correct by construction. The model only has to make it look real. This is an untested idea: the open questions are depth quality on flat facades, and glass/reflections. If depth from a single photo turns out too unreliable, the same pipeline still works for your users' renders, where the real 3D model and exact geometry exist.
3. **Serving:** start on a hosted API (fal, about $0.04/image) to validate demand. Move to own GPUs (L40S class, under ~$0.002/image if a 4-step model runs in a few seconds) once volume justifies it. Gate every release on a small regression set like this one: "did the shadows move to the target side" and "did facade colour stay within tolerance", plus a human check.
