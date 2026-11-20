# RoadVision AI — Technical Specification

**Project:** RoadVision AI — Synthetic Visual Reasoning Benchmark  
**Track:** Innodata AI Hackathon 2026 — Visual Task Factory  
**Status:** implementation in progress. This document is normative for the code
in `src/roadvision`; when the two disagree, the code plus tests are the source
of truth and this document must be corrected.

---

## 1. Purpose

RoadVision AI is a deterministic task factory for visual reasoning. It creates
synthetic images of road-pavement defects, derives ground truth from the scene
representation that produced the image, generates natural-language questions,
and verifies model answers with deterministic checkers.

The benchmark measures:

| Capability | What it tests |
| --- | --- |
| `counting` | How many instances of a defect class are visible |
| `classification` | Which defect class a described instance belongs to |
| `localization` | Bounding-box regression in normalized coordinates |
| `spatial_relation` | Left/right and above/below relationships |
| `area_comparison` | Which of two instances has the larger visible area |
| `occlusion` | Whether/how many visible instances are partially covered |
| `evidence_sufficiency` | Recognising that the image cannot support an answer |

This is an evaluation benchmark, **not** a certified road-inspection system.
Depth, defect volume and material composition are explicitly out of scope: a
single 2D image does not contain that information, so the generator never
invents a "true" value for it.

---

## 2. Design invariants

1. **Ground truth is generative, not perceptual.** Object geometry, masks,
   visibility and counts come from the same scene data that the renderer uses.
   No multimodal model is used to create or repair labels.
2. **Reference answers are functions of ground truth only.** Each task records
   the object IDs and the deterministic derivation rule used to produce its
   reference answer.
3. **Answers never reach the evaluated model.** Internal task files contain
   answers; public task files and provider requests do not. The provider
   interface accepts only an image, a question and an answer format.
4. **Determinism.** A seed plus a normalised configuration reproduces identical
   pixels, masks and metadata on the same generator version and environment.
5. **No hidden tolerance inflation.** Validator thresholds are fixed, tested
   with negative cases, and documented in `docs/VALIDATION.md`.
6. **Splits are grouped.** Development and test use disjoint seed blocks and
   family IDs; duplicate images are rejected.
7. **Generation is offline.** Dataset creation, validation and reporting work
   with no API key and no network.

---

## 3. System context

```
YAML config
   |
   v
+----------------+     +----------------+     +------------------+
| Scene planner  | --> | Renderer       | --> | Ground truth     |
| (seeded)       |     | (NumPy/Pillow/ |     | builder + masks  |
|                |     |  OpenCV)       |     |                  |
+----------------+     +----------------+     +------------------+
                                                     |
                                                     v
                                            +------------------+
                                            | Question factory |
                                            | + validators     |
                                            +------------------+
                                              |              |
                                              v              v
                                       internal tasks   public tasks
                                       (answer + GT)    (image + Q)
                                                             |
                                                             v
                                                    Model adapters
                                                             |
                                                             v
                                                    Metrics / reports
```

The core produces `image + ground_truth.json + instance masks + tasks`. The
Harbor exporter and the Streamlit demo are consumers of those artefacts; neither
is allowed to become a second source of truth.

---

## 4. Module responsibilities

| Module | Responsibility | Must not do |
| --- | --- | --- |
| `config` | Pydantic-validated YAML, canonical hashing | Contain scene logic |
| `generator` | Seed derivation, defect placement, scene object plan | Render pixels |
| `renderer` | Surface texture, defect rendering, lighting, occlusion | Invent labels |
| `ground_truth` | Schema, mask statistics, visibility, validation, I/O | Call a model |
| `questions` | Candidate selection, templates, reference answers | Ask unverifiable questions |
| `validators` | Parse and score model answers deterministically | Use an LLM judge |
| `dataset` | Build/validate datasets, split manifests, exports | Mix split families |
| `providers` | Common adapter API, HTTP transport, retries, usage | Leak reference answers |
| `evaluation` | Run tasks, store responses, compute metrics | Simulate model output |
| `reporting` | JSON/CSV reports and plots | Hide per-category breakdowns |
| `harbor` | Export real Harbor-format task packages | Put answers in the agent image |
| `demo` | Streamlit inspection UI | Show answers during a live run |

Extra modules relative to the requested layout (`dataset`, `reporting`,
`providers`, `validators` are present as requested): `roadvision/dataset` is
separate because dataset orchestration is not a question-generation concern;
`roadvision/cli.py` and `roadvision/demo` provide entry points.

---

## 5. Scene and rendering model

### 5.1 Road plane

Defects are planned in a normalised **road plane** `(u, v) ∈ [0,1]²`. A fixed
projective homography maps that plane to the image. The road quad has a narrow
far edge and a wide near edge, producing perspective foreshortening. All
geometry, including bounding boxes and masks, is computed **after** projection,
so ground truth describes what is actually in the image.

A verge/background region surrounds the road. It is decorative only; questions
only refer to road objects.

### 5.2 Surface

The asphalt base is a multi-octave value-noise field with:

- coarse mottling and fine aggregate,
- directional wear streaks,
- per-image grey tone and colour cast,
- optional lane-edge paint and patches of exposed aggregate.

Noise is generated from named, independently seeded streams. Rendering uses
float32 RGB and deterministic OpenCV/Pillow operations.

### 5.3 Defects

| Class | Geometry | Visual signature |
| --- | --- | --- |
| `pothole` | Irregular radial polygon, projected | Dark interior, rough rim, gravel, depth shading |
| `crack` | Jagged polyline with branches, tapered width | Thin dark fissure with a worn edge |
| `alligator_crack` | Network of connected cracks inside an ellipse | Dense polygonal cracking pattern |
| `patch` | Irregular quadrilateral, projected | Repair material with a contrasting tone and seam |

Defect geometry is planned with rejection sampling in the road plane. Defects
do not overlap by default; this keeps instance masks unambiguous. Each defect
has a full-resolution binary mask and a visible mask (after opaque occluders).

### 5.4 Conditions

- **Weather:** `dry`, `damp`, `wet`. Wet scenes receive darker asphalt, puddle
  films and specular highlights. These are environmental effects, not
  occluders.
- **Lighting:** `overcast`, `golden`, `harsh`. Global tone mapping plus a soft
  or hard shadow pattern.
- **Shadows:** procedural canopy/structure shadows with penumbra. Shadows
  change appearance but **not** visibility in ground truth.
- **Occluders:** opaque debris sheets/cardboard/leaf clusters. These *do* change
  the visible mask and `visible_fraction`, and are stored as separate scene
  objects with their own masks.
- **Perspective:** controlled by the road quad and a `perspective_strength`
  parameter.

An occlusion scene that is used for an abstention question has exactly one
target defect covered to 90–97 % of its area, and the occluder is visibly opaque.
The image therefore does not contain enough evidence to identify the hidden
class, even though the generator knows it.

### 5.5 Determinism

`generator_version` and `schema_version` are recorded in every annotation.
Seeds are derived as:

```
stream_seed = SeedSequence([root_seed, stable_int("stream_name")])
rng = numpy.random.default_rng(stream_seed)
```

`stable_int` uses SHA-256, not Python's salted `hash()`. The configuration is
canonicalised (sorted keys, no output paths, no timestamps) and hashed with
SHA-256. `image_id = f"RV_{seed:06d}_{config_hash[:6]}"`.

PNG output is written without volatile metadata. Annotations contain no
generation timestamp. Dataset-level manifests may contain a timestamp, but they
are not part of the image-level reproducibility contract.

---

## 6. Ground-truth schema (v1.0)

Each image has one JSON annotation. Key fields:

```json
{
  "schema_version": "1.0",
  "generator_version": "0.1.0",
  "image_id": "RV_010452_ab12cd",
  "seed": 10452,
  "family_id": "fam_02090",
  "config_hash": "…",
  "image": {
    "path": "images/RV_010452_ab12cd.png",
    "width": 768,
    "height": 512,
    "sha256": "…"
  },
  "scene": {
    "surface": "asphalt",
    "weather": "wet",
    "lighting": "overcast",
    "shadow_coverage": 0.31,
    "difficulty": "medium",
    "defect_count": 3,
    "occluder_count": 1
  },
  "objects": [
    {
      "id": "damage_001",
      "type": "pothole",
      "bbox_xyxy": [120.0, 80.0, 250.0, 190.0],
      "bbox_normalized": [0.15625, 0.15625, 0.32552, 0.37109],
      "bbox_visible_xyxy": [120.0, 80.0, 250.0, 190.0],
      "centroid_xy": [185.0, 135.0],
      "centroid_normalized": [0.24089, 0.26367],
      "area_px": 14300,
      "visible_area_px": 14300,
      "visible_fraction": 1.0,
      "visibility": "fully_visible",
      "occluded": false,
      "mask_index": 1,
      "mask_path": "masks/RV_010452_ab12cd_instances.png",
      "visible_mask_path": "masks/RV_010452_ab12cd_visible.png"
    }
  ],
  "occluders": [
    {
      "id": "occluder_001",
      "type": "debris_sheet",
      "bbox_xyxy": [100.0, 70.0, 260.0, 200.0],
      "mask_index": 1,
      "mask_path": "masks/RV_010452_ab12cd_occluders.png"
    }
  ],
  "counts": {
    "pothole": 1,
    "crack": 1,
    "alligator_crack": 1,
    "patch": 0,
    "fully_visible": 2,
    "partially_occluded": 1,
    "severely_occluded": 0
  }
}
```

The values above are illustrative. Mask files are indexed PNGs: pixel value `k`
belongs to the object whose `mask_index` is `k`; `0` is background. Full and
visible masks are separate files. Occluder masks are separate from defect masks.

Visibility thresholds (fixed, documented and tested):

| Label | Criterion |
| --- | --- |
| `fully_visible` | `visible_fraction >= 0.95` |
| `partially_occluded` | `0.15 <= visible_fraction < 0.95` |
| `severely_occluded` | `visible_fraction < 0.15` |

`occluded` is defined as `visible_fraction < 0.95`.

Question generation avoids the boundaries: counting uses classes whose members
are all `>= 0.95` visible; occlusion counting only runs when partial objects are
in `[0.25, 0.75]` and no severe objects exist; abstention targets are `<= 0.10`
visible.

Validation recomputes area, bounding box, centroid and visible fraction from the
saved masks and rejects any mismatch. See `docs/DATA_SCHEMA.md`.

---

## 7. Task schema

Internal tasks (answers included) live in `tasks.jsonl`. Public tasks (no
answers) live in `public_tasks.jsonl`.

```json
{
  "task_id": "RV_CNT_RV_010452_ab12cd_01",
  "image_id": "RV_010452_ab12cd",
  "capability": "counting",
  "difficulty": "easy",
  "question": "How many potholes are visible in this image?",
  "answer_format": "single_integer",
  "reference_answer": { "count": 2 },
  "verification": { "type": "exact_integer", "params": {} },
  "generation_seed": 10452,
  "expected_abstention": false,
  "defect_types": ["pothole"],
  "object_ids": ["damage_001", "damage_002"],
  "derivation": "count(objects where type=pothole and visible_fraction>=0.95)"
}
```

`reference_answer` and `verification` are stripped from public tasks. The public
form contains `task_id`, `image_id`, `image_path`, `capability`, `difficulty`,
`question`, `answer_format`, `defect_types` and no answer-bearing field.

Question templates are semantic variants, not just paraphrase: different
descriptors (largest, nearest centre, leftmost), different pairs, different
conditions, and different answer formats are selected from the actual scene.

---

## 8. Answer validation

All checkers are pure Python and deterministic:

| Verification | Rule | Pass criterion |
| --- | --- | --- |
| `exact_integer` | Parse exactly one integer (digits or number word 0–10) | equality |
| `class_label` | Normalise synonyms, require exactly one known class | equality |
| `bbox_iou` | Parse 4 numbers; normalise if needed; enforce valid box | IoU ≥ 0.50 |
| `relation` | Normalise yes/no or left/right/above/below | equality |
| `pairwise_choice` | Exactly one allowed option | equality |
| `abstention` | Require an explicit uncertainty phrase | equality with `cannot_determine` |

Parsing rejects ambiguity. For example, `"2 or 3"` is not a count, `"pothole or
crack"` is not a class, and a box with `x2 <= x1` is invalid. Localization also
reports continuous IoU so aggregate results are not reduced to a single
thresholded number. Full rationale and negative cases are in
`docs/VALIDATION.md`.

The evaluator never uses another LLM as final judge. A model's natural-language
answer is normalised to a typed value and compared with the reference.

---

## 9. Dataset and splits

A dataset config declares a seed range, scenario distribution, render settings
and question settings. `family_id = seed // family_block`; development and test
seed ranges are chosen so their family blocks are disjoint. The split manifest
records every image hash and family. Validation rejects:

- duplicate seeds or image IDs,
- duplicate image bytes across splits,
- overlapping family IDs,
- missing images/masks/annotations,
- tasks referencing absent or invisible objects,
- any public task containing reference-answer fields.

Artefact layout:

```
datasets/
  generated/<dataset>/images/*.png
  annotations/<dataset>/*.json
  annotations/<dataset>/masks/*.png
  tasks/<dataset>/tasks.jsonl
  tasks/<dataset>/public_tasks.jsonl
  splits/<dataset>/split.json
  manifests/<dataset>/manifest.json
```

---

## 10. Model evaluation

`providers.base.MultimodalProvider` is the only path to a model:

```python
answer = provider.answer(image_bytes, question, answer_format, system_prompt)
```

The adapter receives no task object, no reference answer and no ground truth.
The OpenAI-compatible adapter uses `POST /chat/completions`, base64 image
content, temperature 0 where supported, a timeout, bounded retries with
exponential backoff, and records raw response, latency, token usage and errors.
Cost is `null` unless the operator supplies per-token prices. The `manual`
provider scores an answer file produced elsewhere and labels the run
accordingly. A scripted oracle exists only for pipeline tests and is never
included in benchmark reports as a model result.

No API key is required to generate, validate or export datasets.

---

## 11. Metrics

Reported separately, never collapsed into an unexplained single average:

- recognition accuracy,
- counting exact-match accuracy,
- localization mean IoU and pass rate@0.5,
- spatial-relation accuracy,
- pairwise-comparison accuracy,
- occlusion accuracy,
- abstention precision and recall,
- robustness: per-condition accuracy, macro-average and worst condition.

Breakdowns: model, capability, difficulty, weather, lighting, occlusion status,
defect type, and failure list. Every rate includes numerator and denominator.
JSON and CSV outputs are produced under `reports/<run_id>/`.

Robustness is defined as a **macro-average over visual condition cells**, with
the worst cell reported alongside. It is not mixed into the overall accuracy
without a label.

---

## 12. Harbor integration

Harbor is integrated as an **export target**, not as a dependency of the core.
The exporter writes the real Harbor format verified against Harbor 0.26.0 docs:

```
<task_id>/
  instruction.md
  task.toml                 # schema_version = "1.4"
  environment/Dockerfile
  environment/<image>.png
  solution/solve.sh         # oracle only
  tests/test.sh             # verifier, uploaded after the agent
  tests/reference.json      # never in the agent image
```

The agent is told to write `/app/answer.json`. The verifier applies the same
deterministic checker as the core and writes `/logs/verifier/reward.txt`.
Reference answers live only under `tests/` and `solution/`; the environment
image contains the image and instruction, never the answer. `harbor tasks lint`
and an oracle run are the acceptance checks. Details and exact commands are in
`docs/HARBOR_INTEGRATION.md`.

Harbor evaluates agents in containers; this is complementary to the direct
multimodal API evaluation, which is the primary benchmark path.

---

## 13. Security and isolation

- API keys come from environment variables / `.env`; `.env` is git-ignored.
- The public task export is checked by a test that asserts no answer-bearing
  keys exist.
- Provider prompts include only the question, answer format and image.
- The demo has separate "evaluation" and "audit" views; the audit view is not
  available during a live model run.
- Paths in annotations are relative to the dataset root; dataset validation
  refuses paths outside it.
- No downloaded model or dataset is trusted as a label source.

---

## 14. Known limitations

1. The renderer is 2D/projective; it does not model true 3D pothole volume.
2. Synthetic class signatures are controlled, so absolute scores are not a
   substitute for field validation.
3. Occlusion visibility is a geometric area fraction, not a perceptual
   visibility model. Thresholds are declared and tested, not claimed universal.
4. The abstention labels define "insufficient" operationally (near-total opaque
   occlusion). Other kinds of underdetermination are future work.
5. Harbor evaluation requires Docker and, for non-oracle agents, model
   credentials. The direct API path is independent of Harbor.
6. Absolute performance depends on the evaluated model's vision stack; the
   benchmark reports conditions and denominators so results can be interpreted.
