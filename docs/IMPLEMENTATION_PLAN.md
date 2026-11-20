# RoadVision AI — Implementation Plan

**Window:** 20 November – 6 December 2026  
**Rule:** a phase is complete only when its command runs and its tests pass. No
phase is marked complete from a code review alone.

---

## Phase 1 — Bootstrap and environment (20–22 Nov)

**Objective:** a clean, reproducible repository and a validated environment.

- [x] Inspect OS, Python, Git, Docker, installed libraries, hardware.
- [x] Confirm no pre-existing project was overwritten.
- [x] Create isolated virtual environment and pin dependencies.
- [x] Create package layout, config layout, test layout.
- [x] Write `docs/TECHNICAL_SPEC.md` and this plan.
- [ ] Record environment in `docs/ENVIRONMENT.md` (generated from the actual
      venv).
- [ ] Initial commit.

**Acceptance:** `python -c "import roadvision"` works; `pytest` collects the
suite; versions are pinned.

**Environment facts (20 Nov 2026):** Windows 11, Python 3.14.3, Git 2.56.0,
Docker 29.2.0 (daemon stopped at inspection time), 4 cores / 8 threads,
15.7 GB RAM, 25 GB free. NumPy 2.5.2, Pillow 12.2.0, OpenCV 4.13.0.92,
Pydantic 2.14.2, PyYAML 6.0.3, pytest 8.4.2, pandas 3.1.1,
matplotlib 3.11.2. Harbor 0.26.0 is the current release and requires a separate
environment because it pulls agent/model dependencies.

---

## Phase 2 — Procedural image generation (23–25 Nov)

**Objective:** images that are varied, plausible and geometrically controlled.

- [ ] Deterministic RNG hierarchy and config hashing.
- [ ] Projective road plane and verge.
- [ ] Multi-octave asphalt texture; dry/damp/wet response.
- [ ] Pothole, crack, alligator crack, patch renderers with masks.
- [ ] Lighting, shadows, perspective, grain, vignette.
- [ ] Opaque occluders with independent masks.
- [ ] Visual inspection of a contact sheet.
- [ ] Reproducibility test: same seed → identical PNG bytes and annotation.

**Acceptance:** `roadvision generate --config configs/smoke.yaml --seed 10452`
produces an image containing at least a pothole and a crack, with masks.

**Dependencies:** Phase 1.

---

## Phase 3 — Ground truth (25–26 Nov)

**Objective:** a schema that is complete, internally consistent and validated.

- [ ] Pydantic models with strict types and allowed classes.
- [ ] Build bbox, normalized bbox, centroid, area, visible area, visibility.
- [ ] Save indexed instance/visible/occluder masks.
- [ ] Recompute statistics from masks and reject mismatches.
- [ ] Integrity tests: unique IDs, in-image boxes, mask consistency.
- [ ] JSON schema documentation.

**Acceptance:** generated annotation passes `validate_ground_truth` and a
corrupted annotation is rejected by tests.

**Dependencies:** Phase 2.

---

## Phase 4 — Question factory and validators (26–28 Nov)

**Objective:** questions whose answers are derivable and checkable.

- [ ] Deterministic candidate selection with ambiguity margins.
- [ ] Seven capability generators and semantic variants.
- [ ] Answer formats and reference answers.
- [ ] Parsers: integer, class, box, relation, choice, abstention.
- [ ] Positive and negative tests for every checker.
- [ ] Cross-check: every reference answer recomputed from the annotation.
- [ ] Public/internal task separation tests.

**Acceptance:** at least three valid tasks from the milestone image; wrong
answers fail; public tasks contain no answers.

**Dependencies:** Phase 3.

---

## Phase 5 — Dataset pipeline (28–29 Nov)

**Objective:** reproducible, leakage-free datasets.

- [ ] YAML configs: smoke, development, evaluation.
- [ ] Batch generation, manifest, split assignment.
- [ ] Grouped family splits; duplicate-image detection.
- [ ] Dataset validator and leakage tests.
- [ ] JSONL and CSV exports.
- [ ] Reproducibility test over a multi-image dataset.

**Acceptance:** smoke/dev/eval build; validation passes; an injected leakage is
detected; rebuilding a seed is byte-identical.

**Dependencies:** Phase 4.

---

## Phase 6 — Multimodal evaluation and metrics (29 Nov–1 Dec)

**Objective:** real model runs, honest metrics.

- [ ] Provider interface and OpenAI-compatible adapter.
- [ ] Timeout, retry, error logging, token usage, optional cost.
- [ ] Manual answer-file provider.
- [ ] Runner that sends only image/question/format.
- [ ] Result storage and resumable runs.
- [ ] Metric engine with denominators and condition breakdowns.
- [ ] Report JSON/CSV, failure list.
- [ ] Tests for adapter isolation and metrics with hand-computed fixtures.

**Acceptance:** with a configured endpoint, a small real run produces stored
answers and reports; with no key, the command fails clearly and generation still
works. No simulated model result appears in reports.

**Dependencies:** Phase 5.

---

## Phase 7 — Harbor integration (2–4 Dec)

**Objective:** real task packages in the official contract.

- [x] Read official docs and confirm the task contract (Harbor 0.26.0,
      `schema_version = "1.4"`).
- [ ] Export one task and a dataset directory.
- [ ] `tests/test.sh` writes a Harbor reward; `solution/solve.sh` is oracle-only.
- [ ] `harbor tasks lint` passes.
- [ ] Oracle run (requires Docker daemon) passes.
- [ ] Test that answers are absent from `instruction.md` and environment image.
- [ ] Document exact commands and remaining Docker requirement.

**Acceptance:** lint + oracle run. If Docker is unavailable, lint and structural
parity tests are completed and the blocked item is reported explicitly.

**Dependencies:** Phase 4 (task data) and Docker for the run.

---

## Phase 8 — Reporting, demo and submission (5–6 Dec)

**Objective:** a clear, honest demo and a complete submission.

- [ ] Streamlit audit view: image, boxes, masks, metadata, questions.
- [ ] Answer checker playground with positive/negative examples.
- [ ] Results browser with per-category breakdowns.
- [ ] Report export.
- [ ] Final README, architecture, schema, validation, Harbor and experiments
      docs.
- [ ] Dockerfile for the core pipeline.
- [ ] Full test run and final evidence table.
- [ ] 3–5 minute demo script.

**Acceptance:** a new machine can follow the README, run the smoke pipeline,
run tests, inspect a generated scene and (with credentials) run a model
evaluation.

---

## Risk register

| Risk | Mitigation |
| --- | --- |
| Renderer looks synthetic/geometric | Multi-octave texture, irregular masks, lighting, inspect contact sheets |
| Ground truth drifts from pixels | Recompute every statistic from saved masks; fail on mismatch |
| Ambiguous counting due to occlusion | Count only classes with all instances ≥95 % visible; avoid borderline values |
| LLM over-abstains | Paired answerable controls and abstention precision/recall |
| Dataset leakage | Disjoint seed blocks, family IDs, image hashes, validation tests |
| API unavailable | Generated/validated artefacts are offline; manual provider path |
| Docker daemon unavailable | Harbor lint + structural tests; document oracle run as blocked, not passed |
| Model answers hard to parse | Fixed answer formats and strict ambiguity rejection; raw response retained |
| Scope creep | Milestone gates; no dashboard before the pipeline |

---

## Definition of done (MVP)

1. One command generates an image, masks, annotation and ≥3 tasks from a seed.
2. Same seed/config reproduces identical output.
3. Validators accept correct answers and reject wrong/malformed ones.
4. Public tasks and provider requests contain no answers.
5. A multi-image dataset builds, validates and has disjoint splits.
6. Real API evaluation can run when credentials exist; otherwise the command
   fails clearly and offline generation still works.
7. Harbor task packages match the official format and pass lint.
8. README documents exact install/run commands.
9. Test results are reported from actual execution, including failures.
