# Validation Rules and Tolerances

This document defines the deterministic validators used by RoadVision AI. The
implementation lives in `src/roadvision/validators/core.py`.

## General principles

- Validation is **rule-based and deterministic**. No LLM is used for scoring.
- Parsers accept minor presentation noise (case, whitespace, punctuation, code
  fences, `Answer:` prefixes) but **do not guess** when the answer format is
  ambiguous.
- `is_abstention` is detected from natural refusal phrases and recorded in
  metrics, but it does **not** override the task's expected answer except for
  `abstention`-mode tasks where `insufficient` is correct.
- Scores are binary per task (`1.0` or `0.0`) unless a future task explicitly
  defines partial credit; none of the current milestone tasks do.

## Normalization

| Format | Accepted forms | Rejected / not guessed |
|---|---|---|
| Integer | `2`, `"2"`, `2.0`, `"Answer: 2"`, text containing exactly one integer token | booleans, multiple integers, free text without a unique count |
| Class label | canonical labels and aliases (`pothole`, `alligator crack`, `alligator_cracking`, `fatigue crack`, `patch`, etc.) as whole phrases | unknown labels, multiple conflicting class mentions |
| Choice | exact choice, case-insensitive match, letter index (`A`/`B`/`C`/`D` in declared order), relation aliases (`left`, `to the right`) | answers not mappable to a declared choice |
| Yes/No | `yes/no`, `true/false`, `1/0`, `present/absent` | other text |
| BBox | JSON list `[x1,y1,x2,y2]`, object with `x1..y2` or `bbox`, or four numbers in text; normalized `[0,1]` preferred | malformed boxes, wrong arity, non-finite values |
| Abstention | `insufficient`, `unknown`, `cannot determine`, `not enough evidence`, `abstain`, `n/a` and similar phrases | — |

## Scoring rules

| Verification mode | Correct when | Tolerance / threshold |
|---|---|---|
| `exact_int` | parsed integer equals reference | exact match |
| `exact_label` | parsed canonical class equals reference | exact after normalization |
| `exact_choice` | parsed choice equals reference | exact after normalization |
| `exact_bool` | parsed boolean equals reference | exact after normalization |
| `bbox_iou` | IoU(pred, ref) ≥ threshold | **IoU ≥ 0.50** |
| `abstention` | parsed state matches expected sufficiency (`sufficient` / `insufficient`) | exact state |

## Bounding boxes

- Reference boxes are **visible-object** normalized boxes `[x1, y1, x2, y2]` in
  image coordinates with origin at top-left, `x1 < x2`, `y1 < y2`.
- Pixel-coordinate predictions are accepted only when the verifier knows the
  image width/height; they are converted to normalized coordinates before IoU.
- IoU is computed on axis-aligned boxes with standard intersection-over-union.

## Negative-case expectations

Validators must reject or score incorrect:

- wrong counts/classes/choices/bools;
- unparseable or out-of-format answers;
- bounding boxes with IoU below threshold;
- abstentions on tasks where evidence is sufficient;
- confident answers on tasks where evidence is insufficient.

## Leakage checks

Public task payloads must not contain: `reference_answer`, `verification`,
`related_object_ids`, `visual_conditions`, or `defect_types`. The task model
also rejects prompts that embed the integer or yes/no answer token directly.
Choice prompts necessarily list options, so option strings appearing in the
prompt are not treated as leakage.
