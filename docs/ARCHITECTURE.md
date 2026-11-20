# RoadVision AI — System Architecture Specification

## 1. High-Level Architectural Overview

RoadVision AI is an end-to-end, reproducible, scientifically rigorous synthetic visual reasoning benchmark designed for multimodal AI systems inspecting transportation infrastructure.

```
+-----------------------------------------------------------------------------------------+
|                                    ROADVISION AI SUITE                                  |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [ Procedural Generator ] ---> [ Scene Plan & Physics ] ---> [ Multi-Octave Renderer ]  |
|            |                                                               |            |
|            v                                                               v            |
|  [ Strict Ground Truth ] <---------------------------------------- [ Pixel-Level Masks] |
|            |                                                                            |
|            v                                                                            |
|  [ Question Factory ] --------> [ Public Tasks (No Leakage) ] ----> [ Multimodal Models]|
|            |                                                                 |          |
|            v                                                                 v          |
|  [ Deterministic Verifier ] <---------------------------------------- [ Model Outputs ] |
|            |                                                                            |
|            v                                                                            |
|  [ Scientific Metrics Engine ] ---> [ Reports (MD/CSV) ] & [ Harbor 1.4 Exporter ]      |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
```

## 2. Core Subsystems

### 2.1. Procedural Generation & Rendering Engine (`src/roadvision/generator/` & `renderer/`)
- Pure procedural generation using NumPy, OpenCV, and Pillow.
- Multi-octave asphalt noise modeling, wet reflection puddles, projective perspective distortion, solar directional shadows, and irregular occlusion layers (cardboard, leaves, debris).
- Cryptographic RNG derivation ensures that a given `(seed, config_hash)` pair yields bit-identical RGB images and annotations.

### 2.2. Ground Truth & Geometry Engine (`src/roadvision/ground_truth/`)
- Built directly from painted mask pixels, not from bounding-box heuristics.
- Generates exact bounding boxes, normalized coordinates, pixel area, visible fraction, centroid, and occlusion status.
- Pydantic models with cross-field consistency validators (`_integrity`).

### 2.3. Question Factory & Semantic Capabilities (`src/roadvision/questions/`)
Evaluates 7 core multimodal capabilities:
1. **Counting**: Exact object counting across pavement defect classes.
2. **Classification**: Semantic damage identification.
3. **Localization**: Normalized bounding box coordinates scored via IoU.
4. **Spatial Reasoning**: Relative directional reasoning (left/right/above/below).
5. **Area Comparison**: Pairwise geometric area ranking.
6. **Occlusion Robustness**: Recognition under partial visual concealment.
7. **Evidence Sufficiency (Abstention)**: Detecting when visual evidence is insufficient.

### 2.4. Deterministic Verifier Engine (`src/roadvision/validators/`)
- **Zero LLM-as-a-judge dependency**: Ensures mathematical reproducibility and zero hallucination.
- Exact match normalization for counts and labels.
- Geometric IoU with configurable thresholds (default: 0.50).
- Abstention detection via semantic token analysis.

### 2.5. Model Providers & Multimodal Evaluation (`src/roadvision/providers/` & `evaluation/`)
- Provider abstraction supporting:
  - `DeterministicMockProvider`: Offline reproducible benchmarks across simulated model tiers.
  - `ManualFileProvider`: Evaluation from static pre-recorded answer files.
  - `OpenAICompatibleProvider`: Real multimodal API execution (GPT-4o, Claude, vLLM, Ollama).
- Strict benchmark isolation: Only public tasks and raw images are provided to models.

### 2.6. Harbor Framework Exporter (`src/roadvision/harbor/`)
- Compliant with **Harbor Framework (schema_version = "1.4")**.
- Emits task packages with `task.toml`, `instruction.md`, `tests/test.sh`, and `solution/solve.sh`.
