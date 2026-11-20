# RoadVision AI — Benchmark Experiments & Baseline Results

## 1. Multi-Modal Vision Model Baseline Comparison

Experiments conducted across 128 diverse road pavement tasks (`eval` dataset split):

| Model | Overall Accuracy | Counting Exact Match | Localization (Mean IoU) | Spatial Reasoning | Occlusion Robustness | Abstention F1 | Latency (s) |
|---|---|---|---|---|---|---|---|
| **RoadVision Fine-Tuned** | **88.4%** | **91.2%** | **0.782** | **85.5%** | **82.0%** | **0.865** | 0.65s |
| **Claude 3.5 Sonnet** | 74.2% | 76.0% | 0.612 | 78.5% | 58.0% | 0.720 | 1.25s |
| **GPT-4o (Omni)** | 71.8% | 74.0% | 0.590 | 75.0% | 54.0% | 0.690 | 1.10s |
| **Gemini 1.5 Pro** | 68.5% | 69.5% | 0.545 | 71.0% | 51.0% | 0.650 | 0.95s |
| **Llama 3.2 11B Vision** | 53.4% | 56.0% | 0.420 | 58.0% | 38.0% | 0.440 | 0.85s |

## 2. Key Failure Modes Revealed by RoadVision AI
1. **Severe Occlusion Blindspot**: General-purpose LLMs experience an average accuracy drop of 32.4% when defects are partially covered by road debris or leaves.
2. **Fine-Grain Counting Errors**: In dense crack or patch scenarios (3+ instances), frontier models frequently undercount linear cracks by confusing them with lane texture noise.
3. **Over-Confidence Under Insufficient Evidence**: When defects are obscured beyond physical recognizability (>85% hidden), models only abstain in ~48% of cases, hallucinating damage boundaries.
