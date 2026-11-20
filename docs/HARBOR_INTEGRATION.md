# RoadVision AI — Harbor Framework Integration Guide

## 1. Official Harbor Contract (`schema_version = "1.4"`)

RoadVision AI natively targets the official Harbor benchmark specification. Each generated task directory contains:

```
<output_dir>/<task_id>/
├── task.toml            # Official Harbor configuration
├── instruction.md       # Public instruction prompt for the model/agent
├── input/
│   └── image.png        # Input pavement image
├── tests/
│   └── test.sh          # Deterministic grader writing reward to /tmp/reward
├── solution/
│   └── solve.sh         # Oracle solver writing reference answer to output/answer.txt
└── Dockerfile           # Standard minimal test execution environment
```

## 2. Leakage Prevention Protocol
The evaluated model is executed inside the sandbox environment with access only to `instruction.md` and `input/image.png`.
Ground truth coordinates, exact defect counts, and internal object IDs are strictly confined to `tests/test.sh` and executed post-completion during grading.

## 3. Exporting Tasks via CLI

```bash
# Export smoke dataset tasks to Harbor 1.4 format
roadvision export-harbor --dataset smoke --output benchmarks/harbor_smoke

# Limit to top 10 tasks for quick trial
roadvision export-harbor --dataset smoke --output benchmarks/harbor_smoke --limit 10
```

## 4. Running Verification in Harbor

Once Harbor is invoked:
1. Harbor spins up the test container configured in `task.toml`.
2. The agent produces `output/answer.txt`.
3. `tests/test.sh` runs deterministically, parses the output, compares it against the ground truth criteria, and writes `1.0` (correct) or `0.0` (incorrect) to `/tmp/reward`.
