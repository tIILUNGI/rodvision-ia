"""Harbor Framework (schema_version = "1.4") task exporter.

Exports RoadVision AI tasks into standard Harbor-compatible task packages.
Each exported task directory includes:
- `task.toml`: task metadata and configuration
- `instruction.md`: public instructions provided to the agent/model
- `input/`: the input image without ground-truth metadata
- `tests/test.sh`: deterministic scoring script that writes reward to Harbor contract
- `solution/solve.sh`: oracle reference solution for task verification
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..questions.models import Task


def _generate_test_sh(task: Task) -> str:
    """Generate deterministic test script matching Harbor grading contract."""
    mode = task.verification.mode
    val = task.verification.value
    expected_repr = json.dumps(val)

    return f"""#!/usr/bin/env bash
# RoadVision AI deterministic validator for Harbor
set -euo pipefail

ANSWER_FILE="output/answer.txt"
REWARD_FILE="/tmp/reward"

if [ ! -f "$ANSWER_FILE" ]; then
    echo "0.0" > "$REWARD_FILE"
    echo "Missing $ANSWER_FILE"
    exit 0
fi

python3 - << 'EOF'
import json, re, sys

with open("output/answer.txt", "r") as f:
    raw = f.read().strip()

mode = "{mode}"
expected = {expected_repr}
correct = False

def strip_noise(text):
    text = re.sub(r"^```[a-zA-Z0-9_-]*\\s*", "", text)
    text = re.sub(r"\\s*```$", "", text)
    text = re.sub(r"^(answer|final answer)\\s*[:\\-]\\s*", "", text, flags=re.I)
    return text.strip().strip('"\\\' ')

cleaned = strip_noise(raw)

if mode == "exact_int":
    nums = re.findall(r"-?\\d+", cleaned)
    correct = len(nums) > 0 and int(nums[0]) == int(expected)

elif mode in ("exact_label", "exact_choice"):
    canonical = re.sub(r"[\\s\\-_]+", " ", str(expected)).strip().lower()
    candidate = re.sub(r"[\\s\\-_]+", " ", cleaned).strip().lower()
    correct = canonical in candidate

elif mode == "bbox_iou":
    nums = [float(x) for x in re.findall(r"-?\\d+(?:\\.\\d+)?", cleaned)]
    if len(nums) == 4:
        a = nums
        b = expected
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        iou = inter / union if union > 0 else 0.0
        correct = iou >= {task.verification.iou_threshold}

elif mode == "abstention":
    is_abstain = any(p in cleaned.lower() for p in ("insufficient", "cannot", "unable", "abstain"))
    correct = is_abstain if expected == "insufficient" else not is_abstain

reward = "1.0" if correct else "0.0"
with open("/tmp/reward", "w") as f:
    f.write(reward)
EOF

exit 0
"""


def _generate_solve_sh(task: Task) -> str:
    """Generate oracle reference solution for Harbor task validation."""
    ref_val = task.reference_answer
    if isinstance(ref_val, (list, dict)):
        ans_str = json.dumps(ref_val)
    else:
        ans_str = str(ref_val)

    return f"""#!/usr/bin/env bash
mkdir -p output
cat << 'EOF' > output/answer.txt
{ans_str}
EOF
"""


def export_harbor_task(
    task: Task,
    source_datasets_root: Path | str,
    output_harbor_dir: Path | str,
) -> Path:
    """Export a single RoadVision task into an official Harbor 1.4 package."""
    root = Path(source_datasets_root)
    dest = Path(output_harbor_dir) / task.task_id
    dest.mkdir(parents=True, exist_ok=True)

    input_dir = dest / "input"
    tests_dir = dest / "tests"
    solution_dir = dest / "solution"
    input_dir.mkdir(exist_ok=True)
    tests_dir.mkdir(exist_ok=True)
    solution_dir.mkdir(exist_ok=True)

    # 1. Copy image to input/image.png
    src_img = root / task.image_path
    if src_img.exists():
        shutil.copy2(src_img, input_dir / "image.png")

    # 2. task.toml
    task_toml = f"""# Harbor Task Definition
schema_version = "1.4"
title = "RoadVision AI - {task.capability.value.replace('_', ' ').title()}"
description = "Evaluate multimodal visual reasoning on road pavement defects."

[metadata]
task_id = "{task.task_id}"
capability = "{task.capability.value}"
difficulty = "{task.difficulty}"
answer_format = "{task.answer_format.value}"

[environment]
dockerfile = "Dockerfile"
timeout_sec = 60
"""
    (dest / "task.toml").write_text(task_toml, encoding="utf-8")

    # 3. instruction.md (STRICT BENCHMARK ISOLATION: No ground truth!)
    choices_text = ""
    if task.choices:
        choices_text = "\nOptions:\n" + "\n".join(f"- `{c}`" for c in task.choices)

    instruction_md = f"""# Visual Road Inspection Task

Analyze the provided road image (`input/image.png`) and answer the question below.

### Question
**{task.prompt}**
{choices_text}

### Output Requirements
- Format: `{task.answer_format.value}`
- Guidance: {task.answer_hint}
- Save your final concise answer into `output/answer.txt`.
"""
    (dest / "instruction.md").write_text(instruction_md, encoding="utf-8")

    # 4. tests/test.sh
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(_generate_test_sh(task), encoding="utf-8")

    # 5. solution/solve.sh (Oracle)
    solve_sh_path = solution_dir / "solve.sh"
    solve_sh_path.write_text(_generate_solve_sh(task), encoding="utf-8")

    # 6. Dockerfile for Harbor test container
    dockerfile_content = """FROM python:3.11-slim
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends bash && rm -rf /var/lib/apt/lists/*
"""
    (dest / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")

    return dest


def export_harbor_dataset(
    dataset_name: str,
    datasets_root: Path | str,
    output_dir: Path | str,
    limit: int | None = None,
) -> list[Path]:
    """Export an entire dataset into a directory of Harbor tasks."""
    root = Path(datasets_root)
    tasks_file = root / f"tasks/{dataset_name}/tasks.jsonl"
    if not tasks_file.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")

    exported: list[Path] = []
    with tasks_file.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                task = Task.model_validate_json(line)
                path = export_harbor_task(task, root, output_dir)
                exported.append(path)

    return exported
