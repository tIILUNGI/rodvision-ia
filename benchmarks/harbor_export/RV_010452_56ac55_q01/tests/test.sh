#!/usr/bin/env bash
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

mode = "exact_int"
expected = 1
correct = False

def strip_noise(text):
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", text, flags=re.I)
    return text.strip().strip('"\' ')

cleaned = strip_noise(raw)

if mode == "exact_int":
    nums = re.findall(r"-?\d+", cleaned)
    correct = len(nums) > 0 and int(nums[0]) == int(expected)

elif mode in ("exact_label", "exact_choice"):
    canonical = re.sub(r"[\s\-_]+", " ", str(expected)).strip().lower()
    candidate = re.sub(r"[\s\-_]+", " ", cleaned).strip().lower()
    correct = canonical in candidate

elif mode == "bbox_iou":
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cleaned)]
    if len(nums) == 4:
        a = nums
        b = expected
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        iou = inter / union if union > 0 else 0.0
        correct = iou >= 0.5

elif mode == "abstention":
    is_abstain = any(p in cleaned.lower() for p in ("insufficient", "cannot", "unable", "abstain"))
    correct = is_abstain if expected == "insufficient" else not is_abstain

reward = "1.0" if correct else "0.0"
with open("/tmp/reward", "w") as f:
    f.write(reward)
EOF

exit 0
