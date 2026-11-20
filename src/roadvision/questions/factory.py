"""Deterministic question factory.

Every question is generated from the ground-truth record only. No model is
consulted, no random sampling is used, and answers are computed from the same
pixel masks that produced the image.
"""

from __future__ import annotations

from typing import Callable

from ..config.models import Capability, DefectClass, RoadVisionConfig
from ..ground_truth.models import GroundTruth, ObjectInfo
from .models import AnswerFormat, Task, Verification

CLASS_ORDER = [c.value for c in DefectClass]

DIFFICULTY: dict[Capability, str] = {
    Capability.COUNTING: "easy",
    Capability.CLASSIFICATION: "easy",
    Capability.LOCALIZATION: "medium",
    Capability.SPATIAL_RELATION: "medium",
    Capability.AREA_COMPARISON: "medium",
    Capability.OCCLUSION: "hard",
    Capability.EVIDENCE_SUFFICIENCY: "hard",
}

PRETTY = {
    "pothole": "pothole",
    "crack": "crack",
    "alligator_crack": "alligator crack",
    "patch": "patch",
}

PLURAL = {
    "pothole": "potholes",
    "crack": "cracks",
    "alligator_crack": "alligator cracks",
    "patch": "patches",
}


def _visible(gt: GroundTruth) -> list[ObjectInfo]:
    return [o for o in gt.objects if o.visible_area_px > 0]


def _largest(objects: list[ObjectInfo]) -> ObjectInfo:
    return max(objects, key=lambda o: (o.visible_area_px, o.id))


def _by_class(objects: list[ObjectInfo]) -> dict[str, list[ObjectInfo]]:
    out: dict[str, list[ObjectInfo]] = {}
    for obj in objects:
        out.setdefault(obj.type.value, []).append(obj)
    return out


def _largest_of(objects: list[ObjectInfo], class_name: str) -> ObjectInfo | None:
    subset = [o for o in objects if o.type.value == class_name]
    return _largest(subset) if subset else None


def _conditions(gt: GroundTruth) -> dict[str, str]:
    return {
        "weather": gt.scene.weather,
        "lighting": gt.scene.lighting,
        "shadow": "present" if gt.scene.shadow_present else "absent",
        "occluder": gt.occluders[0].type if gt.occluders else "none",
        "difficulty": gt.scene.difficulty,
        "surface": gt.scene.surface,
    }


class _Builder:
    def __init__(self, gt: GroundTruth, config: RoadVisionConfig) -> None:
        self.gt = gt
        self.config = config
        self.image_path = gt.image.path
        self.split = config.dataset.split
        self.conditions = _conditions(gt)
        self.tasks: list[tuple[Capability, Callable[[], Task | None]]] = [
            (Capability.COUNTING, self.counting),
            (Capability.CLASSIFICATION, self.classification),
            (Capability.LOCALIZATION, self.localization),
            (Capability.SPATIAL_RELATION, self.spatial_relation),
            (Capability.AREA_COMPARISON, self.area_comparison),
            (Capability.OCCLUSION, self.occlusion),
            (Capability.EVIDENCE_SUFFICIENCY, self.evidence_sufficiency),
        ]

    def _task(
        self,
        index: int,
        capability: Capability,
        family: str,
        prompt: str,
        answer_format: AnswerFormat,
        reference: object,
        verification: Verification,
        answer_hint: str,
        related: list[ObjectInfo] | ObjectInfo | None = None,
        choices: list[str] | None = None,
    ) -> Task:
        if isinstance(related, ObjectInfo):
            related_objects = [related]
        else:
            related_objects = related or []
        task = Task(
            task_id=f"{self.gt.image_id}_q{index:02d}",
            image_id=self.gt.image_id,
            image_path=self.image_path,
            split=self.split,
            capability=capability,
            difficulty=DIFFICULTY[capability],
            question_family=family,
            prompt=prompt,
            answer_format=answer_format,
            choices=choices,
            answer_hint=answer_hint,
            reference_answer=reference,
            verification=verification,
            related_object_ids=[o.id for o in related_objects],
            visual_conditions=self.conditions,
            defect_types=sorted({o.type.value for o in related_objects}),
        )
        task.assert_no_leakage()
        return task

    # -- individual families ------------------------------------------------

    def counting(self) -> Task | None:
        visible = _visible(self.gt)
        if not visible:
            return None
        groups = _by_class(visible)
        class_name = max(
            CLASS_ORDER,
            key=lambda name: (len(groups.get(name, [])), -CLASS_ORDER.index(name)),
        )
        count = len(groups.get(class_name, []))
        plural = PLURAL[class_name]
        prompt = (
            f"How many {plural} are at least partially visible in this image? "
            "Answer with a single integer."
        )
        return self._task(
            0,
            Capability.COUNTING,
            "count_visible_by_class",
            prompt,
            AnswerFormat.INTEGER,
            count,
            Verification(mode="exact_int", value=count),
            "A single non-negative integer.",
            related=groups.get(class_name, []),
        )

    def classification(self) -> Task | None:
        visible = _visible(self.gt)
        if not visible:
            return None
        target = _largest(visible)
        prompt = (
            "What type of road defect covers the largest visible area in this image? "
            "Choose one class from the options."
        )
        return self._task(
            0,
            Capability.CLASSIFICATION,
            "largest_visible_class",
            prompt,
            AnswerFormat.CLASS_LABEL,
            target.type.value,
            Verification(mode="exact_label", value=target.type.value),
            "One class name from the options.",
            related=target,
            choices=list(CLASS_ORDER),
        )

    def localization(self) -> Task | None:
        visible = _visible(self.gt)
        if not visible:
            return None
        target = _largest(visible)
        label = PRETTY[target.type.value]
        prompt = (
            f"Return the bounding box of the largest {label} in the image. "
            "Use normalized coordinates with the origin at the top-left corner."
        )
        return self._task(
            0,
            Capability.LOCALIZATION,
            "bbox_largest_visible",
            prompt,
            AnswerFormat.BBOX_NORMALIZED,
            list(target.bbox_visible_normalized),
            Verification(
                mode="bbox_iou",
                value=list(target.bbox_visible_normalized),
                iou_threshold=0.5,
                image_width=self.gt.image.width,
                image_height=self.gt.image.height,
            ),
            "JSON list [x1, y1, x2, y2] with values in [0, 1], x1<x2 and y1<y2.",
            related=target,
        )

    def spatial_relation(self) -> Task | None:
        visible = _visible(self.gt)
        if len(visible) < 2:
            return None
        groups = _by_class(visible)
        pairs: list[tuple[ObjectInfo, ObjectInfo]] = []
        for i, first in enumerate(CLASS_ORDER):
            for second in CLASS_ORDER[i + 1 :]:
                a = _largest_of(visible, first)
                b = _largest_of(visible, second)
                if a is not None and b is not None:
                    pairs.append((a, b))
        if not pairs:
            return None
        # Prefer a pair whose relation is unambiguous on one axis.
        best: tuple[float, str, ObjectInfo, ObjectInfo] | None = None
        for a, b in pairs:
            dx = a.centroid_normalized[0] - b.centroid_normalized[0]
            dy = a.centroid_normalized[1] - b.centroid_normalized[1]
            if abs(dx) >= 1.6 * abs(dy) and abs(dx) >= 0.12:
                relation = "right of" if dx > 0 else "left of"
                strength = abs(dx)
            elif abs(dy) >= 1.6 * abs(dx) and abs(dy) >= 0.10:
                relation = "below" if dy > 0 else "above"
                strength = abs(dy)
            else:
                continue
            if best is None or strength > best[0]:
                best = (strength, relation, a, b)
        if best is None:
            return None
        _, relation, a, b = best
        prompt = (
            f"Is the largest {PRETTY[a.type.value]} to the left of, right of, "
            f"above, or below the largest {PRETTY[b.type.value]}? "
            "Answer with exactly one option."
        )
        choices = ["left of", "right of", "above", "below"]
        return self._task(
            0,
            Capability.SPATIAL_RELATION,
            "relative_position",
            prompt,
            AnswerFormat.CHOICE,
            relation,
            Verification(mode="exact_choice", value=relation),
            "One of: left of, right of, above, below.",
            related=[a, b],
            choices=choices,
        )

    def area_comparison(self) -> Task | None:
        visible = _visible(self.gt)
        groups = _by_class(visible)
        candidates = [
            (name, _largest(group)) for name, group in groups.items() if group
        ]
        if len(candidates) < 2:
            return None
        candidates.sort(key=lambda item: item[1].visible_area_px, reverse=True)
        (name_a, obj_a), (name_b, obj_b) = candidates[0], candidates[1]
        small = max(1, min(obj_a.visible_area_px, obj_b.visible_area_px))
        ratio = max(obj_a.visible_area_px, obj_b.visible_area_px) / small
        if ratio < 1.35:
            return None
        choices = sorted([name_a, name_b])
        larger = name_a if obj_a.visible_area_px >= obj_b.visible_area_px else name_b
        prompt = (
            "Which defect covers the larger visible area: the largest "
            f"{PRETTY[name_a]} or the largest {PRETTY[name_b]}?"
        )
        return self._task(
            0,
            Capability.AREA_COMPARISON,
            "larger_visible_area",
            prompt,
            AnswerFormat.CHOICE,
            larger,
            Verification(mode="exact_choice", value=larger),
            "One class name from the options.",
            related=[obj_a, obj_b],
            choices=choices,
        )

    def occlusion(self) -> Task | None:
        partially = [o for o in self.gt.objects if 0.15 <= o.visible_fraction < 0.95]
        severe = [
            o for o in self.gt.objects if 0.0 < o.visible_fraction < 0.15
        ]
        if severe:
            prompt = (
                "How many defects are severely occluded, meaning less than 15% of "
                "the defect area is still visible?"
            )
            count = len(severe)
            family = "count_severely_occluded"
            related = severe
        elif partially:
            prompt = (
                "How many defects are partially occluded, meaning at least 15% but "
                "less than 95% of the defect area is visible?"
            )
            count = len(partially)
            family = "count_partially_occluded"
            related = partially
        else:
            return None
        return self._task(
            0,
            Capability.OCCLUSION,
            family,
            prompt,
            AnswerFormat.INTEGER,
            count,
            Verification(mode="exact_int", value=count),
            "A single non-negative integer.",
            related=related,
        )

    def evidence_sufficiency(self) -> Task | None:
        if not self.config.questions.include_abstention_controls:
            return None
        severe = [o for o in self.gt.objects if 0.0 < o.visible_fraction < 0.15]
        if severe:
            occluder = self.gt.occluders[0] if self.gt.occluders else None
            cover = PRETTY.get(occluder.type, "object") if occluder else "object"
            if occluder is not None and occluder.type == "debris_sheet":
                cover = "debris sheet"
            elif occluder is not None and occluder.type == "leaf_litter":
                cover = "leaf litter"
            prompt = (
                f"Is there enough visible evidence to determine the full extent of "
                f"the defect under the {cover}? Choose 'sufficient' or 'insufficient'."
            )
            return self._task(
                0,
                Capability.EVIDENCE_SUFFICIENCY,
                "extent_under_occluder",
                prompt,
                AnswerFormat.ABSTAIN_LABEL,
                "insufficient",
                Verification(mode="abstention", value="insufficient"),
                "One of: sufficient, insufficient.",
                related=severe,
                choices=["sufficient", "insufficient"],
            )
        if all(o.visible_fraction >= 0.95 for o in self.gt.objects) and not self.gt.occluders:
            target = _largest(self.gt.objects)
            prompt = (
                "Is the full extent of the largest defect visible in this image, "
                "with no part hidden? Choose 'sufficient' or 'insufficient'."
            )
            return self._task(
                0,
                Capability.EVIDENCE_SUFFICIENCY,
                "extent_fully_visible",
                prompt,
                AnswerFormat.ABSTAIN_LABEL,
                "sufficient",
                Verification(mode="abstention", value="sufficient"),
                "One of: sufficient, insufficient.",
                related=target,
                choices=["sufficient", "insufficient"],
            )
        return None


def generate_tasks(gt: GroundTruth, config: RoadVisionConfig) -> list[Task]:
    builder = _Builder(gt, config)
    allowed = set(config.questions.capabilities)
    selected: list[tuple[Capability, Callable[[], Task | None]]] = [
        item for item in builder.tasks if item[0] in allowed
    ]
    produced: list[Task] = []
    for index, (_, factory) in enumerate(selected, start=1):
        task = factory()
        if task is None:
            continue
        produced.append(task.model_copy(update={"task_id": f"{gt.image_id}_q{index:02d}"}))
        if len(produced) >= config.questions.max_per_image:
            break
    if len(produced) < config.questions.min_questions:
        raise ValueError(
            f"{gt.image_id}: generated {len(produced)} questions, "
            f"minimum is {config.questions.min_questions}"
        )
    return produced
