"""Command-line interface for RoadVision AI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config.loader import ConfigError, load_config
from .dataset.build import build_dataset, generate_image
from .dataset.validate import DatasetValidationError, validate_all, validate_dataset
from .evaluation.runner import run_evaluation
from .ground_truth import load_annotation
from .harbor.exporter import export_harbor_dataset
from .providers import get_provider
from .questions import generate_tasks
from .reporting.export import export_csv_report, export_markdown_report
from .reporting.metrics import MetricsEngine
from .version import __version__


def _progress(index: int, total: int, image_id: str) -> None:
    print(f"[{index}/{total}] {image_id}", flush=True)


def cmd_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    artifacts = generate_image(config, args.seed, datasets_root=args.output)
    gt = artifacts.ground_truth
    print(f"image_id: {gt.image_id}")
    print(f"image: {artifacts.image_path}")
    print(f"annotation: {artifacts.annotation_path}")
    print(f"seed: {args.seed}")
    print(f"defects: {len(gt.objects)}")
    print(f"occluders: {len(gt.occluders)}")
    print(f"weather: {gt.scene.weather}")
    print(f"lighting: {gt.scene.lighting}")
    print(f"shadow: {gt.scene.shadow_present}")
    print(f"difficulty: {gt.scene.difficulty}")
    print(f"elapsed_sec: {artifacts.elapsed_sec}")
    if args.show_tasks:
        tasks = generate_tasks(gt, config)
        for task in tasks:
            print(
                f"task {task.task_id} [{task.capability.value}] "
                f"{task.answer_format.value}: {task.prompt}"
            )
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest = build_dataset(
        config,
        datasets_root=args.output,
        force=args.force,
        progress=_progress,
    )
    print(f"dataset: {manifest.name}")
    print(f"split: {manifest.split}")
    print(f"images: {len(manifest.images)}")
    print(f"tasks: {manifest.task_counts['total']}")
    print(f"capabilities: {json.dumps(manifest.task_counts_by_capability, sort_keys=True)}")
    print(f"difficulty: {json.dumps(manifest.task_counts_by_difficulty, sort_keys=True)}")
    print(f"manifest: datasets/manifests/{manifest.name}.json")
    return 0


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    if args.name == "all":
        reports = validate_all(args.datasets_root)
        for report in reports:
            print(json.dumps(report, sort_keys=True))
    else:
        report = validate_dataset(args.datasets_root, args.name)
        print(json.dumps(report, sort_keys=True))
    return 0


def cmd_show_annotation(args: argparse.Namespace) -> int:
    gt = load_annotation(args.path)
    print(json.dumps(gt.model_dump(mode="json"), indent=2, ensure_ascii=True))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    provider = get_provider(
        provider_type=args.provider,
        model_name=args.model,
        answers_file=args.answers_file,
    )
    print(f"Running evaluation on dataset '{args.dataset}' with provider '{args.provider}'...")
    session = run_evaluation(
        dataset_name=args.dataset,
        provider=provider,
        datasets_root=args.datasets_root,
        max_tasks=args.max_tasks,
    )
    print(f"Completed! Tasks: {session.total_tasks} | Correct: {session.correct_tasks} | Accuracy: {session.accuracy * 100:.2f}%")
    
    if args.report_dir:
        out_dir = Path(args.report_dir)
        md = export_markdown_report(session, out_dir / f"report_{session.model_name}_{session.dataset_name}.md")
        csv = export_csv_report(session, out_dir / f"report_{session.model_name}_{session.dataset_name}.csv")
        print(f"Reports saved to {md} and {csv}")
    return 0


def cmd_export_harbor(args: argparse.Namespace) -> int:
    print(f"Exporting dataset '{args.dataset}' to Harbor 1.4 format at '{args.output}'...")
    exported = export_harbor_dataset(
        dataset_name=args.dataset,
        datasets_root=args.datasets_root,
        output_dir=args.output,
        limit=args.limit,
    )
    print(f"Successfully exported {len(exported)} Harbor tasks.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    print(f"Starting RoadVision AI Platform at http://{args.host}:{args.port}")
    uvicorn.run("roadvision.api.app:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roadvision", description="RoadVision AI CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    generate = sub.add_parser("generate", help="Generate one deterministic image")
    generate.add_argument("--config", required=True, help="Path to YAML config")
    generate.add_argument("--seed", required=True, type=int, help="Image seed")
    generate.add_argument("--output", default="datasets", help="Datasets root directory")
    generate.add_argument("--show-tasks", action="store_true", help="Print generated questions")
    generate.set_defaults(func=cmd_generate)

    # build-dataset
    build = sub.add_parser("build-dataset", help="Build a full dataset from config")
    build.add_argument("--config", required=True, help="Path to YAML config")
    build.add_argument("--output", default="datasets", help="Datasets root directory")
    build.add_argument("--force", action="store_true", help="Overwrite an existing dataset")
    build.set_defaults(func=cmd_build_dataset)

    # validate-dataset
    validate = sub.add_parser("validate-dataset", help="Validate dataset files on disk")
    validate.add_argument("name", help="Dataset name or 'all'")
    validate.add_argument("--datasets-root", default="datasets", help="Datasets root directory")
    validate.set_defaults(func=cmd_validate_dataset)

    # show-annotation
    show = sub.add_parser("show-annotation", help="Print a ground-truth JSON file")
    show.add_argument("path", help="Path to annotation JSON")
    show.set_defaults(func=cmd_show_annotation)

    # evaluate
    evaluate = sub.add_parser("evaluate", help="Evaluate a multimodal model on a dataset")
    evaluate.add_argument("--dataset", required=True, help="Dataset name (e.g. smoke, dev, eval)")
    evaluate.add_argument("--provider", default="mock", help="Provider type (mock, manual, openai)")
    evaluate.add_argument("--model", default="roadvision-simulated-sota", help="Model name")
    evaluate.add_argument("--answers-file", help="Path to JSON/JSONL answers for manual provider")
    evaluate.add_argument("--max-tasks", type=int, help="Maximum tasks to evaluate")
    evaluate.add_argument("--datasets-root", default="datasets", help="Datasets root directory")
    evaluate.add_argument("--report-dir", default="reports", help="Directory to save reports")
    evaluate.set_defaults(func=cmd_evaluate)

    # export-harbor
    harbor = sub.add_parser("export-harbor", help="Export dataset into official Harbor 1.4 packages")
    harbor.add_argument("--dataset", required=True, help="Dataset name (e.g. smoke)")
    harbor.add_argument("--output", default="benchmarks/harbor", help="Harbor tasks destination directory")
    harbor.add_argument("--datasets-root", default="datasets", help="Datasets root directory")
    harbor.add_argument("--limit", type=int, help="Limit number of tasks to export")
    harbor.set_defaults(func=cmd_export_harbor)

    # serve
    serve = sub.add_parser("serve", help="Launch the RoadVision AI Web Platform & API")
    serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    serve.add_argument("--port", default=8000, type=int, help="Port to listen on")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, DatasetValidationError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
