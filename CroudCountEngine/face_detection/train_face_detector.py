"""Train a high-accuracy, checkpoint-safe face detector with Ultralytics YOLO."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_YAML = PROJECT_ROOT / "generated" / "face_detection_yolo" / "face_dataset.yaml"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "face_detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or safely resume the face detector.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_YAML)
    parser.add_argument("--model", default="yolo11l.pt", help="Pretrained starting weights for a new run.")
    parser.add_argument("--run-name", default="face_yolo11l_1024")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--resume", default="auto", help="'auto' (default), 'never', or an explicit last.pt path.")
    parser.add_argument("--epochs", type=int, default=300, help="Total epochs for a new run; choose this before starting.")
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Stop a new run after this many epochs without validation-metric improvement.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="High resolution helps with small WIDER FACE boxes.")
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="-1 selects a conservative GPU batch size automatically from the detected VRAM.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help=(
            "CPU data-loader processes. -1 (default) uses about half of the available logical CPU cores; "
            "0 disables parallel loading."
        ),
    )
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0. Defaults to the first CUDA GPU.")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit slow CPU training explicitly.")
    parser.add_argument("--skip-ufdd-eval", action="store_true")
    return parser.parse_args()


def choose_device(requested_device: str | None, allow_cpu: bool) -> str | int:
    if requested_device is not None:
        return requested_device
    if torch.cuda.is_available():
        return 0
    if allow_cpu:
        return "cpu"
    raise RuntimeError(
        "No CUDA GPU is available. CPU training at 1024px with YOLO11l is impractically slow. "
        "Use a CUDA-enabled machine, or pass --allow-cpu only if you accept the long runtime."
    )


def detected_cuda_memory_gib(device: str | int) -> float | None:
    if not torch.cuda.is_available():
        return None
    if isinstance(device, str) and device.lower() == "cpu":
        return None
    index = int(device)
    properties = torch.cuda.get_device_properties(index)
    return properties.total_memory / (1024**3)


def choose_safe_batch(requested_batch: int, device: str | int) -> int:
    if requested_batch != -1:
        return requested_batch
    if isinstance(device, str) and device.lower() == "cpu":
        return 1
    memory_gib = detected_cuda_memory_gib(device)
    if memory_gib is None:
        return 1
    if memory_gib < 10:
        return 2
    if memory_gib < 14:
        return 4
    if memory_gib < 20:
        return 8
    if memory_gib < 28:
        return 12
    return 16


def choose_data_workers(requested_workers: int) -> int:
    """Keep image decoding/augmentation within a conservative CPU budget.

    The GPU still performs all model forward/backward work.  These workers only
    read images and apply augmentations, so reserving roughly half the logical
    CPU cores leaves the rest of the machine responsive without moving slow
    neural-network computation to the CPU.
    """
    if requested_workers == -1:
        logical_cores = os.cpu_count() or 2
        return max(1, logical_cores // 2)
    if requested_workers < 0:
        raise ValueError("--workers must be -1 (automatic) or a non-negative integer.")
    return requested_workers


def resolve_resume_checkpoint(resume_value: str, runs_root: Path, run_name: str) -> Path | None:
    if resume_value.lower() == "never":
        return None
    if resume_value.lower() == "auto":
        candidate = runs_root / run_name / "weights" / "last.pt"
        return candidate if candidate.is_file() else None
    checkpoint = Path(resume_value).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    if checkpoint.name != "last.pt":
        print("Warning: best.pt starts fine-tuning, not an exact resume; use last.pt to continue safely.")
    return checkpoint


def run_new_training(args: argparse.Namespace, device: str | int, data_workers: int) -> Path:
    if args.epochs < 2:
        raise ValueError("--epochs must be at least 2 for a meaningful training run.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    print(f"Starting from pretrained weights: {args.model}")
    print(f"Checkpoints: {args.runs_root / args.run_name / 'weights'}")
    effective_batch = choose_safe_batch(args.batch, device)
    memory_gib = detected_cuda_memory_gib(device)
    if memory_gib is not None and args.batch == -1:
        print(f"Detected CUDA memory: {memory_gib:.1f} GiB; using batch={effective_batch}.")
    elif args.batch != -1:
        print(f"Using manually requested batch={effective_batch}.")
    logical_cores = os.cpu_count() or 1
    if args.workers == -1:
        print(f"CPU data pipeline: workers={data_workers} of {logical_cores} logical cores (automatic half-core cap).")
    else:
        print(f"CPU data pipeline: workers={data_workers} (manually requested).")
    model = YOLO(args.model)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=effective_batch, device=device,
        workers=data_workers, project=str(args.runs_root), name=args.run_name, exist_ok=True,
        pretrained=True, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=3.0,
        patience=args.patience, cache="disk", seed=42, deterministic=True, save=True, save_period=1, val=True, plots=True,
        # Moderate augmentation preserves tiny/occluded faces while adding camera robustness.
        hsv_h=0.01, hsv_s=0.45, hsv_v=0.25, translate=0.05, scale=0.35, fliplr=0.5,
        mosaic=0.25, close_mosaic=20, mixup=0.0, copy_paste=0.0, erasing=0.0,
    )
    return args.runs_root / args.run_name / "weights" / "best.pt"


def resume_training(checkpoint: Path, device: str | int) -> Path:
    print(f"Resuming from the last completed epoch: {checkpoint}")
    print("The saved optimizer, scheduler, EMA, data split, and augmentations are restored unchanged.")
    model = YOLO(str(checkpoint))
    # Passing no new hyperparameters is intentional: it preserves the original run exactly.
    model.train(resume=str(checkpoint), device=device)
    return checkpoint.parent / "best.pt"


def evaluate_on_ufdd(
    best_weights: Path,
    data_yaml: Path,
    device: str | int,
    runs_root: Path,
    run_name: str,
    imgsz: int,
    data_workers: int,
) -> None:
    if not best_weights.is_file():
        print("No best.pt was produced, so UFDD evaluation was skipped.")
        return
    print(f"Evaluating on held-out UFDD: {best_weights}")
    YOLO(str(best_weights)).val(
        data=str(data_yaml), split="test", imgsz=imgsz, device=device,
        batch=1 if str(device).lower() == "cpu" else -1, workers=data_workers,
        project=str(runs_root / run_name), name="ufdd_evaluation", exist_ok=True, plots=True,
    )


def main() -> int:
    args = parse_args()
    args.data, args.runs_root = args.data.resolve(), args.runs_root.resolve()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {args.data}\nRun prepare_face_dataset.py first.")
    device = choose_device(args.device, args.allow_cpu)
    data_workers = choose_data_workers(args.workers)
    checkpoint = resolve_resume_checkpoint(args.resume, args.runs_root, args.run_name)
    best_weights = resume_training(checkpoint, device) if checkpoint else run_new_training(args, device, data_workers)
    if not args.skip_ufdd_eval:
        evaluate_on_ufdd(best_weights, args.data, device, args.runs_root, args.run_name, args.imgsz, data_workers)
    print(f"Training workflow complete. Deployment weights: {best_weights}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise SystemExit(1)
