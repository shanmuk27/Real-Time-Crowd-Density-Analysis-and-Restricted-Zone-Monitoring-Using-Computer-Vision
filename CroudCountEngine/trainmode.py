import argparse
import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image, ImageFile
import torch
from ultralytics import YOLO


DATASET_ROOT = Path(r"C:\Crowd Count Dataset")
REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARED_ROOT = REPO_ROOT / "generated" / "nwpu_yolo"
RUNS_ROOT = REPO_ROOT / "runs"
PREPARE_VERSION = 3

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_CPU_THREADS = max(1, os.cpu_count() or 1)
DEFAULT_PREP_WORKERS = max(2, min(8, DEFAULT_CPU_THREADS))
DEFAULT_TRAIN_WORKERS = max(2, min(8, DEFAULT_CPU_THREADS // 2 or 1))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the NWPU-Crowd dataset once and train a YOLO model manually."
    )
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="NWPU dataset root.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO weights to fine-tune.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device override such as cpu, 0, or 0,1.")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_TRAIN_WORKERS,
        help="Data loader worker count for training.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help="CPU thread count for PyTorch when training on CPU.",
    )
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=DEFAULT_PREP_WORKERS,
        help="Thread count for one-time dataset preparation.",
    )
    parser.add_argument(
        "--cache",
        default="disk",
        choices=["False", "ram", "disk"],
        help="Dataset caching mode for training. 'disk' is a good default for large datasets.",
    )
    parser.add_argument(
        "--project",
        default=str(RUNS_ROOT),
        help="Ultralytics project directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        default="crowd_train",
        help="Ultralytics run name. The best weights will be saved under this folder.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rebuild the generated YOLO label files even if they already exist.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Train again even if best weights already exist.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint if it exists.",
    )
    parser.add_argument(
        "--cpu-utilization",
        choices=["balanced", "aggressive"],
        default="aggressive",
        help="How hard to push CPU training resources.",
    )
    return parser.parse_args()


def read_split_ids(split_file: Path):
    ids = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            ids.append(stripped.split()[0])
    return ids


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_image_path(dataset_root: Path, image_id: str):
    image_number = int(image_id)
    preferred_part = min(((image_number - 1) // 1000) + 1, 5)
    candidate = dataset_root / f"images_part{preferred_part}" / f"{image_id}.jpg"
    if candidate.exists():
        return candidate

    for part_index in range(1, 6):
        candidate = dataset_root / f"images_part{part_index}" / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for id {image_id}")


def get_image_shape(image_path: Path):
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            return width, height
    except Exception:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unable to read image {image_path}")
        height, width = image.shape[:2]
        return width, height


def ensure_image_link(source_path: Path, destination_path: Path):
    ensure_parent_dir(destination_path)
    if destination_path.exists():
        return
    try:
        destination_path.hardlink_to(source_path)
    except OSError:
        try:
            destination_path.symlink_to(source_path)
        except OSError:
            import shutil

            shutil.copy2(source_path, destination_path)


def normalize_box(xmin, ymin, xmax, ymax, image_width: int, image_height: int):
    xmin = min(max(float(xmin), 0.0), float(image_width))
    ymin = min(max(float(ymin), 0.0), float(image_height))
    xmax = min(max(float(xmax), 0.0), float(image_width))
    ymax = min(max(float(ymax), 0.0), float(image_height))
    if xmax <= xmin or ymax <= ymin:
        return None

    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + (box_width / 2.0)
    y_center = ymin + (box_height / 2.0)
    return (
        min(max(x_center / image_width, 0.0), 1.0),
        min(max(y_center / image_height, 0.0), 1.0),
        min(max(box_width / image_width, 0.0), 1.0),
        min(max(box_height / image_height, 0.0), 1.0),
    )


def convert_boxes_to_yolo(boxes, image_width: int, image_height: int):
    unique_rows = set()
    for xmin, ymin, xmax, ymax in boxes:
        normalized = normalize_box(xmin, ymin, xmax, ymax, image_width, image_height)
        if normalized is None:
            continue
        x_center, y_center, box_width, box_height = normalized
        unique_rows.add(
            "0 "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )
    return sorted(unique_rows)


def process_image_id(dataset_root: Path, images_dir: Path, labels_dir: Path, image_id: str):
    source_image_path = resolve_image_path(dataset_root, image_id)
    prepared_image_path = images_dir / f"{image_id}.jpg"
    label_path = labels_dir / f"{image_id}.txt"
    annotation_path = dataset_root / "jsons" / f"{image_id}.json"

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    boxes = annotation.get("boxes", [])
    image_width, image_height = get_image_shape(source_image_path)
    yolo_rows = convert_boxes_to_yolo(boxes, image_width, image_height)
    label_path.write_text("\n".join(yolo_rows), encoding="utf-8")
    ensure_image_link(source_image_path, prepared_image_path)


def build_split(dataset_root: Path, prepared_root: Path, split_name: str, prep_workers: int):
    split_ids = read_split_ids(dataset_root / f"{split_name}.txt")
    labels_dir = prepared_root / "labels" / split_name
    images_dir = prepared_root / "images" / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, prep_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_image_id, dataset_root, images_dir, labels_dir, image_id)
            for image_id in split_ids
        ]
        for future in futures:
            future.result()

    return len(split_ids)


def write_dataset_yaml(prepared_root: Path):
    dataset_yaml = prepared_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {prepared_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_yaml


def write_manifest(prepared_root: Path, dataset_root: Path, train_count: int, val_count: int):
    manifest_path = prepared_root / "prepare_manifest.json"
    manifest = {
        "prepare_version": PREPARE_VERSION,
        "dataset_root": str(dataset_root),
        "train_images": train_count,
        "val_images": val_count,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def prepare_dataset(dataset_root: Path, prepared_root: Path, force_prepare: bool, prep_workers: int):
    manifest_path = prepared_root / "prepare_manifest.json"
    dataset_yaml = prepared_root / "dataset.yaml"
    manifest = load_manifest(manifest_path)
    if (
        not force_prepare
        and manifest
        and manifest.get("prepare_version") == PREPARE_VERSION
        and dataset_yaml.exists()
    ):
        print(f"[prepare] Reusing cached prepared dataset from {prepared_root}")
        return dataset_yaml

    prepared_root.mkdir(parents=True, exist_ok=True)
    print("[prepare] Building YOLO labels and linked image structure from NWPU JSON annotations")
    train_count = build_split(dataset_root, prepared_root, "train", prep_workers)
    val_count = build_split(dataset_root, prepared_root, "val", prep_workers)
    dataset_yaml = write_dataset_yaml(prepared_root)
    write_manifest(prepared_root, dataset_root, train_count, val_count)
    print(
        f"[prepare] Finished preparing dataset: train={train_count}, val={val_count}, yaml={dataset_yaml}"
    )
    return dataset_yaml


def resolve_cpu_threads(requested_threads: int, cpu_utilization: str):
    logical_cpus = max(1, os.cpu_count() or 1)
    if cpu_utilization == "aggressive":
        target_threads = logical_cpus
    else:
        target_threads = max(1, int(logical_cpus * 0.8))

    if requested_threads > 0:
        target_threads = min(requested_threads, logical_cpus)

    return max(1, target_threads), logical_cpus


def configure_runtime(args):
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        if args.device is None:
            args.device = "0"
    else:
        if args.device is None:
            args.device = "cpu"
        resolved_threads, logical_cpus = resolve_cpu_threads(args.threads, args.cpu_utilization)
        args.threads = resolved_threads
        args.logical_cpus = logical_cpus
        torch.set_num_threads(resolved_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, resolved_threads)))
        except RuntimeError:
            pass
    return has_cuda


def format_metric_value(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return str(value)


def extract_live_metrics(trainer):
    metrics = {}

    metric_source = getattr(trainer, "metrics", None)
    if isinstance(metric_source, dict):
        for key, value in metric_source.items():
            if isinstance(value, (int, float)):
                metrics[key] = value

    train_losses = getattr(trainer, "tloss", None)
    if isinstance(train_losses, dict):
        for key, value in train_losses.items():
            try:
                metrics[f"train/{key}"] = float(value)
            except (TypeError, ValueError):
                continue
    elif train_losses is not None:
        loss_names = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
        try:
            if hasattr(train_losses, "detach"):
                values = train_losses.detach().cpu().tolist()
            else:
                values = list(train_losses)
            for name, value in zip(loss_names, values):
                try:
                    metrics[name] = float(value)
                except (TypeError, ValueError):
                    continue
        except TypeError:
            pass

    return metrics


def add_live_logging(model: YOLO):
    def on_train_start(trainer):
        print(f"[train] Starting training in {trainer.save_dir}")

    def on_fit_epoch_end(trainer):
        epoch_number = trainer.epoch + 1
        total_epochs = getattr(trainer.args, "epochs", "?")
        metrics = extract_live_metrics(trainer)
        interesting_keys = [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "fitness",
        ]
        summary = " | ".join(
            f"{key}={format_metric_value(metrics.get(key))}"
            for key in interesting_keys
            if key in metrics
        )
        if not summary:
            summary = "metrics pending"
        print(f"[epoch {epoch_number}/{total_epochs}] {summary}")

    def on_train_end(trainer):
        best_path = Path(trainer.best) if getattr(trainer, "best", None) else None
        if best_path:
            print(f"[train] Finished. Best weights saved to {best_path}")
        else:
            print("[train] Finished.")

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    prepared_root = PREPARED_ROOT
    best_weights_path = Path(args.project) / args.name / "weights" / "best.pt"
    last_weights_path = Path(args.project) / args.name / "weights" / "last.pt"

    if args.resume and not last_weights_path.exists():
        print(
            f"[train] Resume requested, but no last checkpoint was found at {last_weights_path}. "
            "Starting a fresh training run instead."
        )
        args.resume = False

    if best_weights_path.exists() and not args.force_train and not args.resume:
        print(
            f"[train] Skipping training because weights already exist at {best_weights_path}. "
            "Use --force-train if you want to train again."
        )
        return

    has_cuda = configure_runtime(args)
    dataset_yaml = prepare_dataset(dataset_root, prepared_root, args.force_prepare, args.prep_workers)
    model_source = str(last_weights_path) if args.resume else args.model
    model = YOLO(model_source)
    add_live_logging(model)

    cache_mode = False if args.cache == "False" else args.cache

    train_kwargs = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "verbose": True,
        "workers": args.workers,
        "cache": cache_mode,
        "plots": False,
        "deterministic": False,
        "amp": has_cuda,
    }
    if args.device:
        train_kwargs["device"] = args.device
    if args.resume:
        train_kwargs["resume"] = True
    else:
        train_kwargs["data"] = str(dataset_yaml)

    if not has_cuda:
        print(
            "[train] GPU was not detected. Applying CPU-focused settings. "
            "For much faster training, use a CUDA-capable NVIDIA GPU if available."
        )
        print(
            f"[train] CPU resources: threads={args.threads}/{getattr(args, 'logical_cpus', args.threads)}, workers={args.workers}, "
            f"prep_workers={args.prep_workers}, cache={args.cache}, imgsz={args.imgsz}"
        )
        if (os.cpu_count() or 1) < 20:
            print(
                f"[train] This PC has {getattr(args, 'logical_cpus', os.cpu_count() or 1)} logical CPUs, "
                "so using more than that would not make training faster."
            )

    if args.resume:
        print(f"[train] Resuming training from {last_weights_path}")
    print("[train] Training will start only because you launched trainmode.py manually.")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
