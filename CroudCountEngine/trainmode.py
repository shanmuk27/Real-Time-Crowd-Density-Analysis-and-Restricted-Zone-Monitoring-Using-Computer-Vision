import argparse
import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image, ImageFile
import torch
from ultralytics import YOLO


DATASET_ROOT = Path(r"C:\CrowdHuman")
REPO_ROOT = Path(__file__).resolve().parents[1]
# Never rebuild a prepared-data folder used by a prior experiment. New runs
# receive versioned roots; `--resume` bypasses preparation and keeps its saved
# data path exactly as it was.
PREPARED_ROOT = REPO_ROOT / "generated" / "crowdhuman_yolov8n_full_v1"
RUNS_ROOT = REPO_ROOT / "runs"
# Bump only for newly prepared data. A running job holds its own dataloader and
# is never modified by this script; tiled data is always written elsewhere.
PREPARE_VERSION = 1
STARTING_WEIGHTS = RUNS_ROOT / "crowd_train" / "weights" / "best.pt"
RUN_NAME = "crowdhuman_yolov8n_full"
MAX_WEIGHT_BYTES = 10 * 1024 * 1024

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_CPU_THREADS = max(1, os.cpu_count() or 1)
DEFAULT_PREP_WORKERS = max(2, min(8, DEFAULT_CPU_THREADS))
DEFAULT_TRAIN_WORKERS = max(2, min(8, DEFAULT_CPU_THREADS // 2 or 1))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare every CrowdHuman image and fine-tune YOLOv8n on CUDA."
    )
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="CrowdHuman dataset root.")
    parser.add_argument(
        "--model", default=str(STARTING_WEIGHTS),
        help="Starting YOLOv8n weights; defaults to runs/crowd_train/weights/best.pt.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs; each uses the full training split.")
    parser.add_argument("--patience", type=int, default=40, help="Early-stop after this many non-improving validation epochs.")
    parser.add_argument(
        "--full-frame-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate every epoch on the complete CrowdHuman validation split.",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Training size that preserves small, occluded people.")
    parser.add_argument(
        "--batch", type=int, default=-1,
        help="GPU batch size; -1 selects the largest safe CUDA batch automatically.",
    )
    parser.add_argument("--device", default=None, help="Device override such as cpu, 0, or 0,1.")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU-only training. CUDA is required by default for this full dataset run.",
    )
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
        default=RUN_NAME,
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
        help="Resume an interrupted CrowdHuman run from this run's last.pt.",
    )
    parser.add_argument(
        "--cpu-utilization",
        choices=["balanced", "aggressive"],
        default="aggressive",
        help="How hard to push CPU training resources.",
    )
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
        help="Optimizer used for a fresh training run.",
    )
    parser.add_argument("--lr0", type=float, default=0.0003, help="AdamW fine-tuning learning rate for the existing YOLOv8n weights.")
    parser.add_argument("--lrf", type=float, default=0.05, help="Final cosine learning-rate fraction.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay for a fresh training run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for a fresh, reproducible run.")
    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="Maximum detections per image, above YOLO's default crowd cap of 300.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save an additional checkpoint every N epochs; -1 saves only best.pt and last.pt. last.pt is updated each epoch for resume.",
    )
    parser.add_argument("--verify-only", action="store_true", help="Prepare and audit every image and label, then exit.")
    parser.add_argument(
        "--mosaic", type=float, default=0.20,
        help="Mosaic augmentation probability. Moderate augmentation protects dense-crowd box geometry.",
    )
    parser.add_argument("--close-mosaic", type=int, default=20, help="Disable mosaic during the final epochs.")
    return parser.parse_args()


def read_split_ids(split_file: Path):
    ids = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            ids.append(stripped.split()[0])
    return ids


def evenly_spaced_ids(image_ids, fraction: float):
    """Select a deterministic spread across the whole split for a quick pilot."""
    if fraction >= 1.0:
        return image_ids
    target_count = max(1, round(len(image_ids) * fraction))
    if target_count == 1:
        return [image_ids[len(image_ids) // 2]]
    indexes = {round(index * (len(image_ids) - 1) / (target_count - 1)) for index in range(target_count)}
    return [image_id for index, image_id in enumerate(image_ids) if index in indexes]


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
    expected_count = annotation.get("human_num")
    image_width, image_height = get_image_shape(source_image_path)
    yolo_rows = convert_boxes_to_yolo(boxes, image_width, image_height)
    label_path.write_text("\n".join(yolo_rows), encoding="utf-8")
    ensure_image_link(source_image_path, prepared_image_path)
    return {
        "source_boxes": len(boxes),
        "yolo_boxes": len(yolo_rows),
        "expected_count": expected_count,
    }


def tile_positions(length: int, tile_size: int, step: int):
    """Return a deterministic grid including the far image edge exactly once."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, step))
    last_position = length - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def boxes_with_center_in_tile(boxes, x1: int, y1: int, x2: int, y2: int):
    """Keep each head in one tile by its centre, then clip its visible box."""
    tile_boxes = []
    for xmin, ymin, xmax, ymax in boxes:
        xmin, ymin, xmax, ymax = map(float, (xmin, ymin, xmax, ymax))
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        if x1 <= center_x < x2 and y1 <= center_y < y2:
            tile_boxes.append((xmin - x1, ymin - y1, xmax - x1, ymax - y1))
    return tile_boxes


def build_center_grid(boxes, cell_size: int):
    """Index head boxes by centre for near-linear tiled-label preparation."""
    grid = {}
    for box in boxes:
        xmin, ymin, xmax, ymax = map(float, box)
        cell = (int(((xmin + xmax) / 2.0) // cell_size), int(((ymin + ymax) / 2.0) // cell_size))
        grid.setdefault(cell, []).append(box)
    return grid


def grid_boxes_in_tile(grid, cell_size: int, x1: int, y1: int, x2: int, y2: int):
    """Return only boxes whose centre can be inside this tile."""
    candidates = []
    for cell_y in range(int(y1 // cell_size), int((max(y1, y2 - 1)) // cell_size) + 1):
        for cell_x in range(int(x1 // cell_size), int((max(x1, x2 - 1)) // cell_size) + 1):
            candidates.extend(grid.get((cell_x, cell_y), ()))
    return boxes_with_center_in_tile(candidates, x1, y1, x2, y2)


def process_training_tiles(
    dataset_root: Path,
    images_dir: Path,
    labels_dir: Path,
    image_id: str,
    tile_size: int,
    tile_overlap: float,
):
    """Create independent training crops without touching full-image labels.

    Validation deliberately remains full-image. During validation/inference the
    same tile geometry is used and all detections are merged in original-image
    coordinates. This avoids validating on easier pre-cropped images.
    """
    source_image_path = resolve_image_path(dataset_root, image_id)
    annotation_path = dataset_root / "jsons" / f"{image_id}.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    boxes = annotation.get("boxes", [])
    image = cv2.imread(str(source_image_path))
    if image is None:
        raise RuntimeError(f"Unable to read image {source_image_path}")

    image_height, image_width = image.shape[:2]
    step = max(1, int(tile_size * (1.0 - tile_overlap)))
    # A tile used to scan every annotation in its source image. Dense images
    # contain tens of thousands of heads, so this grid reduces preparation from
    # O(number_of_tiles * number_of_boxes) to local candidate lookups.
    box_grid = build_center_grid(boxes, step)
    tile_count = 0
    yolo_boxes = 0
    for y1 in tile_positions(image_height, tile_size, step):
        for x1 in tile_positions(image_width, tile_size, step):
            x2 = min(x1 + tile_size, image_width)
            y2 = min(y1 + tile_size, image_height)
            crop = image[y1:y2, x1:x2]
            crop_name = f"{image_id}_x{x1}_y{y1}"
            image_path = images_dir / f"{crop_name}.jpg"
            label_path = labels_dir / f"{crop_name}.txt"
            ensure_parent_dir(image_path)
            tile_boxes = grid_boxes_in_tile(box_grid, step, x1, y1, x2, y2)
            yolo_rows = convert_boxes_to_yolo(tile_boxes, x2 - x1, y2 - y1)
            label_path.write_text("\n".join(yolo_rows), encoding="utf-8")
            if not image_path.exists() and not cv2.imwrite(str(image_path), crop):
                raise RuntimeError(f"Unable to write training tile {image_path}")
            tile_count += 1
            yolo_boxes += len(yolo_rows)
    return {"images": tile_count, "source_boxes": len(boxes), "yolo_boxes": yolo_boxes}


def build_split(dataset_root: Path, prepared_root: Path, split_name: str, prep_workers: int):
    split_ids = read_split_ids(dataset_root / f"{split_name}.txt")
    labels_dir = prepared_root / "labels" / split_name
    images_dir = prepared_root / "images" / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, prep_workers)
    source_boxes = 0
    yolo_boxes = 0
    annotation_count_mismatches = 0
    sanitised_boxes = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_image_id, dataset_root, images_dir, labels_dir, image_id)
            for image_id in split_ids
        ]
        for future in futures:
            result = future.result()
            source_boxes += result["source_boxes"]
            yolo_boxes += result["yolo_boxes"]
            sanitised_boxes += result["source_boxes"] - result["yolo_boxes"]
            expected_count = result["expected_count"]
            if expected_count is not None and int(expected_count) != result["source_boxes"]:
                annotation_count_mismatches += 1

    return {
        "images": len(split_ids),
        "source_boxes": source_boxes,
        "yolo_boxes": yolo_boxes,
        "annotation_count_mismatches": annotation_count_mismatches,
        "sanitised_boxes": sanitised_boxes,
    }


def build_tiled_train_split(
    dataset_root: Path,
    prepared_root: Path,
    prep_workers: int,
    tile_size: int,
    tile_overlap: float,
    source_fraction: float,
):
    split_ids = evenly_spaced_ids(read_split_ids(dataset_root / "train.txt"), source_fraction)
    labels_dir = prepared_root / "labels" / "train"
    images_dir = prepared_root / "images" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    source_boxes = 0
    yolo_boxes = 0
    tile_images = 0
    with ThreadPoolExecutor(max_workers=max(1, prep_workers)) as executor:
        futures = [
            executor.submit(
                process_training_tiles,
                dataset_root,
                images_dir,
                labels_dir,
                image_id,
                tile_size,
                tile_overlap,
            )
            for image_id in split_ids
        ]
        for future in futures:
            result = future.result()
            source_boxes += result["source_boxes"]
            yolo_boxes += result["yolo_boxes"]
            tile_images += result["images"]

    return {
        "images": tile_images,
        "source_boxes": source_boxes,
        "yolo_boxes": yolo_boxes,
        "annotation_count_mismatches": 0,
        "sanitised_boxes": max(0, source_boxes - yolo_boxes),
    }


def write_dataset_yaml(prepared_root: Path):
    dataset_yaml = prepared_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {prepared_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: head",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_yaml


def write_manifest(prepared_root: Path, dataset_root: Path, train_summary: dict, val_summary: dict):
    manifest_path = prepared_root / "prepare_manifest.json"
    manifest = {
        "prepare_version": PREPARE_VERSION,
        "dataset_root": str(dataset_root),
        "train": train_summary,
        "val": val_summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def prepare_dataset(
    dataset_root: Path,
    prepared_root: Path,
    force_prepare: bool,
    prep_workers: int,
    train_tiles: bool = False,
    tile_size: int = 1024,
    tile_overlap: float = 0.20,
    tile_source_fraction: float = 1.0,
):
    manifest_path = prepared_root / "prepare_manifest.json"
    dataset_yaml = prepared_root / "dataset.yaml"
    manifest = load_manifest(manifest_path)
    if (
        not force_prepare
        and manifest
        and manifest.get("prepare_version") == PREPARE_VERSION
        and manifest.get("train_tiles") == train_tiles
        and manifest.get("tile_size") == (tile_size if train_tiles else None)
        and manifest.get("tile_overlap") == (tile_overlap if train_tiles else None)
        and manifest.get("tile_source_fraction") == (tile_source_fraction if train_tiles else None)
        and dataset_yaml.exists()
    ):
        print(f"[prepare] Reusing cached prepared dataset from {prepared_root}")
        return dataset_yaml

    prepared_root.mkdir(parents=True, exist_ok=True)
    if train_tiles:
        print(
            "[prepare] Building a separate original-resolution tiled training set from NWPU JSON annotations. "
            "The full-image dataset and any active run are not modified."
        )
        train_summary = build_tiled_train_split(
            dataset_root, prepared_root, prep_workers, tile_size, tile_overlap, tile_source_fraction
        )
    else:
        print("[prepare] Building YOLO labels and linked image structure from NWPU JSON annotations")
        train_summary = build_split(dataset_root, prepared_root, "train", prep_workers)
    val_summary = build_split(dataset_root, prepared_root, "val", prep_workers)
    dataset_yaml = write_dataset_yaml(prepared_root)
    write_manifest(prepared_root, dataset_root, train_summary, val_summary)
    manifest_path = prepared_root / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "train_tiles": train_tiles,
            "tile_size": tile_size if train_tiles else None,
            "tile_overlap": tile_overlap if train_tiles else None,
            "tile_source_fraction": tile_source_fraction if train_tiles else None,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "[prepare] Finished preparing dataset: "
        f"train={train_summary['images']:,} images/{train_summary['yolo_boxes']:,} head boxes, "
        f"val={val_summary['images']:,} images/{val_summary['yolo_boxes']:,} head boxes, yaml={dataset_yaml}"
    )
    for split_name, summary in (("train", train_summary), ("val", val_summary)):
        if summary["annotation_count_mismatches"] or summary["sanitised_boxes"]:
            print(
                f"[prepare] Warning: {split_name} has {summary['annotation_count_mismatches']} image(s) whose "
                f"declared crowd count differs from its source boxes and {summary['sanitised_boxes']} invalid/duplicate box(es) removed."
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
    cuda_available = torch.cuda.is_available()
    has_cuda = cuda_available and (args.device is None or str(args.device).lower() != "cpu")
    if has_cuda:
        if args.device is None:
            args.device = "0"
        device_index = int(str(args.device).replace("cuda:", "").split(",")[0])
        device_name = torch.cuda.get_device_name(device_index)
        print(f"[runtime] CUDA training on {device_name} (device={args.device})")
    else:
        if args.device is not None and str(args.device).lower() != "cpu" and not cuda_available:
            raise RuntimeError("A CUDA device was requested, but CUDA-enabled PyTorch cannot see an NVIDIA GPU.")
        if not args.allow_cpu:
            raise RuntimeError(
                "No CUDA-enabled NVIDIA GPU was found. High-accuracy YOLO11l/1024 training requires CUDA. "
                "Install a CUDA-enabled PyTorch build on the training computer, or use --allow-cpu to override."
            )
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


def validate_args(args):
    if args.epochs <= 0 or args.patience <= 0 or args.imgsz <= 0:
        raise ValueError("--epochs, --patience, and --imgsz must be greater than zero.")
    if args.batch == 0 or args.batch < -1:
        raise ValueError("--batch must be -1 (automatic) or a positive integer.")
    if args.workers < 0 or args.threads < 0 or args.prep_workers <= 0:
        raise ValueError("--workers and --threads cannot be negative; --prep-workers must be greater than zero.")
    if args.lr0 <= 0 or args.lrf <= 0 or args.weight_decay < 0:
        raise ValueError("--lr0 and --lrf must be positive; --weight-decay cannot be negative.")
    if not 0.0 <= args.mosaic <= 1.0 or args.close_mosaic < 0:
        raise ValueError("--mosaic must be between 0 and 1; --close-mosaic cannot be negative.")
    if args.max_det <= 0 or args.save_period == 0 or args.save_period < -1:
        raise ValueError("--max-det must be positive and --save-period must be -1 or a positive integer.")


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


def crowdhuman_records(annotation_path: Path):
    """Read CrowdHuman's line-delimited JSON without skipping any record."""
    with annotation_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise RuntimeError(f"No annotations found in {annotation_path}")
    return records


def crowdhuman_image_index(dataset_root: Path, split: str):
    folders = (["CrowdHuman_train01", "CrowdHuman_train02", "CrowdHuman_train03"] if split == "train"
               else ["CrowdHuman_val"])
    index = {}
    for folder_name in folders:
        folder = dataset_root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"CrowdHuman image folder is missing: {folder}")
        for image_path in folder.rglob("*.jpg"):
            previous = index.setdefault(image_path.stem, image_path)
            if previous != image_path:
                raise RuntimeError(f"Duplicate CrowdHuman image ID {image_path.stem}: {previous}, {image_path}")
    if not index:
        raise RuntimeError(f"No JPG images found for CrowdHuman {split} split.")
    return index


def crowdhuman_yolo_rows(record: dict, image_width: int, image_height: int):
    """Use non-ignored person full-body boxes; CrowdHuman masks are ignore regions."""
    rows = set()
    source_people = 0
    ignored = 0
    for gtbox in record.get("gtboxes", []):
        extra = gtbox.get("extra") or {}
        if gtbox.get("tag") != "person" or int(extra.get("ignore", 0)):
            ignored += 1
            continue
        fbox = gtbox.get("fbox")
        if not isinstance(fbox, list) or len(fbox) != 4:
            ignored += 1
            continue
        x, y, width, height = map(float, fbox)
        normalized = normalize_box(x, y, x + width, y + height, image_width, image_height)
        if normalized is None:
            ignored += 1
            continue
        source_people += 1
        xc, yc, bw, bh = normalized
        rows.add(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return sorted(rows), source_people, ignored


def prepare_crowdhuman_split(records, image_index, prepared_root: Path, split: str, workers: int):
    images_dir = prepared_root / "images" / split
    labels_dir = prepared_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    def prepare_one(record):
        image_id = record.get("ID")
        image_path = image_index.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Annotation image is missing from {split}: {image_id}")
        width, height = get_image_shape(image_path)
        rows, persons, ignored = crowdhuman_yolo_rows(record, width, height)
        (labels_dir / f"{image_id}.txt").write_text("\n".join(rows), encoding="utf-8")
        ensure_image_link(image_path, images_dir / f"{image_id}.jpg")
        return persons, len(rows), ignored

    people = labels = ignored = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for person_count, label_count, ignored_count in executor.map(prepare_one, records):
            people += person_count
            labels += label_count
            ignored += ignored_count
    return {"images": len(records), "persons": people, "labels": labels, "ignored": ignored}


def prepare_crowdhuman_dataset(dataset_root: Path, force_prepare: bool, workers: int):
    train_records = crowdhuman_records(dataset_root / "annotation_train.odgt")
    val_records = crowdhuman_records(dataset_root / "annotation_val.odgt")
    train_index = crowdhuman_image_index(dataset_root, "train")
    val_index = crowdhuman_image_index(dataset_root, "val")
    manifest_path = PREPARED_ROOT / "prepare_manifest.json"
    dataset_yaml = PREPARED_ROOT / "dataset.yaml"
    expected = {"train": len(train_records), "val": len(val_records)}
    manifest = load_manifest(manifest_path)
    prepared_label_counts = {
        split: len(list((PREPARED_ROOT / "labels" / split).glob("*.txt")))
        for split in expected
    }
    prepared_image_counts = {
        split: len(list((PREPARED_ROOT / "images" / split).glob("*.jpg")))
        for split in expected
    }
    if (not force_prepare and manifest and manifest.get("prepare_version") == PREPARE_VERSION
            and manifest.get("expected_images") == expected and dataset_yaml.is_file()
            and prepared_label_counts == expected and prepared_image_counts == expected):
        print(f"[prepare] Reusing fully audited CrowdHuman labels from {PREPARED_ROOT}")
        return dataset_yaml

    print(f"[prepare] Auditing and converting every image: {expected['train']:,} train + {expected['val']:,} val")
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    train_summary = prepare_crowdhuman_split(train_records, train_index, PREPARED_ROOT, "train", workers)
    val_summary = prepare_crowdhuman_split(val_records, val_index, PREPARED_ROOT, "val", workers)
    dataset_yaml.write_text(
        "\n".join([f"path: {PREPARED_ROOT.as_posix()}", "train: images/train", "val: images/val", "names:", "  0: person"]) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({
        "prepare_version": PREPARE_VERSION, "dataset_root": str(dataset_root), "expected_images": expected,
        "train": train_summary, "val": val_summary, "box_type": "fbox", "ignored_policy": "tag != person or extra.ignore == 1",
    }, indent=2), encoding="utf-8")
    print(f"[prepare] Complete: train={train_summary['images']:,}/{train_summary['labels']:,} labels; "
          f"val={val_summary['images']:,}/{val_summary['labels']:,} labels.")
    return dataset_yaml


def check_weight_sizes(weights_dir: Path):
    for filename in ("best.pt", "last.pt"):
        path = weights_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Training did not produce {path}")
        size = path.stat().st_size
        print(f"[weights] {filename}: {size / (1024 * 1024):.2f} MiB")
        if size >= MAX_WEIGHT_BYTES:
            raise RuntimeError(f"{path} is {size / (1024 * 1024):.2f} MiB; expected < 10 MiB for YOLOv8n deployment.")


def main():
    args = parse_args()
    validate_args(args)
    dataset_root = Path(args.dataset_root)
    best_weights_path = Path(args.project) / args.name / "weights" / "best.pt"
    last_weights_path = Path(args.project) / args.name / "weights" / "last.pt"

    if args.resume and not last_weights_path.exists():
        raise FileNotFoundError(f"--resume requires an interrupted CrowdHuman checkpoint: {last_weights_path}")

    if best_weights_path.exists() and not args.force_train and not args.resume:
        print(
            f"[train] Skipping training because weights already exist at {best_weights_path}. "
            "Use --force-train if you want to train again."
        )
        return

    dataset_yaml = prepare_crowdhuman_dataset(dataset_root, args.force_prepare, args.prep_workers)
    if args.verify_only:
        print("[verify] Every CrowdHuman train and validation annotation/image pair was prepared successfully.")
        return

    has_cuda = configure_runtime(args)
    if args.resume:
        model = YOLO(str(last_weights_path))
        add_live_logging(model)
        print(f"[train] Resuming CrowdHuman run from {last_weights_path}")
        model.train(resume=True, data=str(dataset_yaml), device=args.device)
        check_weight_sizes(last_weights_path.parent)
        return

    if not Path(args.model).is_file():
        raise FileNotFoundError(f"Starting weights are missing: {args.model}")
    model = YOLO(args.model)
    add_live_logging(model)

    cache_mode = False if args.cache == "False" else args.cache

    train_kwargs = {
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "verbose": True,
        "workers": args.workers,
        "cache": cache_mode,
        "plots": False,
        "val": args.full_frame_val,
        "deterministic": True,
        "seed": args.seed,
        "amp": has_cuda,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "cos_lr": True,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "max_det": args.max_det,
        "save_period": args.save_period,
    }
    if args.device:
        train_kwargs["device"] = args.device
    train_kwargs["data"] = str(dataset_yaml)

    print(
        f"[train] Full CrowdHuman profile: model={args.model}, imgsz={args.imgsz}, epochs={args.epochs}, "
        f"batch={args.batch}, optimizer={args.optimizer}, lr0={args.lr0}, lrf={args.lrf}, "
        f"max_det={args.max_det}, full_frame_val={args.full_frame_val}, CUDA={has_cuda}"
    )
    model.train(**train_kwargs)
    check_weight_sizes(best_weights_path.parent)


if __name__ == "__main__":
    main()
