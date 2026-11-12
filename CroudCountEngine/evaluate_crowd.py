"""Evaluate an NWPU crowd checkpoint without changing training state.

This script keeps full-frame detector validation separate from the tiled
count/localization evaluation. It deliberately uses official NWPU point labels
for tiled localization and ``human_num`` for count ground truth; display
smoothing and count corrections are never applied.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DATASET_ROOT = Path(r"C:\Crowd Count Dataset")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "crowd_yolo11l_1024" / "weights" / "best.pt"
DEFAULT_DATA = REPO_ROOT / "generated" / "nwpu_yolo" / "dataset.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NWPU head detection and crowd-count error.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="YOLO dataset YAML for detector mAP validation.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=None, help="Defaults to CUDA 0 when available, otherwise CPU.")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=15000)
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Original NWPU split for tiled count/localization evaluation.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate this many selected images; 0 evaluates the complete split.")
    parser.add_argument(
        "--per-density",
        type=int,
        default=0,
        help="Fast representative check: evaluate at most this many images in every density bin. 0 evaluates all.",
    )
    parser.add_argument(
        "--conf-sweep",
        default="",
        help="Comma-separated confidence thresholds. One model pass per image is shared by every threshold.",
    )
    parser.add_argument(
        "--point-margin",
        type=float,
        default=0.0,
        help="Pixels to expand each predicted box when matching it to an NWPU head point.",
    )
    parser.add_argument("--skip-detector-val", action="store_true", help="Skip separate full-frame Ultralytics mAP validation.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


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


def tile_positions(length: int, tile_size: int, step: int):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def nms_keep_indexes(boxes, scores, confidence: float, iou: float):
    if not boxes:
        return []
    nms_boxes = [
        [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]
        for x1, y1, x2, y2 in boxes
    ]
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores, confidence, iou)
    return [] if len(keep) == 0 else np.asarray(keep).reshape(-1).tolist()


def merge_boxes(boxes, scores, confidence: float, iou: float):
    if not boxes:
        return [], []
    indexes = nms_keep_indexes(boxes, scores, confidence, iou)
    return [boxes[index] for index in indexes], [scores[index] for index in indexes]


def model_boxes(results, x_offset: int, y_offset: int):
    boxes, scores = [], []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        coordinates = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(coordinates, confidences):
            boxes.append((float(x1 + x_offset), float(y1 + y_offset), float(x2 + x_offset), float(y2 + y_offset)))
            scores.append(float(score))
    return boxes, scores


def predict_boxes(model, image, args, device, class_ids, minimum_confidence: float):
    height, width = image.shape[:2]
    predict_args = {
        "conf": minimum_confidence,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "device": device,
        "classes": class_ids,
        "verbose": False,
    }
    if device != "cpu":
        predict_args["half"] = True
    if not args.tiled:
        results = model.predict(image, **predict_args)
        return model_boxes(results, 0, 0)

    step = max(1, int(args.tile_size * (1.0 - args.tile_overlap)))
    all_boxes, all_scores = [], []
    tiles = []
    for y1 in tile_positions(height, args.tile_size, step):
        for x1 in tile_positions(width, args.tile_size, step):
            tiles.append((image[y1 : min(y1 + args.tile_size, height), x1 : min(x1 + args.tile_size, width)], x1, y1))
    for start in range(0, len(tiles), args.tile_batch_size):
        batch = tiles[start : start + args.tile_batch_size]
        results = model.predict([tile for tile, _, _ in batch], **predict_args)
        for (_, x_offset, y_offset), result in zip(batch, results):
            boxes, scores = model_boxes([result], x_offset, y_offset)
            all_boxes.extend(boxes)
            all_scores.extend(scores)
    return all_boxes, all_scores


def density_bin(count: int):
    if count == 0:
        return "0"
    if count <= 50:
        return "1-50"
    if count <= 100:
        return "51-100"
    if count <= 500:
        return "101-500"
    if count <= 1000:
        return "501-1000"
    return ">1000"


def metric_summary(rows):
    if not rows:
        return {"images": 0}
    errors = [prediction - truth for prediction, truth in rows]
    nonzero_truth = [(prediction, truth) for prediction, truth in rows if truth > 0]
    return {
        "images": len(rows),
        "mae": round(sum(abs(error) for error in errors) / len(errors), 4),
        "rmse": round((sum(error * error for error in errors) / len(errors)) ** 0.5, 4),
        "bias": round(sum(errors) / len(errors), 4),
        "mape_percent": round(
            100.0 * sum(abs(prediction - truth) / truth for prediction, truth in nonzero_truth) / len(nonzero_truth), 4
        ) if nonzero_truth else None,
    }


def localization_summary(rows):
    """Micro-averaged point localization metrics over a collection of images."""
    predicted = sum(row[0] for row in rows)
    truth = sum(row[1] for row in rows)
    matched = sum(row[2] for row in rows)
    precision = matched / predicted if predicted else 0.0
    recall = matched / truth if truth else 0.0
    return {
        "images": len(rows),
        "predicted_heads": predicted,
        "ground_truth_heads": truth,
        "matched_heads": matched,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round((2 * precision * recall / (precision + recall)) if precision + recall else 0.0, 6),
    }


def match_boxes_to_points(boxes, scores, points, margin: float = 0.0, cell_size: int = 128):
    """Greedily match predicted head boxes to distinct NWPU point labels.

    A spatial hash keeps matching local even in images with thousands of heads.
    A predicted box can match at most one point and each point at most one box.
    """
    point_grid = defaultdict(list)
    for point_index, (x, y) in enumerate(points):
        point_grid[(int(float(x) // cell_size), int(float(y) // cell_size))].append(point_index)

    matched_points = set()
    matched = 0
    for box_index in sorted(range(len(boxes)), key=lambda index: scores[index], reverse=True):
        x1, y1, x2, y2 = boxes[box_index]
        x1, y1, x2, y2 = x1 - margin, y1 - margin, x2 + margin, y2 + margin
        best_point = None
        best_distance = None
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for grid_y in range(int(y1 // cell_size), int(y2 // cell_size) + 1):
            for grid_x in range(int(x1 // cell_size), int(x2 // cell_size) + 1):
                for point_index in point_grid.get((grid_x, grid_y), ()):
                    if point_index in matched_points:
                        continue
                    point_x, point_y = map(float, points[point_index])
                    if x1 <= point_x <= x2 and y1 <= point_y <= y2:
                        distance = (point_x - center_x) ** 2 + (point_y - center_y) ** 2
                        if best_distance is None or distance < best_distance:
                            best_point, best_distance = point_index, distance
        if best_point is not None:
            matched_points.add(best_point)
            matched += 1
    return matched


def parse_confidences(text: str, default_confidence: float):
    values = {default_confidence}
    for value in text.split(","):
        value = value.strip()
        if value:
            values.add(float(value))
    thresholds = sorted(values)
    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Every confidence threshold must be between 0 and 1.")
    return thresholds


def select_image_ids(dataset_root: Path, split: str, per_density: int, limit: int):
    image_ids = [line.strip().split()[0] for line in (dataset_root / f"{split}.txt").read_text().splitlines() if line.strip()]
    if per_density:
        grouped_ids = defaultdict(list)
        for image_id in image_ids:
            annotation = json.loads((dataset_root / "jsons" / f"{image_id}.json").read_text(encoding="utf-8"))
            grouped_ids[density_bin(int(annotation["human_num"]))].append(image_id)
        image_ids = [image_id for group in grouped_ids.values() for image_id in group[:per_density]]
    return image_ids[:limit] if limit else image_ids


def main():
    args = parse_args()
    if args.max_det <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.tile_batch_size <= 0 or args.limit < 0:
        raise ValueError("Image sizes, batch sizes, and max-det must be positive.")
    if args.per_density < 0 or args.point_margin < 0:
        raise ValueError("--per-density and --point-margin cannot be negative.")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or not 0 <= args.tile_overlap < 1:
        raise ValueError("Invalid confidence, IoU, or tile overlap.")
    confidence_thresholds = parse_confidences(args.conf_sweep, args.conf)

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA-enabled PyTorch cannot see a GPU.")
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)
    class_ids = [int(index) for index, name in model.names.items() if str(name).strip().lower() in {"person", "head"}]
    if not class_ids:
        raise RuntimeError(f"Expected a person/head detector; checkpoint classes are {model.names}")
    print(
        f"[runtime] device={device}, weights={args.weights}, split={args.split}, max_det={args.max_det}, "
        f"tiled={args.tiled}, confidences={confidence_thresholds}"
    )

    detector_metrics = None
    if not args.skip_detector_val and args.split == "val":
        data_path = Path(args.data)
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset YAML not found for detector validation: {data_path}")
        detector_metrics = model.val(
            data=str(data_path),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            max_det=args.max_det,
            classes=class_ids,
            verbose=True,
            plots=False,
        )
    elif not args.skip_detector_val:
        print("[detector] Skipping full-frame Ultralytics validation for --split train; tiled point metrics are reported instead.")

    dataset_root = Path(args.dataset_root)
    image_ids = select_image_ids(dataset_root, args.split, args.per_density, args.limit)
    if not image_ids:
        raise RuntimeError("No images were selected for evaluation.")
    per_confidence = {
        threshold: {"counts": [], "by_density": defaultdict(list), "localization": [], "localization_by_density": defaultdict(list)}
        for threshold in confidence_thresholds
    }
    lowest_confidence = confidence_thresholds[0]
    for index, image_id in enumerate(image_ids, start=1):
        annotation = json.loads((dataset_root / "jsons" / f"{image_id}.json").read_text(encoding="utf-8"))
        truth = int(annotation["human_num"])
        points = annotation.get("points", [])
        if len(points) != truth:
            raise RuntimeError(f"Image {image_id}: human_num={truth} but points={len(points)}; refusing inconsistent count evaluation.")
        image = cv2.imread(str(resolve_image_path(dataset_root, image_id)))
        if image is None:
            raise RuntimeError(f"Unable to read validation image {image_id}")
        raw_boxes, raw_scores = predict_boxes(model, image, args, device, class_ids, lowest_confidence)
        group = density_bin(truth)
        displayed_prediction = None
        for threshold, records in per_confidence.items():
            selected = [score >= threshold for score in raw_scores]
            boxes = [box for box, keep in zip(raw_boxes, selected) if keep]
            scores = [score for score, keep in zip(raw_scores, selected) if keep]
            boxes, scores = merge_boxes(boxes, scores, threshold, args.iou)
            prediction = len(boxes)
            matched = match_boxes_to_points(boxes, scores, points, args.point_margin)
            count_row = (prediction, truth)
            localization_row = (prediction, truth, matched)
            records["counts"].append(count_row)
            records["by_density"][group].append(count_row)
            records["localization"].append(localization_row)
            records["localization_by_density"][group].append(localization_row)
            if threshold == args.conf:
                displayed_prediction = prediction
        if index % 10 == 0 or index == len(image_ids):
            print(f"[count] {index}/{len(image_ids)} image={image_id} predicted={displayed_prediction} truth={truth}")

    operating_records = per_confidence[args.conf]
    report = {
        "settings": {
            "weights": str(args.weights), "device": device, "imgsz": args.imgsz, "conf": args.conf,
            "iou": args.iou, "max_det": args.max_det, "tiled": args.tiled,
            "tile_size": args.tile_size, "tile_overlap": args.tile_overlap, "split": args.split,
            "point_margin": args.point_margin, "evaluated_images": len(image_ids),
        },
        "count_overall": metric_summary(operating_records["counts"]),
        "count_by_density": {name: metric_summary(rows) for name, rows in operating_records["by_density"].items()},
        "tiled_point_localization": localization_summary(operating_records["localization"]),
        "tiled_point_localization_by_density": {
            name: localization_summary(rows) for name, rows in operating_records["localization_by_density"].items()
        },
        "confidence_sweep": {
            f"{threshold:.4f}": {
                "count_overall": metric_summary(records["counts"]),
                "tiled_point_localization": localization_summary(records["localization"]),
            }
            for threshold, records in per_confidence.items()
        },
    }
    if detector_metrics is not None:
        report["full_frame_detector_validation"] = {
            "precision": float(detector_metrics.box.mp), "recall": float(detector_metrics.box.mr),
            "map50": float(detector_metrics.box.map50), "map50_95": float(detector_metrics.box.map),
        }
    print(json.dumps(report, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[report] Saved {output_path}")


if __name__ == "__main__":
    main()
