"""Convert WIDER FACE and UFDD annotations to the YOLO detection format.

WIDER FACE is kept for training and validation. UFDD is kept as an external
test set, so its metrics remain a meaningful measure of robustness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
import sys
from typing import Iterable

import cv2


DEFAULT_DATASET_ROOT = Path(r"C:\Face Validation and detection Dataset")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "generated" / "face_detection_yolo"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ConversionSummary:
    split: str
    annotated_images: int
    labels_written: int
    boxes_written: int
    invalid_boxes_skipped: int
    missing_images: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WIDER FACE + UFDD for Ultralytics YOLO training.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--wider-annotations-dir",
        type=Path,
        default=None,
        help="Directory that contains wider_face_train_bbx_gt.txt and wider_face_val_bbx_gt.txt.",
    )
    return parser.parse_args()


def find_existing_path(root: Path, candidates: Iterable[Path], description: str) -> Path:
    for relative_path in candidates:
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(root / path) for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Looked in:\n  {searched}")


def find_annotation(root: Path, filename: str, description: str, annotations_dir: Path | None) -> Path:
    candidates = []
    if annotations_dir is not None:
        candidates.append(annotations_dir / filename)
    candidates.extend(
        [
            root / "WIDER" / "wider_face_split" / "wider_face_split" / filename,
            root / "WIDER" / "wider_face_split" / filename,
            root / "wider_face_split" / filename,
            root / "WIDER" / filename,
            root / filename,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {description} ({filename}).\n"
        "Download the WIDER FACE 'Face annotations' archive, extract its wider_face_split folder "
        "to the dataset root or its WIDER folder, then run this script again."
    )


def label_directory_for(images_directory: Path) -> Path:
    if images_directory.name != "images":
        raise ValueError(f"Expected an images directory, received: {images_directory}")
    return images_directory.parent / "labels"


def iter_annotation_records(annotation_file: Path):
    """Yield (relative-image-path, bounding-box rows) from WIDER/UFDD text files."""
    with annotation_file.open("r", encoding="utf-8-sig") as annotation_stream:
        lines = iter(annotation_stream)
        while True:
            try:
                image_name = next(lines).strip()
            except StopIteration:
                break
            if not image_name:
                continue
            try:
                box_count_line = next(lines).strip()
            except StopIteration as error:
                raise ValueError(f"Unexpected end of annotation file after {image_name!r}") from error
            try:
                box_count = int(box_count_line)
            except ValueError as error:
                raise ValueError(
                    f"Expected number of boxes after {image_name!r}, found {box_count_line!r}"
                ) from error
            box_rows = []
            for _ in range(box_count):
                try:
                    row = next(lines).strip()
                except StopIteration as error:
                    raise ValueError(f"Unexpected end of annotation file while reading {image_name!r}") from error
                if row:
                    box_rows.append([float(value) for value in row.split()])

            # Some official WIDER annotations represent a zero-face image as
            # ``0`` followed by a sentinel row of ten zeroes. It is not a
            # bounding box and must be consumed, otherwise it is mistaken for
            # the next image name. UFDD does not normally use this convention.
            if box_count == 0:
                try:
                    possible_sentinel = next(lines).strip()
                except StopIteration:
                    possible_sentinel = None
                if possible_sentinel is not None:
                    fields = possible_sentinel.split()
                    try:
                        is_numeric_row = len(fields) >= 4 and all(float(value) == 0.0 for value in fields)
                    except ValueError:
                        is_numeric_row = False
                    if not is_numeric_row:
                        # Put back a non-sentinel line without materialising
                        # the whole annotation file in memory.
                        lines = chain([possible_sentinel], lines)
            yield Path(image_name.replace("\\", "/")), box_rows


def convert_split(*, split_name: str, images_directory: Path, annotation_file: Path) -> ConversionSummary:
    """Write normalized YOLO labels while preserving all valid face boxes."""
    labels_directory = label_directory_for(images_directory)
    labels_directory.mkdir(parents=True, exist_ok=True)
    annotated_images = labels_written = boxes_written = invalid_boxes_skipped = missing_images = 0

    for relative_image_path, box_rows in iter_annotation_records(annotation_file):
        annotated_images += 1
        image_path = images_directory / relative_image_path
        if not image_path.exists():
            missing_images += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images += 1
            continue
        image_height, image_width = image.shape[:2]
        yolo_rows: list[str] = []

        for row in box_rows:
            if len(row) < 4:
                invalid_boxes_skipped += 1
                continue
            x, y, width, height = row[:4]
            # The eighth WIDER/UFDD field is the official invalid marker.
            is_invalid = len(row) >= 8 and int(row[7]) == 1
            if is_invalid or width <= 1 or height <= 1:
                invalid_boxes_skipped += 1
                continue
            # Clip edge-touching annotations before normalising.
            left = max(0.0, min(x, image_width - 1.0))
            top = max(0.0, min(y, image_height - 1.0))
            right = max(0.0, min(x + width, float(image_width)))
            bottom = max(0.0, min(y + height, float(image_height)))
            clipped_width, clipped_height = right - left, bottom - top
            if clipped_width <= 1 or clipped_height <= 1:
                invalid_boxes_skipped += 1
                continue
            center_x = (left + right) / 2.0 / image_width
            center_y = (top + bottom) / 2.0 / image_height
            yolo_rows.append(
                f"0 {center_x:.8f} {center_y:.8f} {clipped_width / image_width:.8f} "
                f"{clipped_height / image_height:.8f}"
            )
            boxes_written += 1

        label_path = (labels_directory / relative_image_path).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(yolo_rows) + ("\n" if yolo_rows else ""), encoding="utf-8")
        labels_written += 1

    return ConversionSummary(
        split_name, annotated_images, labels_written, boxes_written, invalid_boxes_skipped, missing_images
    )


def count_images(directory: Path) -> int:
    return sum(path.suffix.lower() in IMAGE_SUFFIXES for path in directory.rglob("*"))


def yaml_value(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def write_dataset_yaml(output_root: Path, train_images: Path, val_images: Path, test_images: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    yaml_path = output_root / "face_dataset.yaml"
    yaml_path.write_text(
        "# Generated by prepare_face_dataset.py\n"
        f"train: {yaml_value(train_images)}\n"
        f"val: {yaml_value(val_images)}\n"
        f"test: {yaml_value(test_images)}\n"
        "names:\n  0: face\n",
        encoding="utf-8",
    )
    return yaml_path


def print_summary(summary: ConversionSummary) -> None:
    print(
        f"[{summary.split}] annotations={summary.annotated_images:,}, labels={summary.labels_written:,}, "
        f"faces={summary.boxes_written:,}, skipped={summary.invalid_boxes_skipped:,}, "
        f"missing-images={summary.missing_images:,}"
    )


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    wider_train_images = find_existing_path(
        dataset_root, [Path("WIDER/WIDER_train/WIDER_train/images")], "WIDER FACE training images"
    )
    wider_val_images = find_existing_path(
        dataset_root, [Path("WIDER/WIDER_val/WIDER_val/images")], "WIDER FACE validation images"
    )
    ufdd_images = find_existing_path(
        dataset_root, [Path("UFDD/UFDD_val/UFDD_val/images")], "UFDD validation images"
    )
    wider_train_annotations = find_annotation(
        dataset_root, "wider_face_train_bbx_gt.txt", "WIDER FACE training annotations", args.wider_annotations_dir
    )
    wider_val_annotations = find_annotation(
        dataset_root, "wider_face_val_bbx_gt.txt", "WIDER FACE validation annotations", args.wider_annotations_dir
    )
    ufdd_annotations = find_existing_path(
        dataset_root,
        [Path("UFDD/UFDD-annotationfile/UFDD-annotationfile/UFDD_split/UFDD_val_bbx_gt.txt")],
        "UFDD validation annotations",
    )

    print("Preparing labels; source images are left in place.")
    summaries = [
        convert_split(split_name="WIDER train", images_directory=wider_train_images, annotation_file=wider_train_annotations),
        convert_split(split_name="WIDER validation", images_directory=wider_val_images, annotation_file=wider_val_annotations),
        convert_split(split_name="UFDD external test", images_directory=ufdd_images, annotation_file=ufdd_annotations),
    ]
    for summary in summaries:
        print_summary(summary)
    if any(summary.missing_images for summary in summaries):
        raise RuntimeError("Some annotations did not match images. Training was not started; fix the dataset layout first.")

    dataset_yaml = write_dataset_yaml(args.output_root, wider_train_images, wider_val_images, ufdd_images)
    print(
        f"Image counts: WIDER train={count_images(wider_train_images):,}, "
        f"WIDER validation={count_images(wider_val_images):,}, UFDD test={count_images(ufdd_images):,}"
    )
    print(f"Dataset configuration written to: {dataset_yaml}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
