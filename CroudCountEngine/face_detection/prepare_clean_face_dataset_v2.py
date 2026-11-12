"""Build a separate, conservative v2 YOLO dataset from WIDER FACE and UFDD.

Original images and v1 labels are never changed.  The v2 dataset contains
hard-linked images when possible (copied only across volumes) plus clean labels
in a standard YOLO folder layout, so its filtering cannot affect v1 training.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

import cv2

try:  # Supports both `python file.py` and `python -m ...` execution.
    from .prepare_face_dataset import (
        DEFAULT_DATASET_ROOT,
        find_annotation,
        find_existing_path,
        iter_annotation_records,
        yaml_value,
    )
except ImportError:
    from prepare_face_dataset import (  # type: ignore
        DEFAULT_DATASET_ROOT,
        find_annotation,
        find_existing_path,
        iter_annotation_records,
        yaml_value,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "generated" / "face_detection_yolo_v2"
REFERENCE_IMAGE_SIZE = 1024


@dataclass
class BuildSummary:
    split: str
    annotations: int = 0
    images_included: int = 0
    images_excluded: int = 0
    image_links: int = 0
    image_copies: int = 0
    faces_written: int = 0
    invalid_boxes_removed: int = 0
    missing_or_unreadable_images: int = 0
    images_with_only_tiny_faces: int = 0
    images_without_usable_boxes: int = 0
    exact_duplicate_training_images_removed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a separate cleaned v2 face-detection dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wider-annotations-dir", type=Path, default=None)
    parser.add_argument(
        "--duplicate-manifest",
        type=Path,
        default=PROJECT_ROOT / "generated" / "face_dataset_audit" / "exact_duplicate_images.csv",
        help=(
            "CSV produced by audit_face_dataset.py. Training-side exact duplicates are excluded while "
            "validation and UFDD test are preserved. Missing manifests are allowed with a warning."
        ),
    )
    parser.add_argument(
        "--min-effective-face-pixels",
        type=float,
        default=8.0,
        help=(
            "Exclude an image only when every valid face is smaller than this side length after scaling "
            f"its longest side to {REFERENCE_IMAGE_SIZE}px. Default: 8."
        ),
    )
    parser.add_argument(
        "--exclude-train-empty-images",
        action="store_true",
        help="Exclude declared zero-face images from training. By default they are retained as negative examples.",
    )
    parser.add_argument(
        "--also-filter-validation",
        action="store_true",
        help=(
            "Also apply tiny-only/no-usable-box exclusions to validation. Disabled by default so v1/v2 "
            "validation metrics remain directly comparable. UFDD test is never filtered."
        ),
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="auto uses hard links on the same drive and copies only when necessary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild an existing v2 output directory. Use only for an intentional rebuild.",
    )
    return parser.parse_args()


def normalise_box(row: list[float], image_width: int, image_height: int) -> tuple[str | None, bool, float]:
    """Return YOLO label, invalid flag, and effective minimum face side at 1024px."""
    if len(row) < 4:
        return None, True, 0.0
    x, y, width, height = row[:4]
    invalid_marker = len(row) >= 8 and int(row[7]) == 1
    if invalid_marker or width <= 1 or height <= 1:
        return None, True, 0.0

    left = max(0.0, min(x, image_width - 1.0))
    top = max(0.0, min(y, image_height - 1.0))
    right = max(0.0, min(x + width, float(image_width)))
    bottom = max(0.0, min(y + height, float(image_height)))
    clipped_width, clipped_height = right - left, bottom - top
    if clipped_width <= 1 or clipped_height <= 1:
        return None, True, 0.0

    centre_x = (left + right) / 2.0 / image_width
    centre_y = (top + bottom) / 2.0 / image_height
    effective_side = min(clipped_width, clipped_height) * REFERENCE_IMAGE_SIZE / max(image_width, image_height)
    return (
        f"0 {centre_x:.8f} {centre_y:.8f} {clipped_width / image_width:.8f} {clipped_height / image_height:.8f}",
        False,
        effective_side,
    )


def link_or_copy(source: Path, destination: Path, link_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_mode != "copy":
        try:
            destination.hardlink_to(source)
            return "linked"
        except OSError as error:
            if link_mode == "hardlink":
                raise RuntimeError(f"Could not hard-link {source} to {destination}: {error}") from error
    shutil.copy2(source, destination)
    return "copied"


def read_training_duplicate_exclusions(manifest: Path) -> set[Path]:
    """Choose the train-side file from every exact-duplicate pair for exclusion.

    Validation and external-test examples are never excluded, because they are
    the fixed evidence used to compare v1 and v2.  For a train/validation pair
    we therefore remove the training copy, even if it appeared first in the
    duplicate manifest.
    """
    manifest = manifest.resolve()
    if not manifest.is_file():
        print(f"Warning: exact-duplicate manifest was not found; skipping duplicate filtering: {manifest}")
        return set()

    exclusions: set[Path] = set()
    with manifest.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            first_split = row.get("first_split", "")
            duplicate_split = row.get("duplicate_split", "")
            first_image = row.get("first_image", "")
            duplicate_image = row.get("duplicate_image", "")
            if first_split == "WIDER train" and duplicate_split != "WIDER train" and first_image:
                exclusions.add(Path(first_image).resolve())
            if duplicate_split == "WIDER train" and duplicate_image:
                exclusions.add(Path(duplicate_image).resolve())
    return exclusions


def build_split(
    *,
    split_name: str,
    output_split: str,
    images_directory: Path,
    annotation_file: Path,
    output_root: Path,
    min_effective_face_pixels: float,
    exclude_tiny_only_images: bool,
    exclude_no_usable_box_images: bool,
    exclude_declared_empty_images: bool,
    exact_duplicate_training_images: set[Path],
    link_mode: str,
    exclusions: list[dict[str, str]],
) -> BuildSummary:
    summary = BuildSummary(split=split_name)
    images_output = output_root / "images" / output_split
    labels_output = output_root / "labels" / output_split

    for relative_path, rows in iter_annotation_records(annotation_file):
        summary.annotations += 1
        source_image = images_directory / relative_path
        if source_image.resolve() in exact_duplicate_training_images:
            summary.images_excluded += 1
            summary.exact_duplicate_training_images_removed += 1
            exclusions.append(
                {
                    "split": split_name,
                    "image": relative_path.as_posix(),
                    "reason": "exact_duplicate_or_cross_split_leakage",
                }
            )
            continue
        if not source_image.is_file():
            summary.images_excluded += 1
            summary.missing_or_unreadable_images += 1
            exclusions.append({"split": split_name, "image": relative_path.as_posix(), "reason": "missing_image"})
            continue
        image = cv2.imread(str(source_image))
        if image is None:
            summary.images_excluded += 1
            summary.missing_or_unreadable_images += 1
            exclusions.append({"split": split_name, "image": relative_path.as_posix(), "reason": "unreadable_image"})
            continue

        image_height, image_width = image.shape[:2]
        labels: list[str] = []
        learnable_faces = 0
        for row in rows:
            label, invalid, effective_side = normalise_box(row, image_width, image_height)
            if invalid or label is None:
                summary.invalid_boxes_removed += 1
                continue
            labels.append(label)
            if effective_side >= min_effective_face_pixels:
                learnable_faces += 1

        if rows and not labels and exclude_no_usable_box_images:
            summary.images_excluded += 1
            summary.images_without_usable_boxes += 1
            exclusions.append({"split": split_name, "image": relative_path.as_posix(), "reason": "no_usable_face_boxes"})
            continue
        if labels and learnable_faces == 0 and exclude_tiny_only_images:
            # Excluding the entire image avoids treating tiny faces as background.
            summary.images_excluded += 1
            summary.images_with_only_tiny_faces += 1
            exclusions.append(
                {
                    "split": split_name,
                    "image": relative_path.as_posix(),
                    "reason": f"all_faces_below_{min_effective_face_pixels:g}px_at_{REFERENCE_IMAGE_SIZE}",
                }
            )
            continue
        if not rows and exclude_declared_empty_images:
            summary.images_excluded += 1
            exclusions.append({"split": split_name, "image": relative_path.as_posix(), "reason": "declared_empty_image"})
            continue

        destination_image = images_output / relative_path
        result = link_or_copy(source_image, destination_image, link_mode)
        if result == "linked":
            summary.image_links += 1
        else:
            summary.image_copies += 1
        destination_label = (labels_output / relative_path).with_suffix(".txt")
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        destination_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        summary.images_included += 1
        summary.faces_written += len(labels)
    return summary


def write_dataset_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "face_dataset_v2.yaml"
    yaml_path.write_text(
        "# Generated by prepare_clean_face_dataset_v2.py\n"
        f"train: {yaml_value(output_root / 'images' / 'train')}\n"
        f"val: {yaml_value(output_root / 'images' / 'val')}\n"
        f"test: {yaml_value(output_root / 'images' / 'test')}\n"
        "names:\n  0: face\n",
        encoding="utf-8",
    )
    return yaml_path


def write_exclusions(path: Path, exclusions: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "image", "reason"])
        writer.writeheader()
        writer.writerows(exclusions)


def main() -> int:
    args = parse_args()
    if args.min_effective_face_pixels < 0:
        raise ValueError("--min-effective-face-pixels cannot be negative.")
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise RuntimeError(
                f"The v2 output directory already contains files: {output_root}\n"
                "Use another --output-root, or pass --overwrite only when you intentionally want to rebuild v2."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

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

    print("Building v2 in a new directory; source images and v1 labels are not modified.")
    training_duplicate_exclusions = read_training_duplicate_exclusions(args.duplicate_manifest)
    if training_duplicate_exclusions:
        print(f"Excluding {len(training_duplicate_exclusions):,} training-side exact duplicate(s) from the audit manifest.")
    exclusions: list[dict[str, str]] = []
    summaries = [
        build_split(
            split_name="WIDER train",
            output_split="train",
            images_directory=wider_train_images,
            annotation_file=wider_train_annotations,
            output_root=output_root,
            min_effective_face_pixels=args.min_effective_face_pixels,
            exclude_tiny_only_images=True,
            exclude_no_usable_box_images=True,
            exclude_declared_empty_images=args.exclude_train_empty_images,
            exact_duplicate_training_images=training_duplicate_exclusions,
            link_mode=args.link_mode,
            exclusions=exclusions,
        ),
        build_split(
            split_name="WIDER validation",
            output_split="val",
            images_directory=wider_val_images,
            annotation_file=wider_val_annotations,
            output_root=output_root,
            min_effective_face_pixels=args.min_effective_face_pixels,
            exclude_tiny_only_images=args.also_filter_validation,
            exclude_no_usable_box_images=args.also_filter_validation,
            exclude_declared_empty_images=False,
            exact_duplicate_training_images=set(),
            link_mode=args.link_mode,
            exclusions=exclusions,
        ),
        build_split(
            split_name="UFDD external test",
            output_split="test",
            images_directory=ufdd_images,
            annotation_file=ufdd_annotations,
            output_root=output_root,
            min_effective_face_pixels=args.min_effective_face_pixels,
            # Keep UFDD exactly intact for a fair external v1/v2 comparison.
            exclude_tiny_only_images=False,
            exclude_no_usable_box_images=False,
            exclude_declared_empty_images=False,
            exact_duplicate_training_images=set(),
            link_mode=args.link_mode,
            exclusions=exclusions,
        ),
    ]
    write_exclusions(output_root / "v2_excluded_images.csv", exclusions)
    dataset_yaml = write_dataset_yaml(output_root)
    for summary in summaries:
        print(
            f"[{summary.split}] included={summary.images_included:,}, faces={summary.faces_written:,}, "
            f"excluded={summary.images_excluded:,}, tiny-only={summary.images_with_only_tiny_faces:,}, "
            f"duplicates={summary.exact_duplicate_training_images_removed:,}, hard-links={summary.image_links:,}, "
            f"copies={summary.image_copies:,}"
        )
    print(f"v2 dataset YAML written to: {dataset_yaml}")
    print(f"Exclusion manifest written to: {output_root / 'v2_excluded_images.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"v2 dataset preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
