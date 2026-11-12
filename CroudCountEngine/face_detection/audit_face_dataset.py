"""Read-only quality audit for the WIDER FACE and UFDD training sources.

The audit never changes source images or labels.  It produces compact reports
that identify missing/corrupt images, unusable annotations, images containing
only faces below the configured training resolution, and exact byte duplicates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import cv2

try:  # Supports both `python file.py` and `python -m ...` execution.
    from .prepare_face_dataset import (
        DEFAULT_DATASET_ROOT,
        find_annotation,
        find_existing_path,
        iter_annotation_records,
    )
except ImportError:
    from prepare_face_dataset import (  # type: ignore
        DEFAULT_DATASET_ROOT,
        find_annotation,
        find_existing_path,
        iter_annotation_records,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "generated" / "face_dataset_audit"
REFERENCE_IMAGE_SIZE = 1024


@dataclass
class SplitAudit:
    split: str
    annotation_images: int = 0
    images_read: int = 0
    missing_images: int = 0
    unreadable_images: int = 0
    declared_empty_images: int = 0
    raw_boxes: int = 0
    valid_boxes: int = 0
    invalid_marked_boxes: int = 0
    malformed_or_degenerate_boxes: int = 0
    clipped_boxes: int = 0
    images_with_only_tiny_faces: int = 0
    faces_below_min_effective_size: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit WIDER FACE + UFDD without changing the source data.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wider-annotations-dir", type=Path, default=None)
    parser.add_argument(
        "--min-effective-face-pixels",
        type=float,
        default=8.0,
        help=(
            "Face side length at a 1024px training image below which an image is flagged as tiny-only. "
            "This is a report threshold; it does not delete anything."
        ),
    )
    parser.add_argument(
        "--skip-exact-duplicate-check",
        action="store_true",
        help="Skip the slower SHA-256 scan for byte-identical images.",
    )
    return parser.parse_args()


def is_valid_box(row: list[float], image_width: int, image_height: int) -> tuple[bool, bool, bool, float]:
    """Return valid, clipped, invalid-marker, effective-min-side-at-1024."""
    if len(row) < 4:
        return False, False, False, 0.0
    x, y, width, height = row[:4]
    invalid_marker = len(row) >= 8 and int(row[7]) == 1
    if invalid_marker or width <= 1 or height <= 1:
        return False, False, invalid_marker, 0.0

    left = max(0.0, min(x, image_width - 1.0))
    top = max(0.0, min(y, image_height - 1.0))
    right = max(0.0, min(x + width, float(image_width)))
    bottom = max(0.0, min(y + height, float(image_height)))
    clipped_width, clipped_height = right - left, bottom - top
    if clipped_width <= 1 or clipped_height <= 1:
        return False, False, False, 0.0

    was_clipped = left != x or top != y or right != x + width or bottom != y + height
    scale_to_reference = REFERENCE_IMAGE_SIZE / max(image_width, image_height)
    return True, was_clipped, False, min(clipped_width, clipped_height) * scale_to_reference


def audit_split(
    *,
    split_name: str,
    images_directory: Path,
    annotation_file: Path,
    min_effective_face_pixels: float,
    candidates: list[dict[str, str]],
) -> SplitAudit:
    summary = SplitAudit(split=split_name)
    for relative_path, rows in iter_annotation_records(annotation_file):
        summary.annotation_images += 1
        image_path = images_directory / relative_path
        if not image_path.is_file():
            summary.missing_images += 1
            candidates.append({"split": split_name, "image": relative_path.as_posix(), "reason": "missing_image"})
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            summary.unreadable_images += 1
            candidates.append({"split": split_name, "image": relative_path.as_posix(), "reason": "unreadable_image"})
            continue
        summary.images_read += 1
        image_height, image_width = image.shape[:2]
        if not rows:
            summary.declared_empty_images += 1
            continue

        valid_in_image = 0
        learnable_in_image = 0
        for row in rows:
            summary.raw_boxes += 1
            valid, was_clipped, invalid_marker, effective_side = is_valid_box(row, image_width, image_height)
            if not valid:
                if invalid_marker:
                    summary.invalid_marked_boxes += 1
                else:
                    summary.malformed_or_degenerate_boxes += 1
                continue
            summary.valid_boxes += 1
            valid_in_image += 1
            if was_clipped:
                summary.clipped_boxes += 1
            if effective_side < min_effective_face_pixels:
                summary.faces_below_min_effective_size += 1
            else:
                learnable_in_image += 1

        if valid_in_image == 0:
            candidates.append({"split": split_name, "image": relative_path.as_posix(), "reason": "no_usable_face_boxes"})
        elif learnable_in_image == 0:
            summary.images_with_only_tiny_faces += 1
            candidates.append(
                {
                    "split": split_name,
                    "image": relative_path.as_posix(),
                    "reason": f"all_faces_below_{min_effective_face_pixels:g}px_at_{REFERENCE_IMAGE_SIZE}",
                }
            )
    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_stream:
        for chunk in iter(lambda: image_stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_exact_duplicates(sources: Iterable[tuple[str, Path]]) -> list[dict[str, str]]:
    """Find byte-identical files only; near-duplicate video-frame detection is camera-data work."""
    first_occurrence: dict[str, tuple[str, Path]] = {}
    duplicates: list[dict[str, str]] = []
    for split_name, directory in sources:
        for image_path in sorted(path for path in directory.rglob("*") if path.is_file()):
            try:
                file_hash = sha256_file(image_path)
            except OSError:
                continue
            existing = first_occurrence.get(file_hash)
            if existing is None:
                first_occurrence[file_hash] = (split_name, image_path)
                continue
            original_split, original_path = existing
            duplicates.append(
                {
                    "sha256": file_hash,
                    "first_split": original_split,
                    "first_image": str(original_path),
                    "duplicate_split": split_name,
                    "duplicate_image": str(image_path),
                    "cross_split": str(original_split != split_name),
                }
            )
    return duplicates


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.min_effective_face_pixels < 0:
        raise ValueError("--min-effective-face-pixels cannot be negative.")
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

    candidates: list[dict[str, str]] = []
    summaries = [
        audit_split(
            split_name="WIDER train",
            images_directory=wider_train_images,
            annotation_file=wider_train_annotations,
            min_effective_face_pixels=args.min_effective_face_pixels,
            candidates=candidates,
        ),
        audit_split(
            split_name="WIDER validation",
            images_directory=wider_val_images,
            annotation_file=wider_val_annotations,
            min_effective_face_pixels=args.min_effective_face_pixels,
            candidates=candidates,
        ),
        audit_split(
            split_name="UFDD external test",
            images_directory=ufdd_images,
            annotation_file=ufdd_annotations,
            min_effective_face_pixels=args.min_effective_face_pixels,
            candidates=candidates,
        ),
    ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [asdict(summary) for summary in summaries]
    write_csv(output_dir / "dataset_audit_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(output_dir / "dataset_quality_candidates.csv", candidates, ["split", "image", "reason"])

    duplicate_rows: list[dict[str, str]] = []
    if not args.skip_exact_duplicate_check:
        print("Scanning for exact byte-identical images; source files remain unchanged.")
        duplicate_rows = find_exact_duplicates(
            [
                ("WIDER train", wider_train_images),
                ("WIDER validation", wider_val_images),
                ("UFDD external test", ufdd_images),
            ]
        )
    write_csv(
        output_dir / "exact_duplicate_images.csv",
        duplicate_rows,
        ["sha256", "first_split", "first_image", "duplicate_split", "duplicate_image", "cross_split"],
    )

    report = {
        "dataset_root": str(dataset_root),
        "reference_image_size": REFERENCE_IMAGE_SIZE,
        "min_effective_face_pixels": args.min_effective_face_pixels,
        "splits": summary_rows,
        "quality_candidate_images": len(candidates),
        "exact_duplicate_pairs": len(duplicate_rows),
        "exact_cross_split_duplicate_pairs": sum(row["cross_split"] == "True" for row in duplicate_rows),
        "note": "The audit is read-only. Review candidates before excluding data; difficult but valid faces are not noise.",
    }
    (output_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for summary in summaries:
        print(
            f"[{summary.split}] images={summary.annotation_images:,}, valid-faces={summary.valid_boxes:,}, "
            f"invalid={summary.invalid_marked_boxes + summary.malformed_or_degenerate_boxes:,}, "
            f"tiny-only-images={summary.images_with_only_tiny_faces:,}, "
            f"missing-or-unreadable={summary.missing_images + summary.unreadable_images:,}"
        )
    print(f"Audit reports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Dataset audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
