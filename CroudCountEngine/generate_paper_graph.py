"""Create a reproducible grouped-bar graph from selected real detector inputs.

The graph reports frame-level detector output (retained-person count and mean
confidence); it is not an accuracy benchmark. Use ground-truth annotations and
the evaluation script for formal MAE, RMSE, precision, recall, and mAP claims.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from generate_paper_figures import build_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a paper graph from actual crowd-detector runs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Two or more input images.")
    parser.add_argument("--output-dir", default="paper_figures", help="Output directory.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--tile-batch-size", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--tta-flip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--count-scale", type=float, default=1.0)
    parser.add_argument("--count-offset", type=float, default=0.0)
    parser.add_argument("--roi", default="")
    parser.add_argument("--min-box-area", type=float, default=0.0)
    parser.add_argument("--max-box-area", type=float, default=0.0)
    parser.add_argument("--min-box-height", type=float, default=0.0)
    parser.add_argument("--max-box-width", type=float, default=0.0)
    parser.add_argument("--max-box-height", type=float, default=0.0)
    parser.add_argument("--min-aspect", type=float, default=0.0)
    parser.add_argument("--max-aspect", type=float, default=0.0)
    parser.add_argument("--center-merge", type=float, default=0.0)
    parser.add_argument("--center-merge-px", type=float, default=0.0)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    detector = build_detector(args)
    rows = []
    for index, name in enumerate(args.inputs, 1):
        path = Path(name).resolve(); image = cv2.imread(str(path))
        if image is None: raise ValueError(f"Cannot read {path}")
        boxes, scores = detector.detect_tiled(image) if args.tiled else detector.detect_image(image)
        if not args.tiled: boxes, scores = detector.merge_boxes(boxes, scores)
        rows.append({"frame": f"Frame {index}", "source": path.name, "count": len(boxes), "mean_confidence": float(np.mean(scores)) if scores else 0.0})
    labels = [row["frame"] for row in rows]; x = np.arange(len(rows)); width = 0.36
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8), dpi=220)
    bars1 = ax1.bar(x-width/2, [r["count"] for r in rows], width, label="Retained people", color="#2878B5")
    ax1.set_ylabel("Retained detections")
    ax1.set_xticks(x, labels); ax1.set_xlabel("Selected input frames")
    ax1.grid(axis="y", linestyle=":", alpha=.55)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x+width/2, [r["mean_confidence"] for r in rows], width, label="Mean confidence", color="#E17C05")
    ax2.set_ylabel("Mean confidence"); ax2.set_ylim(0, 1.05)
    ax1.set_title("Detector Output Summary Across Selected Input Frames", pad=12)
    for bar in bars1: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.2, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2: ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax1.legend([bars1, bars2], ["Retained people", "Mean confidence"], loc="upper left", frameon=True)
    fig.tight_layout()
    figure_path = out / "Fig_05_detector_output_summary.png"; fig.savefig(figure_path, bbox_inches="tight"); plt.close(fig)
    with (out / "Fig_05_detector_output_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream, fieldnames=["frame","source","count","mean_confidence"]); writer.writeheader(); writer.writerows(rows)
    print(figure_path)


if __name__ == "__main__":
    main()
