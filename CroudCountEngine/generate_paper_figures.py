"""Generate numbered, publication-ready result figures from one input image.

This utility deliberately reuses the inference, tiling, ROI, NMS, and drawing
methods implemented in ``main.py``.  It does not create synthetic results.

Usage – detector result figures (requires a real crowd image):
    python CroudCountEngine/generate_paper_figures.py --input path\\to\\image.jpg

Usage – Related Work concept diagrams (no image required):
    python CroudCountEngine/generate_paper_figures.py --related-work
    python CroudCountEngine/generate_paper_figures.py --related-work --output-dir paper_figures

Both modes can be combined in one call:
    python CroudCountEngine/generate_paper_figures.py --input img.jpg --related-work

For a high-resolution crowded image, add --tiled.  The created folder contains
numbered PNGs and a manifest.json that records the exact settings used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# matplotlib is only imported when --related-work is used so the detector-only
# path does not require it.
def _mpl_imports():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    return plt, FancyBboxPatch, np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from main import VideoFeed, default_weights_path, parse_scales  # noqa: E402


def parse_roi(value: str) -> list[tuple[float, float]]:
    if not value.strip():
        return []
    points = []
    for pair in value.split(";"):
        x, y = pair.split(",", 1)
        points.append((float(x), float(y)))
    if len(points) < 3:
        raise ValueError("ROI needs at least three points: x1,y1;x2,y2;x3,y3")
    return points


def build_detector(args: argparse.Namespace) -> VideoFeed:
    """Build a VideoFeed-compatible detector without starting video threads."""
    detector = VideoFeed.__new__(VideoFeed)
    detector.device, detector.uses_cuda, detector.cpu_threads = VideoFeed.configure_runtime(
        args.cpu_threads, args.device
    )
    detector.tile_batch_size = detector.resolve_tile_batch_size(args.tile_batch_size)
    detector.model = YOLO(args.weights)
    detector.person_class_ids = [
        int(index)
        for index, name in detector.model.names.items()
        if str(name).strip().lower() in {"person", "head"}
    ]
    if not detector.person_class_ids:
        raise RuntimeError("The selected weights must contain a 'person' or 'head' class.")
    detector.conf = args.conf
    detector.iou = args.iou
    detector.imgsz = args.imgsz
    detector.max_det = args.max_det
    detector.tiled = args.tiled
    detector.tile_size = args.tile_size
    detector.tile_overlap = args.tile_overlap
    detector.min_box_area = args.min_box_area
    detector.max_box_area = args.max_box_area
    detector.min_box_height = args.min_box_height
    detector.max_box_width = args.max_box_width
    detector.max_box_height = args.max_box_height
    detector.min_aspect = args.min_aspect
    detector.max_aspect = args.max_aspect
    detector.center_merge = args.center_merge
    detector.center_merge_px = args.center_merge_px
    detector.scales = parse_scales(args.scales)
    detector.tta_flip = args.tta_flip
    detector.count_scale = args.count_scale
    detector.count_offset = args.count_offset
    detector.roi_points = parse_roi(args.roi)
    detector.roi_contour = detector.build_roi_contour(detector.roi_points)
    detector.roi_bounds = cv2.boundingRect(detector.roi_contour) if detector.roi_contour is not None else None
    return detector


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(result, (0, 0), (width + 24, height + 22), (20, 20, 20), -1)
    cv2.putText(result, text, (12, height + 8), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return result


def draw_tile_grid(image: np.ndarray, detector: VideoFeed) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    step = max(1, int(detector.tile_size * (1.0 - detector.tile_overlap)))
    for y in detector.tile_positions(height, detector.tile_size, step):
        for x in detector.tile_positions(width, detector.tile_size, step):
            x2, y2 = min(x + detector.tile_size, width), min(y + detector.tile_size, height)
            cv2.rectangle(result, (x, y), (x2, y2), (0, 165, 255), 2)
    return label(result, f"Tiled inference: {detector.tile_size}px, overlap {detector.tile_overlap:.0%}")


def draw_confidence_chart(scores: list[float], count: int) -> np.ndarray:
    canvas = np.full((540, 900, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Detection confidence distribution", (45, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Retained people: {count}", (45, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (55, 55, 55), 2, cv2.LINE_AA)
    values = sorted(scores, reverse=True)
    if not values:
        cv2.putText(canvas, "No person detections satisfied the current settings.", (45, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 180), 2, cv2.LINE_AA)
        return canvas
    chart_left, chart_right, base, top = 80, 850, 460, 150
    cv2.line(canvas, (chart_left, base), (chart_right, base), (0, 0, 0), 2)
    cv2.line(canvas, (chart_left, top), (chart_left, base), (0, 0, 0), 2)
    for tick in range(0, 11, 2):
        y = int(base - (base - top) * tick / 10)
        cv2.line(canvas, (chart_left - 8, y), (chart_left, y), (0, 0, 0), 1)
        cv2.putText(canvas, f"{tick / 10:.1f}", (22, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    bar_width = max(3, int((chart_right - chart_left) / max(1, len(values))))
    for index, score in enumerate(values):
        x1 = chart_left + index * bar_width
        y1 = int(base - (base - top) * score)
        cv2.rectangle(canvas, (x1, y1), (x1 + max(2, bar_width - 1), base), (30, 115, 190), -1)
    cv2.putText(canvas, "detections sorted by confidence", (320, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  RELATED WORK CONCEPT DIAGRAMS  (matplotlib, no YOLO required)
# ─────────────────────────────────────────────────────────────────────────────

# Shared colour palette
_C = dict(
    dark   = "#1a2e44",
    mid    = "#2e6da4",
    light  = "#a8c8e8",
    accent = "#e07b2a",
    green  = "#2ca02c",
    red    = "#d62728",
    grey   = "#f0f4f8",
    text   = "#1a1a1a",
)
_FONT = "DejaVu Sans"
_DPI  = 180


def _save_fig(fig, path: Path) -> None:
    fig.savefig(str(path), dpi=_DPI, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved  {path.name}")


def _rw_fig_a(output_dir: Path) -> None:
    """(a) Crowd Counting & Density Estimation – method evolution timeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    ax.annotate("", xy=(10.6, 2.5), xytext=(0.4, 2.5),
                arrowprops=dict(arrowstyle="-|>", color=_C["dark"], lw=2.2))

    milestones = [
        (1.0, "Regression\nNetworks",         "Pre-2016",           _C["mid"]),
        (3.0, "Density Map\nCNN",              "2016 – 2019",        _C["mid"]),
        (5.1, "Dilated /\nMulti-scale CNN",    "2019 – 2021\n[6]",   _C["mid"]),
        (7.1, "Attention &\nPoint Supervision","2021 – 2023\n[4]",   _C["accent"]),
        (9.2, "Transformer\nHybrid",           "2023 – 2026\n[3]",   _C["accent"]),
    ]
    for x, lbl, yr, ec in milestones:
        ax.plot(x, 2.5, "o", ms=14, color=ec, zorder=5)
        ax.text(x, 3.05, lbl, ha="center", va="bottom", fontsize=8.5,
                color=_C["text"], fontfamily=_FONT, multialignment="center")
        ax.text(x, 1.85, yr, ha="center", va="top", fontsize=7.5,
                color=_C["mid"], fontfamily=_FONT, multialignment="center")

    challenges = ["Occlusion", "Perspective\ndistortion", "Poor\nillumination",
                  "Scale\nvariation", "Cross-dataset\ngeneralisation"]
    bw, bh, by = 1.55, 0.52, 0.45
    for i, ch in enumerate(challenges):
        rect = FancyBboxPatch((0.6 + i * 1.9, by), bw, bh,
                              boxstyle="round,pad=0.05",
                              linewidth=1.2, edgecolor=_C["red"],
                              facecolor="#fde8e8")
        ax.add_patch(rect)
        ax.text(0.6 + i * 1.9 + bw / 2, by + bh / 2, ch,
                ha="center", va="center", fontsize=7.8, color=_C["red"],
                fontfamily=_FONT, multialignment="center")

    ax.text(5.5, 0.05, "Common challenges across all generations  [1][9]",
            ha="center", va="bottom", fontsize=8, color=_C["red"],
            fontstyle="italic", fontfamily=_FONT)
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.2)
    ax.set_title("(a)  Crowd Counting and Density Estimation – Method Evolution",
                 fontsize=11, fontweight="bold", color=_C["dark"], pad=10,
                 fontfamily=_FONT)
    fig.tight_layout(pad=0.4)
    _save_fig(fig, output_dir / "RW_Fig_A_crowd_counting_timeline.png")


def _rw_fig_b(output_dir: Path) -> None:
    """(b) Person Detection & Tracking – pipeline block diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    blocks = [
        (1.1,  2.4, 1.6, 0.72, "Camera /\nVideo Input",           _C["grey"],    _C["dark"]),
        (3.2,  2.4, 1.6, 0.72, "YOLO-Based\nDetector  [3][4]",    _C["light"],   _C["mid"]),
        (5.3,  2.4, 1.6, 0.72, "Bounding Box\nFiltering & NMS",   _C["light"],   _C["mid"]),
        (7.4,  2.4, 1.6, 0.72, "ByteTrack\nTracker  [11]",        _C["light"],   _C["accent"]),
        (9.5,  2.4, 1.6, 0.72, "Stable Count\n& Track IDs",       "#e8f5e9",     _C["green"]),
        (11.0, 2.4, 1.0, 0.72, "Dashboard /\nAlert",              "#fde8e8",     _C["red"]),
    ]
    for xc, yc, bw, bh, txt, fc, ec in blocks:
        rect = FancyBboxPatch((xc - bw/2, yc - bh/2), bw, bh,
                              boxstyle="round,pad=0.07",
                              linewidth=1.6, edgecolor=ec, facecolor=fc)
        ax.add_patch(rect)
        ax.text(xc, yc, txt, ha="center", va="center", fontsize=8.5,
                color=_C["text"], fontfamily=_FONT, multialignment="center")

    arrow_xs = [(1.9, 2.4), (3.0, 4.0), (4.1, 5.0), (6.1, 6.3),
                (8.2, 8.7), (10.3, 10.5)]
    for x1, x2 in arrow_xs:
        ax.annotate("", xy=(x2, 2.4), xytext=(x1, 2.4),
                    arrowprops=dict(arrowstyle="-|>", color=_C["dark"], lw=1.8))

    notes = [
        (3.2, 1.55, "Motion feature\nmaps  [3]"),
        (5.3, 1.55, "Confidence threshold\n+ geometry filter"),
        (7.4, 1.55, "Low-score detection\nassociation  [11]"),
    ]
    for xn, yn, txt in notes:
        ax.text(xn, yn, txt, ha="center", va="top", fontsize=7.5,
                color=_C["mid"], fontstyle="italic", fontfamily=_FONT,
                multialignment="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=_C["light"], lw=1))

    ax.text(5.8, 3.45, "Multi-camera / density-fusion extension  [5]",
            ha="center", va="bottom", fontsize=8, color=_C["accent"],
            fontstyle="italic", fontfamily=_FONT)
    ax.set_xlim(0, 12); ax.set_ylim(0.9, 3.9)
    ax.set_title("(b)  Person Detection and Tracking – Pipeline Overview",
                 fontsize=11, fontweight="bold", color=_C["dark"], pad=10,
                 fontfamily=_FONT)
    fig.tight_layout(pad=0.4)
    _save_fig(fig, output_dir / "RW_Fig_B_detection_tracking_pipeline.png")


def _rw_fig_c(output_dir: Path) -> None:
    """(c) Anomaly & Behaviour Analysis – three complementary approaches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    fig.patch.set_facecolor("white")

    titles   = ["Handcrafted\nFeature Methods",
                "Deep Spatiotemporal\nCNN / Autoencoder  [8]",
                "Trajectory-Based\nZone Monitoring  [7]"]
    subtitles = ["Optical flow + SVM /\nrandom forest",
                 "Frame-level AUC on\nUCSD, Avenue, SHT-Anom.",
                 "Predict future zone\nviolations proactively"]
    colours   = [_C["light"], _C["mid"], _C["accent"]]

    for idx, (ax, title, subtitle, col) in enumerate(
            zip(axes, titles, subtitles, colours)):
        ax.set_facecolor("white"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis("off")
        from matplotlib.patches import FancyBboxPatch
        scene = FancyBboxPatch((0.5, 1.5), 9, 7, boxstyle="round,pad=0.1",
                               linewidth=2, edgecolor=col, facecolor=_C["grey"])
        ax.add_patch(scene)

        if idx == 0:
            for gx in np.arange(1.5, 9, 1.5):
                for gy in np.arange(2.5, 8, 1.5):
                    angle = np.random.uniform(-0.6, 0.6)
                    dx, dy = 0.7 * np.cos(angle), 0.7 * np.sin(angle)
                    ax.annotate("", xy=(gx + dx, gy + dy), xytext=(gx, gy),
                                arrowprops=dict(arrowstyle="-|>",
                                                color=_C["mid"], lw=1.4))
            ax.text(5, 0.8, "Velocity / motion vectors", ha="center",
                    fontsize=8, color=_C["mid"], fontfamily=_FONT)

        elif idx == 1:
            x_g = np.linspace(0.5, 9.5, 60); y_g = np.linspace(1.5, 8.5, 60)
            X, Y = np.meshgrid(x_g, y_g)
            Z = (np.exp(-((X - 4)**2 + (Y - 5.5)**2) / 3) +
                 0.4 * np.exp(-((X - 7)**2 + (Y - 3)**2) / 2))
            ax.contourf(X, Y, Z, levels=8, cmap="RdYlGn_r", alpha=0.55)
            ax.text(4, 5.5, "Anomaly\nhot-spot", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold",
                    fontfamily=_FONT)
            ax.text(5, 0.8, "AUC-based frame scoring", ha="center",
                    fontsize=8, color=_C["red"], fontfamily=_FONT)

        else:
            zone = plt.Polygon([(2, 3), (7.5, 3), (8, 7.5), (1.5, 7.5)],
                               closed=True, linewidth=2,
                               edgecolor=_C["red"], facecolor="#fde8e8",
                               alpha=0.5)
            ax.add_patch(zone)
            ax.text(4.8, 5.2, "Restricted\nZone", ha="center", va="center",
                    fontsize=9, color=_C["red"], fontweight="bold",
                    fontfamily=_FONT)
            traj_x = [0.8, 1.5, 2.3, 3.2]; traj_y = [4.5, 4.6, 4.8, 5.0]
            ax.plot(traj_x, traj_y, "-o", color=_C["accent"], lw=2, ms=6,
                    zorder=5)
            ax.annotate("", xy=(3.8, 5.1), xytext=(3.2, 5.0),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=_C["accent"], lw=2))
            ax.text(1.6, 3.8, "Predicted\npath", ha="center", fontsize=7.5,
                    color=_C["accent"], fontfamily=_FONT)
            ax.text(5, 0.8, "Proactive violation prediction", ha="center",
                    fontsize=8, color=_C["red"], fontfamily=_FONT)

        ax.set_title(title, fontsize=9.5, fontweight="bold", color=_C["dark"],
                     pad=6, fontfamily=_FONT)
        ax.text(5, 9.4, subtitle, ha="center", va="top", fontsize=8,
                color=_C["mid"], fontstyle="italic", fontfamily=_FONT,
                multialignment="center")

    fig.suptitle("(c)  Anomaly and Behaviour Analysis – Three Complementary Approaches",
                 fontsize=11, fontweight="bold", color=_C["dark"], y=1.01,
                 fontfamily=_FONT)
    fig.tight_layout(pad=0.6)
    _save_fig(fig, output_dir / "RW_Fig_C_anomaly_behaviour_analysis.png")


def _rw_fig_d(output_dir: Path) -> None:
    """(d) Alerting & IoT / Edge Deployment – end-to-end system flow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)

    def box(xc, yc, bw, bh, txt, fc, ec, fs=8.8):
        rect = FancyBboxPatch((xc - bw/2, yc - bh/2), bw, bh,
                              boxstyle="round,pad=0.08",
                              linewidth=1.8, edgecolor=ec, facecolor=fc)
        ax.add_patch(rect)
        ax.text(xc, yc, txt, ha="center", va="center", fontsize=fs,
                color=_C["text"], fontfamily=_FONT, multialignment="center")

    def arrow(x1, y1, x2, y2, lbl="", col=_C["dark"]):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        if lbl:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, lbl, ha="center",
                    va="bottom", fontsize=7.5, color=col,
                    fontfamily=_FONT, fontstyle="italic")

    for yl, txt, col in [(5.3, "SENSING LAYER",            _C["mid"]),
                          (3.7, "EDGE / INFERENCE LAYER",  _C["accent"]),
                          (2.1, "APPLICATION LAYER",        _C["mid"]),
                          (0.6, "NOTIFICATION LAYER",       _C["red"])]:
        ax.text(0.15, yl, txt, ha="left", va="center", fontsize=7.2,
                color=col, fontweight="bold", fontfamily=_FONT, alpha=0.7)

    for yd in [4.55, 2.95, 1.35]:
        ax.axhline(yd, color="#cccccc", lw=0.8, ls="--")

    box(2.5,  5.0, 1.8, 0.7,  "CCTV /\nWebcam",                 _C["grey"],   _C["dark"])
    box(5.5,  5.0, 1.8, 0.7,  "IP Camera\nNetwork",              _C["grey"],   _C["dark"])
    box(8.5,  5.0, 1.8, 0.7,  "IoT Sensor\n(ESP32)  [13]",       _C["grey"],   _C["dark"])
    box(2.5,  3.5, 2.0, 0.75, "YOLO Detector\n(Edge / GPU)",     _C["light"],  _C["mid"])
    box(5.5,  3.5, 2.0, 0.75, "Raspberry Pi /\nEdge Node  [14]", _C["light"],  _C["mid"])
    box(8.5,  3.5, 2.0, 0.75, "Tiled Inference\n+ NMS Filter",   _C["light"],  _C["mid"])
    box(3.5,  2.0, 2.2, 0.75, "Django Back-end\n(Camera + Records)  [2]", "#e8f0fe", _C["mid"])
    box(7.5,  2.0, 2.2, 0.75, "Threshold\nEvaluator  [10]",      "#e8f0fe",   _C["accent"])
    box(2.2,  0.75, 1.7, 0.65, "Dashboard\nAlert",               "#fde8e8",   _C["red"])
    box(5.0,  0.75, 1.7, 0.65, "Mobile /\nEmail Push",           "#fde8e8",   _C["red"])
    box(7.8,  0.75, 1.7, 0.65, "Physical\nSiren / LED",          "#fde8e8",   _C["red"])
    box(10.2, 0.75, 1.4, 0.65, "Log &\nAudit",                   "#fde8e8",   _C["red"])

    for xc in [2.5, 5.5, 8.5]:
        arrow(xc, 4.65, xc, 3.88)
    arrow(2.5, 3.12, 3.2, 2.38)
    arrow(5.5, 3.12, 5.0, 2.38)
    arrow(8.5, 3.12, 7.8, 2.38)
    arrow(4.6, 2.0,  6.4, 2.0, "count + timestamp", _C["accent"])
    arrow(3.5, 1.62, 2.5,  1.08)
    arrow(3.5, 1.62, 5.0,  1.08)
    arrow(7.5, 1.62, 7.8,  1.08)
    arrow(7.5, 1.62, 10.2, 1.08)

    ax.set_title("(d)  Alerting and IoT / Edge Deployment – End-to-End System Flow",
                 fontsize=11, fontweight="bold", color=_C["dark"], pad=10,
                 fontfamily=_FONT)
    fig.tight_layout(pad=0.4)
    _save_fig(fig, output_dir / "RW_Fig_D_alerting_iot_edge_deployment.png")


def generate_related_work_figures(output_dir: Path) -> list[str]:
    """Generate all 4 Related Work concept diagrams. Returns list of filenames."""
    print("Generating Related Work figures …")
    _rw_fig_a(output_dir)
    _rw_fig_b(output_dir)
    _rw_fig_c(output_dir)
    _rw_fig_d(output_dir)
    names = [
        "RW_Fig_A_crowd_counting_timeline.png",
        "RW_Fig_B_detection_tracking_pipeline.png",
        "RW_Fig_C_anomaly_behaviour_analysis.png",
        "RW_Fig_D_alerting_iot_edge_deployment.png",
    ]
    print(f"Related Work figures saved to: {output_dir}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate numbered manuscript figures from one real input image.")
    parser.add_argument("--input", default=None, help="Path to a consented input image (required unless --related-work only).")
    parser.add_argument("--related-work", action="store_true", default=False,
                        help="Generate the 4 Related Work concept diagrams (no input image needed).")
    parser.add_argument("--output-dir", default="paper_figures", help="Folder for numbered PNG outputs.")
    parser.add_argument("--weights", default=str(default_weights_path()))
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--tile-batch-size", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--tta-flip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--count-scale", type=float, default=1.0)
    parser.add_argument("--count-offset", type=float, default=0.0)
    parser.add_argument("--roi", default="", help="Optional ROI polygon: x1,y1;x2,y2;x3,y3")
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

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Related Work diagrams (no input image needed) ─────────────────────────
    if args.related_work:
        rw_files = generate_related_work_figures(output_dir)
        if args.input is None:
            # Only related-work mode requested – done.
            print(json.dumps({"related_work_files": rw_files}, indent=2))
            return

    # ── Detector result figures (requires --input) ────────────────────────────
    if args.input is None:
        parser.error("--input is required when not using --related-work alone.")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"OpenCV could not read {input_path}")
    detector = build_detector(args)
    started = time.perf_counter()
    boxes, scores = detector.detect_tiled(image) if args.tiled else detector.detect_image(image)
    boxes, scores = detector.merge_boxes(boxes, scores) if not args.tiled else (boxes, scores)
    elapsed_ms = (time.perf_counter() - started) * 1000
    raw_count = len(boxes); adjusted_count = detector.apply_count_correction(raw_count)
    annotated = image.copy(); detector.draw_roi(annotated); detector.draw_boxes(annotated, boxes, scores); detector.draw_count_overlay(annotated, raw_count, adjusted_count, adjusted_count)
    roi_view = image.copy(); detector.draw_roi(roi_view); roi_view = label(roi_view, "Region of interest" if detector.roi_contour is not None else "No ROI configured")
    save(output_dir / "Fig_01_input_image.png", label(image, "Input image"))
    save(output_dir / "Fig_02_person_detections.png", annotated)
    save(output_dir / "Fig_03_roi_or_tile_layout.png", draw_tile_grid(image, detector) if args.tiled else roi_view)
    save(output_dir / "Fig_04_detection_confidence.png", draw_confidence_chart(scores, raw_count))
    manifest = {
        "input": str(input_path), "weights": str(Path(args.weights).resolve()), "raw_count": raw_count,
        "adjusted_count": adjusted_count, "inference_ms": round(elapsed_ms, 2), "settings": vars(args),
        "files": ["Fig_01_input_image.png", "Fig_02_person_detections.png", "Fig_03_roi_or_tile_layout.png", "Fig_04_detection_confidence.png"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
