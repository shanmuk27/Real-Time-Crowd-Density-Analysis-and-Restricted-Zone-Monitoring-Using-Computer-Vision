import argparse
from collections import deque
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_VIDEO_PATH = Path(
    r"C:\infosys_crowdcount_october2025_Shanmuk\uploads\istockphoto-1471311788-640_adpp_is.mp4"
)
TRAINED_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "runs" / "crowd_train" / "weights" / "best.pt"


class VideoFeed:
    def __init__(
        self,
        video_path: str,
        model_path: str,
        conf: float,
        iou: float,
        imgsz: int,
        tiled: bool,
        tile_size: int,
        tile_overlap: float,
        tile_batch_size: int,
        cpu_threads: int,
        smooth: int,
        min_box_area: float,
        max_box_area: float,
        min_box_height: float,
        max_box_width: float,
        max_box_height: float,
        min_aspect: float,
        max_aspect: float,
        center_merge: float,
        center_merge_px: float,
        scales: list[float],
        tta_flip: bool,
        count_scale: float,
        count_offset: float,
        process_every: int,
    ):
        self.video_path = video_path
        self.cpu_threads = self.configure_runtime(cpu_threads)
        self.tile_batch_size = self.resolve_tile_batch_size(tile_batch_size)
        self.cap = self.open_capture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {video_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.count_history = deque(maxlen=max(1, smooth))
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_box_height = min_box_height
        self.max_box_width = max_box_width
        self.max_box_height = max_box_height
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.center_merge = center_merge
        self.center_merge_px = center_merge_px
        self.scales = scales
        self.tta_flip = tta_flip
        self.count_scale = count_scale
        self.count_offset = count_offset
        self.process_every = max(1, process_every)
        self.frame_index = 0
        self.last_annotated_frame = None
        self.last_count = 0
        self.last_detection_ms = 0.0
        self.average_detection_ms = None
        print(f"[model] Loaded weights: {model_path}")
        print(
            f"[mode] {'Tiled accuracy mode' if tiled else 'Single-frame speed mode'} "
            f"conf={conf}, iou={iou}, imgsz={imgsz}, process_every={self.process_every}, "
            f"scales={self.scales}, tta_flip={self.tta_flip}"
        )
        print(
            f"[performance] CPU threads={self.cpu_threads}, "
            f"tile batch={self.tile_batch_size if tiled else 'not used'}"
        )

    @staticmethod
    def configure_runtime(requested_threads: int):
        logical_cpus = os.cpu_count() or 1
        auto_threads = max(1, round(logical_cpus * 0.8))
        thread_count = requested_threads if requested_threads > 0 else auto_threads
        thread_count = min(logical_cpus, max(1, thread_count))

        torch.set_num_threads(thread_count)
        try:
            # A single coordinator avoids nested CPU pools competing with inference workers.
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        try:
            cv2.setNumThreads(thread_count)
        except cv2.error:
            pass
        return thread_count

    def resolve_tile_batch_size(self, requested_batch_size: int):
        if requested_batch_size > 0:
            return requested_batch_size
        # Four 512px tiles fit comfortably on typical laptops while removing model-call overhead.
        return min(4, self.cpu_threads)

    @staticmethod
    def open_capture(video_path):
        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return cv2.VideoCapture(str(video_path))

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, 0

        self.frame_index += 1
        if self.process_every > 1 and self.frame_index % self.process_every != 1:
            if self.last_annotated_frame is not None:
                return self.last_annotated_frame.copy(), self.last_count

        detection_started_at = time.perf_counter()
        if self.tiled:
            boxes, scores = self.detect_tiled(frame)
            raw_count = len(boxes)
            annotated_frame = frame.copy()
            self.draw_boxes(annotated_frame, boxes, scores)
        else:
            boxes, scores = self.detect_image(frame)
            boxes, scores = self.merge_boxes(boxes, scores)
            raw_count = len(boxes)
            annotated_frame = frame.copy()
            self.draw_boxes(annotated_frame, boxes, scores)

        self.last_detection_ms = (time.perf_counter() - detection_started_at) * 1000.0
        if self.average_detection_ms is None:
            self.average_detection_ms = self.last_detection_ms
        else:
            self.average_detection_ms = (self.average_detection_ms * 0.9) + (self.last_detection_ms * 0.1)

        adjusted_count = self.apply_count_correction(raw_count)
        self.count_history.append(adjusted_count)
        smoothed_count = round(sum(self.count_history) / len(self.count_history))
        self.draw_count_overlay(annotated_frame, raw_count, adjusted_count, smoothed_count)
        self.last_annotated_frame = annotated_frame
        self.last_count = adjusted_count
        return annotated_frame, adjusted_count

    def apply_count_correction(self, raw_count: int):
        return max(0, round((raw_count * self.count_scale) + self.count_offset))

    def detect_image(self, image, x_offset: float = 0.0, y_offset: float = 0.0):
        all_boxes = []
        all_scores = []

        for scale in self.scales:
            scaled_image = self.resize_for_scale(image, scale)
            boxes, scores = self.run_model_on_image(scaled_image)
            mapped_boxes, mapped_scores = self.map_detections(
                boxes, scores, x_offset, y_offset, scale, flipped=False, image_width=image.shape[1]
            )
            all_boxes.extend(mapped_boxes)
            all_scores.extend(mapped_scores)

            if self.tta_flip:
                flipped_image = cv2.flip(scaled_image, 1)
                boxes, scores = self.run_model_on_image(flipped_image)
                mapped_boxes, mapped_scores = self.map_detections(
                    boxes, scores, x_offset, y_offset, scale, flipped=True, image_width=image.shape[1]
                )
                all_boxes.extend(mapped_boxes)
                all_scores.extend(mapped_scores)

        return all_boxes, all_scores

    @staticmethod
    def resize_for_scale(image, scale: float):
        if abs(scale - 1.0) < 0.001:
            return image
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    def run_model_on_image(self, image):
        result = self.run_model_on_images([image])[0]
        return self.extract_result_arrays(result)

    def run_model_on_images(self, images):
        if len(images) == 1:
            return self.model(images[0], conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        return self.model.predict(images, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)

    @staticmethod
    def extract_result_arrays(result):
        if result.boxes is None or len(result.boxes) == 0:
            return [], []
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()

    def map_detections(self, boxes, scores, x_offset: float, y_offset: float, scale: float, flipped: bool, image_width: int):
        if len(boxes) == 0:
            return [], []

        mapped_boxes = np.asarray(boxes, dtype=np.float64).copy() / scale
        if flipped:
            original_x1 = mapped_boxes[:, 0].copy()
            mapped_boxes[:, 0] = image_width - mapped_boxes[:, 2]
            mapped_boxes[:, 2] = image_width - original_x1
        mapped_boxes[:, [0, 2]] += x_offset
        mapped_boxes[:, [1, 3]] += y_offset

        valid = self.valid_box_mask(mapped_boxes)
        mapped_scores = np.asarray(scores, dtype=np.float64)
        return mapped_boxes[valid].tolist(), mapped_scores[valid].tolist()

    def extract_boxes(self, result, x_offset: float = 0.0, y_offset: float = 0.0):
        boxes, scores = self.extract_result_arrays(result)
        return self.map_detections(boxes, scores, x_offset, y_offset, 1.0, False, 0)

    def is_valid_box(self, box):
        x1, y1, x2, y2 = box
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        aspect = width / height if height else 0.0
        if height < self.min_box_height:
            return False
        if self.max_box_width and width > self.max_box_width:
            return False
        if self.max_box_height and height > self.max_box_height:
            return False
        if aspect < self.min_aspect or aspect > self.max_aspect:
            return False
        if self.min_box_area and area < self.min_box_area:
            return False
        if self.max_box_area and area > self.max_box_area:
            return False
        return True

    def valid_box_mask(self, boxes):
        widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        areas = widths * heights
        aspects = np.divide(widths, heights, out=np.zeros_like(widths), where=heights != 0)

        valid = heights >= self.min_box_height
        if self.max_box_width:
            valid &= widths <= self.max_box_width
        if self.max_box_height:
            valid &= heights <= self.max_box_height
        valid &= (aspects >= self.min_aspect) & (aspects <= self.max_aspect)
        if self.min_box_area:
            valid &= areas >= self.min_box_area
        if self.max_box_area:
            valid &= areas <= self.max_box_area
        return valid

    def detect_tiled(self, frame):
        height, width = frame.shape[:2]
        step = max(1, int(self.tile_size * (1.0 - self.tile_overlap)))
        all_boxes = []
        all_scores = []

        y_positions = self.tile_positions(height, self.tile_size, step)
        x_positions = self.tile_positions(width, self.tile_size, step)

        tiles = []
        for y1 in y_positions:
            for x1 in x_positions:
                x2 = min(x1 + self.tile_size, width)
                y2 = min(y1 + self.tile_size, height)
                tiles.append((frame[y1:y2, x1:x2], x1, y1))

        for scale in self.scales:
            boxes, scores = self.detect_tile_batch(tiles, scale, flipped=False)
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            if self.tta_flip:
                boxes, scores = self.detect_tile_batch(tiles, scale, flipped=True)
                all_boxes.extend(boxes)
                all_scores.extend(scores)

        return self.merge_boxes(all_boxes, all_scores)

    def detect_tile_batch(self, tiles, scale: float, flipped: bool):
        all_boxes = []
        all_scores = []

        for start in range(0, len(tiles), self.tile_batch_size):
            tile_batch = tiles[start : start + self.tile_batch_size]
            images = []
            for tile, _, _ in tile_batch:
                scaled_tile = self.resize_for_scale(tile, scale)
                images.append(cv2.flip(scaled_tile, 1) if flipped else scaled_tile)

            results = self.run_model_on_images(images)
            for (tile, x_offset, y_offset), result in zip(tile_batch, results):
                boxes, scores = self.extract_result_arrays(result)
                mapped_boxes, mapped_scores = self.map_detections(
                    boxes,
                    scores,
                    x_offset,
                    y_offset,
                    scale,
                    flipped,
                    image_width=tile.shape[1],
                )
                all_boxes.extend(mapped_boxes)
                all_scores.extend(mapped_scores)

        return all_boxes, all_scores

    @staticmethod
    def tile_positions(length: int, tile_size: int, step: int):
        if length <= tile_size:
            return [0]
        positions = list(range(0, length - tile_size + 1, step))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def merge_boxes(self, boxes, scores):
        if not boxes:
            return [], []

        nms_boxes = []
        for x1, y1, x2, y2 in boxes:
            nms_boxes.append([int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))])

        keep = cv2.dnn.NMSBoxes(nms_boxes, scores, self.conf, self.iou)
        if len(keep) == 0:
            return [], []

        keep_indexes = np.array(keep).reshape(-1).tolist()
        kept_boxes = [boxes[index] for index in keep_indexes]
        kept_scores = [scores[index] for index in keep_indexes]
        return self.merge_nearby_centers(kept_boxes, kept_scores)

    def merge_nearby_centers(self, boxes, scores):
        if not boxes:
            return [], []

        sorted_indexes = sorted(range(len(boxes)), key=lambda index: scores[index], reverse=True)
        merged_boxes = []
        merged_scores = []
        accepted_by_cell = {}
        max_merge_distance = self.maximum_center_merge_distance()

        if max_merge_distance is None:
            return self.merge_nearby_centers_linear(boxes, scores, sorted_indexes)

        cell_size = max(1.0, max_merge_distance)

        for index in sorted_indexes:
            candidate = boxes[index]
            candidate_geometry = self.box_geometry(candidate)
            cell = self.geometry_cell(candidate_geometry, cell_size)
            if self.is_duplicate_in_nearby_cells(candidate_geometry, cell, accepted_by_cell):
                continue
            merged_boxes.append(candidate)
            merged_scores.append(scores[index])
            accepted_by_cell.setdefault(cell, []).append(candidate_geometry)

        return merged_boxes, merged_scores

    def merge_nearby_centers_linear(self, boxes, scores, sorted_indexes):
        merged_boxes = []
        merged_scores = []
        for index in sorted_indexes:
            candidate = boxes[index]
            if self.is_duplicate_center(candidate, merged_boxes):
                continue
            merged_boxes.append(candidate)
            merged_scores.append(scores[index])
        return merged_boxes, merged_scores

    def maximum_center_merge_distance(self):
        width_bounds = []
        height_bounds = []
        if self.max_box_width:
            width_bounds.append(self.max_box_width)
            if self.min_aspect > 0:
                height_bounds.append(self.max_box_width / self.min_aspect)
        if self.max_box_height:
            height_bounds.append(self.max_box_height)
            if self.max_aspect > 0:
                width_bounds.append(self.max_box_height * self.max_aspect)
        if self.max_box_area:
            if self.max_aspect > 0:
                width_bounds.append((self.max_box_area * self.max_aspect) ** 0.5)
            if self.min_aspect > 0:
                height_bounds.append((self.max_box_area / self.min_aspect) ** 0.5)
        if not width_bounds or not height_bounds:
            return None

        largest_box_dimension = max(min(width_bounds), min(height_bounds))
        return max(self.center_merge_px, self.center_merge * largest_box_dimension)

    @staticmethod
    def geometry_cell(geometry, cell_size: float):
        center_x, center_y, _, _ = geometry
        return int(center_x // cell_size), int(center_y // cell_size)

    def is_duplicate_in_nearby_cells(self, candidate_geometry, cell, accepted_by_cell):
        cell_x, cell_y = cell
        for y_index in range(cell_y - 1, cell_y + 2):
            for x_index in range(cell_x - 1, cell_x + 2):
                for accepted_geometry in accepted_by_cell.get((x_index, y_index), []):
                    if self.are_duplicate_geometries(candidate_geometry, accepted_geometry):
                        return True
        return False

    def is_duplicate_center(self, candidate, accepted_boxes):
        candidate_geometry = self.box_geometry(candidate)
        for accepted in accepted_boxes:
            if self.are_duplicate_geometries(candidate_geometry, self.box_geometry(accepted)):
                return True
        return False

    def are_duplicate_geometries(self, candidate, accepted):
        cx, cy, width, height = candidate
        ax, ay, accepted_width, accepted_height = accepted
        center_distance = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
        scale = max(width, height, accepted_width, accepted_height)
        threshold = max(self.center_merge_px, self.center_merge * scale)
        return center_distance <= threshold

    @staticmethod
    def box_geometry(box):
        x1, y1, x2, y2 = box
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        return x1 + width / 2.0, y1 + height / 2.0, width, height

    @staticmethod
    def draw_boxes(frame, boxes, scores):
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(value) for value in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
            cv2.putText(
                frame,
                f"{score:.2f}",
                (x1, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 220, 80),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def draw_count_overlay(frame, raw_count: int, adjusted_count: int, smoothed_count: int):
        label = f"Crowd Count: {smoothed_count}"
        if smoothed_count != adjusted_count or adjusted_count != raw_count:
            label = f"{label}  raw: {raw_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_width, text_height = text_size
        x, y = 18, 42

        cv2.rectangle(
            frame,
            (x - 10, y - text_height - 14),
            (x + text_width + 10, y + 12),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, label, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    def release(self):
        self.cap.release()

    def performance_summary(self):
        if not self.average_detection_ms:
            return "detector timing unavailable"
        detector_fps = 1000.0 / self.average_detection_ms
        return f"detect={self.last_detection_ms:.0f}ms avg={self.average_detection_ms:.0f}ms ({detector_fps:.1f} FPS)"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run crowd detection inference on a video feed."
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO_PATH),
        help="Path to the input video.",
    )
    parser.add_argument(
        "--source",
        type=int,
        choices=[0, 1],
        default=None,
        help="Input source switch: 0 = use device camera, 1 = use --video.",
    )
    parser.add_argument(
        "--weights",
        default=str(TRAINED_WEIGHTS_PATH if TRAINED_WEIGHTS_PATH.exists() else "yolov8n.pt"),
        help="Path to YOLO weights. Defaults to trained weights if available.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.08,
        help="Detection confidence threshold. Lower values count more detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.20,
        help="NMS IoU threshold. Higher values keep more overlapping detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Match training size unless you need more detail.",
    )
    parser.add_argument(
        "--tiled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use overlapping tiled inference. Slower, but usually better for small dense heads.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size for tiled inference.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.30,
        help="Tile overlap ratio for tiled inference.",
    )
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=0,
        help="Tiles per YOLO call. 0 automatically uses a safe batch of four.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU threads for inference. 0 automatically uses about 80%% of available threads.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Average the displayed count over this many frames.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=8.0,
        help="Ignore boxes smaller than this pixel area. Increase to remove tiny noise.",
    )
    parser.add_argument(
        "--max-box-area",
        type=float,
        default=20000.0,
        help="Ignore boxes larger than this pixel area. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-box-height",
        type=float,
        default=3.0,
        help="Ignore boxes shorter than this many pixels.",
    )
    parser.add_argument(
        "--max-box-width",
        type=float,
        default=0.0,
        help="Ignore boxes wider than this many pixels. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-box-height",
        type=float,
        default=0.0,
        help="Ignore boxes taller than this many pixels. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-aspect",
        type=float,
        default=0.35,
        help="Ignore boxes narrower than this width/height ratio.",
    )
    parser.add_argument(
        "--max-aspect",
        type=float,
        default=1.80,
        help="Ignore boxes wider than this width/height ratio.",
    )
    parser.add_argument(
        "--center-merge",
        type=float,
        default=0.75,
        help="Merge boxes whose centers are close relative to box size.",
    )
    parser.add_argument(
        "--center-merge-px",
        type=float,
        default=22.0,
        help="Minimum pixel distance used for center-based duplicate merging.",
    )
    parser.add_argument(
        "--scales",
        default="1.0",
        help="Comma-separated inference scales, for example 0.75,1.0,1.25. More scales are slower.",
    )
    parser.add_argument(
        "--tta-flip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run horizontal flip test-time augmentation and merge detections.",
    )
    parser.add_argument(
        "--count-scale",
        type=float,
        default=1.0,
        help="Multiply the detected count by this correction factor.",
    )
    parser.add_argument(
        "--count-offset",
        type=float,
        default=0.0,
        help="Add this value after count scaling.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help="Run detection every N frames and reuse the last count between frames.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print count to terminal every N frames. Use 0 to disable.",
    )
    return parser.parse_args()


def parse_scales(scale_text: str):
    scales = []
    for part in scale_text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        scale = float(stripped)
        if scale <= 0:
            raise ValueError("Inference scales must be greater than 0.")
        scales.append(scale)
    return scales or [1.0]


def validate_args(args):
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")
    if args.imgsz <= 0 or args.tile_size <= 0:
        raise ValueError("--imgsz and --tile-size must be greater than 0.")
    if not 0.0 <= args.tile_overlap < 1.0:
        raise ValueError("--tile-overlap must be at least 0 and less than 1.")
    if args.tile_batch_size < 0 or args.cpu_threads < 0:
        raise ValueError("--tile-batch-size and --cpu-threads cannot be negative.")
    if args.min_aspect <= 0 or args.max_aspect <= 0 or args.min_aspect > args.max_aspect:
        raise ValueError("--min-aspect and --max-aspect must be positive and ordered correctly.")


def choose_video_source(args):
    if args.source is not None:
        video_source = 0 if args.source == 0 else args.video
        source_label = "device camera 0" if args.source == 0 else args.video
        return video_source, source_label

    while True:
        choice = input("Enter source (0 = camera, 1 = recorded video): ").strip()
        if choice == "0":
            return 0, "device camera 0"
        if choice == "1":
            video_path = input("Enter recorded video path: ").strip().strip('"')
            return video_path, video_path
        print("Invalid input. Enter 0 for camera or 1 for recorded video.")


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    scales = parse_scales(args.scales)
    video_source, source_label = choose_video_source(args)
    print(f"[source] Using {source_label}")
    video_feed = VideoFeed(
        video_source,
        args.weights,
        args.conf,
        args.iou,
        args.imgsz,
        args.tiled,
        args.tile_size,
        args.tile_overlap,
        args.tile_batch_size,
        args.cpu_threads,
        args.smooth,
        args.min_box_area,
        args.max_box_area,
        args.min_box_height,
        args.max_box_width,
        args.max_box_height,
        args.min_aspect,
        args.max_aspect,
        args.center_merge,
        args.center_merge_px,
        scales,
        args.tta_flip,
        args.count_scale,
        args.count_offset,
        args.process_every,
    )
    frame_number = 0

    while True:
        frame, crowd_count = video_feed.get_frame()
        if frame is None:
            break
        frame_number += 1
        if args.print_every and frame_number % args.print_every == 0:
            print(
                f"[frame {frame_number}] Crowd Count: {crowd_count} | "
                f"{video_feed.performance_summary()}"
            )
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_feed.release()
    cv2.destroyAllWindows()
