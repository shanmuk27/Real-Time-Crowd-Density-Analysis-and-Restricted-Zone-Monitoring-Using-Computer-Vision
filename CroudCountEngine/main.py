"""Live person detection and crowd counting.

Start this file with ``python main.py``.  It opens a camera or video, sends
frames to the CrowdHuman YOLO model, draws the person boxes, and displays the
number of detected people.  The advanced options below are kept for accuracy,
but the default settings are ready to use.
"""

import argparse
import csv
from collections import deque
import json
import os
from pathlib import Path
from threading import Event, Lock, Thread
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_VIDEO_PATH = Path(
    r"C:\infosys_crowdcount_october2025_Shanmuk\uploads\istockphoto-1471311788-640_adpp_is.mp4"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROWDHUMAN_WEIGHTS_PATH = PROJECT_ROOT / "runs" / "crowd_train" / "weights" / "best.pt"
NWPU_WEIGHTS_PATH = PROJECT_ROOT / "runs" / "crowd_yolo11l_1024" / "weights" / "best.pt"


def default_weights_path():
    """Use the local CrowdHuman YOLOv8n checkpoint, with NWPU as a fallback."""
    if CROWDHUMAN_WEIGHTS_PATH.is_file():
        return CROWDHUMAN_WEIGHTS_PATH
    if NWPU_WEIGHTS_PATH.is_file():
        return NWPU_WEIGHTS_PATH
    return CROWDHUMAN_WEIGHTS_PATH


class VideoFeed:
    """Read frames in one thread and detect people in another.

    Keeping these jobs separate prevents a slow model from making the display
    freeze.  Only the newest camera/video frame is kept; old frames are safely
    dropped when detection cannot keep up.
    """

    def __init__(
        self,
        video_path,
        model_path,
        conf,
        iou,
        imgsz,
        max_det,
        tiled,
        tile_size,
        tile_overlap,
        tile_batch_size,
        cpu_threads,
        device,
        camera_width,
        camera_height,
        smooth,
        min_box_area,
        max_box_area,
        min_box_height,
        max_box_width,
        max_box_height,
        min_aspect,
        max_aspect,
        center_merge,
        center_merge_px,
        scales,
        tta_flip,
        count_scale,
        count_offset,
        process_every,
        roi_points,
    ):
        # Save all user settings first. The rest of this method prepares the
        # camera, model, and shared values used by the two background threads.
        self.video_path = video_path
        self.device, self.uses_cuda, self.cpu_threads = self.configure_runtime(cpu_threads, device)
        self.tile_batch_size = self.resolve_tile_batch_size(tile_batch_size)
        self.cap = self.open_capture(video_path, camera_width, camera_height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {video_path}")
        self.model = YOLO(model_path)
        model_class_items = list(self.model.names.items())
        model_classes = {str(name).strip().lower() for _, name in model_class_items}
        # Older checkpoints call this class "person" even though NWPU boxes are
        # heads. Accept both names so a future correctly named "head" model is
        # also deployable without changing inference behaviour.
        self.person_class_ids = [
            int(index) for index, name in model_class_items if str(name).strip().lower() in {"person", "head"}
        ]
        if not self.person_class_ids:
            raise RuntimeError(
                f"Crowd counting requires a YOLO model trained with a 'person' or 'head' class; loaded classes are {sorted(model_classes)}. "
                "Do not use the face-detection weights here."
            )
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
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
        self.roi_points = roi_points
        self.roi_contour = self.build_roi_contour(roi_points)
        self.roi_bounds = cv2.boundingRect(self.roi_contour) if self.roi_contour is not None else None
        self.frame_index = 0
        self.last_boxes = []
        self.last_scores = []
        self.last_raw_count = None
        self.last_adjusted_count = 0
        self.last_smoothed_count = 0
        self.last_count = 0
        self.reused_detection = False
        self.last_detection_ms = 0.0
        self.average_detection_ms = None
        self.capture_lock = Lock()
        self.result_lock = Lock()
        self.stop_event = Event()
        self.new_frame_event = Event()
        self.inference_event = Event()
        self.latest_frame = None
        self.latest_frame_index = 0
        self.last_returned_frame_index = 0
        self.last_inferred_frame_index = 0
        self.last_inference_group = -1
        self.result_frame_index = 0
        self.capture_finished = False
        print(f"[model] Loaded weights: {model_path}")
        print(
            f"[mode] {'Tiled accuracy mode' if tiled else 'Single-frame speed mode'} "
            f"conf={conf}, iou={iou}, imgsz={imgsz}, max_det={max_det}, "
            f"process_every={self.process_every}, scales={self.scales}, tta_flip={self.tta_flip}"
        )
        if self.roi_contour is not None:
            print(f"[roi] Filtering detections to {len(self.roi_points)} ROI points.")
        print(
            f"[performance] device={self.device}, CUDA={'on' if self.uses_cuda else 'off'}, "
            f"CPU threads={self.cpu_threads}, tile batch={self.tile_batch_size if tiled else 'not used'}"
        )
        self.capture_thread = Thread(target=self.capture_loop, name="crowd-capture", daemon=True)
        self.inference_thread = Thread(target=self.inference_loop, name="crowd-inference", daemon=True)
        self.capture_thread.start()
        self.inference_thread.start()
        print("[runtime] Latest-frame mode is active: display stays responsive and stale frames are dropped before inference.")

    @staticmethod
    def configure_runtime(requested_threads, requested_device):
        logical_cpus = os.cpu_count() or 1
        uses_cuda = torch.cuda.is_available() and (requested_device is None or str(requested_device).lower() != "cpu")
        if requested_device is not None and str(requested_device).lower() != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA-enabled PyTorch cannot see an NVIDIA GPU.")
        device = requested_device if requested_device is not None else ("0" if uses_cuda else "cpu")
        # GPU inference should leave CPU capacity for capture, display, and OS work.
        # CPU inference must leave capacity for capture, display, and the OS.
        # Users can still explicitly request more threads with --cpu-threads.
        auto_threads = max(1, min(4, logical_cpus - 1))
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
        if uses_cuda:
            torch.backends.cudnn.benchmark = True
            try:
                gpu_index = int(str(device).replace("cuda:", "").split(",")[0])
                print(f"[runtime] CUDA inference on {torch.cuda.get_device_name(gpu_index)} (device={device})")
            except (ValueError, RuntimeError):
                print(f"[runtime] CUDA inference enabled (device={device})")
        else:
            print("[runtime] CUDA was not selected; running inference on CPU. Expect lower live throughput.")
        return device, uses_cuda, thread_count

    def resolve_tile_batch_size(self, requested_batch_size):
        if requested_batch_size > 0:
            return requested_batch_size
        if not self.uses_cuda:
            return min(4, self.cpu_threads)
        try:
            gpu_index = int(str(self.device).replace("cuda:", "").split(",")[0])
            memory_gib = torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)
        except (AssertionError, RuntimeError, ValueError):
            return 1
        # YOLO11l at 1024px needs conservative GPU batches; batching does not
        # change detections but removes Python/model-launch overhead when safe.
        if memory_gib < 10:
            return 1
        if memory_gib < 16:
            return 2
        return 4

    @staticmethod
    def build_roi_contour(roi_points):
        if not roi_points:
            return None
        return np.asarray(roi_points, dtype=np.float32).reshape((-1, 1, 2))

    @staticmethod
    def open_capture(video_path, camera_width, camera_height):
        """Open a camera number or a video-file path."""
        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return cv2.VideoCapture(str(video_path))

    def capture_loop(self):
        """Capture continuously, retaining only the newest frame.

        A slow detector therefore cannot create a queue of old frames. Recorded
        video is paced at its source FPS; cameras are read as soon as frames are
        available.
        """
        is_camera = isinstance(self.video_path, int)
        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 0.0 if is_camera or not source_fps or source_fps <= 1 else 1.0 / source_fps
        next_frame_at = time.perf_counter()
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                with self.capture_lock:
                    self.capture_finished = True
                self.new_frame_event.set()
                self.inference_event.set()
                return
            with self.capture_lock:
                self.latest_frame = frame
                self.latest_frame_index += 1
            self.new_frame_event.set()
            self.inference_event.set()
            if frame_interval:
                next_frame_at += frame_interval
                remaining = next_frame_at - time.perf_counter()
                if remaining > 0:
                    self.stop_event.wait(remaining)
                else:
                    next_frame_at = time.perf_counter()

    def inference_loop(self):
        """Run one inference at a time on the most recent available frame."""
        while not self.stop_event.is_set():
            self.inference_event.wait(0.10)
            self.inference_event.clear()
            if self.stop_event.is_set():
                return
            with self.capture_lock:
                if self.latest_frame is None:
                    continue
                frame_index = self.latest_frame_index
                frame = self.latest_frame.copy()
            if frame_index <= self.last_inferred_frame_index:
                continue
            frame_group = (frame_index - 1) // self.process_every
            if frame_group == self.last_inference_group:
                continue
            self.infer_and_publish(frame, frame_index)
            self.last_inferred_frame_index = frame_index
            self.last_inference_group = frame_group

    def infer_and_publish(self, frame, frame_index):
        """Detect people in one frame and save the result for the display."""
        detection_started_at = time.perf_counter()
        if self.tiled:
            boxes, scores = self.detect_tiled(frame)
        else:
            boxes, scores = self.detect_image(frame)
            boxes, scores = self.merge_boxes(boxes, scores)
        raw_count = len(boxes)
        detection_ms = (time.perf_counter() - detection_started_at) * 1000.0
        adjusted_count = self.apply_count_correction(raw_count)
        with self.result_lock:
            self.count_history.append(adjusted_count)
            smoothed_count = round(sum(self.count_history) / len(self.count_history))
            self.last_boxes = boxes
            self.last_scores = scores
            self.last_raw_count = raw_count
            self.last_adjusted_count = adjusted_count
            self.last_smoothed_count = smoothed_count
            self.last_count = smoothed_count
            self.last_detection_ms = detection_ms
            if self.average_detection_ms is None:
                self.average_detection_ms = detection_ms
            else:
                self.average_detection_ms = (self.average_detection_ms * 0.9) + (detection_ms * 0.1)
            self.result_frame_index = frame_index

    def get_frame(self):
        """Return the next captured frame immediately, with the latest boxes."""
        while not self.stop_event.is_set():
            with self.capture_lock:
                frame = self.latest_frame
                frame_index = self.latest_frame_index
                finished = self.capture_finished
            if frame is not None and frame_index > self.last_returned_frame_index:
                self.last_returned_frame_index = frame_index
                break
            if finished:
                return None, 0
            self.new_frame_event.wait(0.03)
            self.new_frame_event.clear()
        else:
            return None, 0

        annotated_frame = frame.copy()
        self.draw_roi(annotated_frame)
        with self.result_lock:
            boxes = list(self.last_boxes)
            scores = list(self.last_scores)
            raw_count = self.last_raw_count
            adjusted_count = self.last_adjusted_count
            smoothed_count = self.last_smoothed_count
            result_frame_index = self.result_frame_index
        if raw_count is None:
            self.draw_status_overlay(annotated_frame, "Detecting crowd...")
            return annotated_frame, 0

        reused = result_frame_index != frame_index
        self.reused_detection = reused
        self.draw_boxes(annotated_frame, boxes, scores)
        self.draw_count_overlay(annotated_frame, raw_count, adjusted_count, smoothed_count, reused=reused)
        return annotated_frame, smoothed_count

    def apply_count_correction(self, raw_count):
        return max(0, round((raw_count * self.count_scale) + self.count_offset))

    def detect_image(self, image, x_offset=0.0, y_offset=0.0):
        """Run the model on one image, including optional scales and mirroring."""
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
    def resize_for_scale(image, scale):
        if abs(scale - 1.0) < 0.001:
            return image
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    def run_model_on_image(self, image):
        result = self.run_model_on_images([image])[0]
        return self.extract_result_arrays(result)

    def run_model_on_images(self, images):
        predict_args = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
            "device": self.device,
            "classes": self.person_class_ids,
        }
        if self.uses_cuda:
            predict_args["half"] = True
        return self.model.predict(images[0] if len(images) == 1 else images, **predict_args)

    @staticmethod
    def extract_result_arrays(result):
        if result.boxes is None or len(result.boxes) == 0:
            return [], []
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()

    def map_detections(self, boxes, scores, x_offset, y_offset, scale, flipped, image_width):
        """Move boxes from a scaled/tiled image back to full-frame coordinates."""
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
        return self.filter_boxes_by_roi(mapped_boxes[valid].tolist(), mapped_scores[valid].tolist())

    def filter_boxes_by_roi(self, boxes, scores):
        if self.roi_contour is None or not boxes:
            return boxes, scores

        filtered_boxes = []
        filtered_scores = []
        for box, score in zip(boxes, scores):
            cx, cy, _, _ = self.box_geometry(box)
            if cv2.pointPolygonTest(self.roi_contour, (float(cx), float(cy)), False) >= 0:
                filtered_boxes.append(box)
                filtered_scores.append(score)
        return filtered_boxes, filtered_scores

    def extract_boxes(self, result, x_offset=0.0, y_offset=0.0):
        boxes, scores = self.extract_result_arrays(result)
        return self.map_detections(boxes, scores, x_offset, y_offset, 1.0, False, 0)

    def is_valid_box(self, box):
        x1, y1, x2, y2 = box
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        aspect = width / height if height else 0.0
        if self.min_box_height and height < self.min_box_height:
            return False
        if self.max_box_width and width > self.max_box_width:
            return False
        if self.max_box_height and height > self.max_box_height:
            return False
        if self.min_aspect and aspect < self.min_aspect:
            return False
        if self.max_aspect and aspect > self.max_aspect:
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

        valid = np.ones(len(boxes), dtype=bool)
        if self.min_box_height:
            valid &= heights >= self.min_box_height
        if self.max_box_width:
            valid &= widths <= self.max_box_width
        if self.max_box_height:
            valid &= heights <= self.max_box_height
        if self.min_aspect:
            valid &= aspects >= self.min_aspect
        if self.max_aspect:
            valid &= aspects <= self.max_aspect
        if self.min_box_area:
            valid &= areas >= self.min_box_area
        if self.max_box_area:
            valid &= areas <= self.max_box_area
        return valid

    def detect_tiled(self, frame):
        """Split a large frame into overlapping pieces, then join the results."""
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
                if not self.tile_intersects_roi(x1, y1, x2, y2):
                    continue
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

    def tile_intersects_roi(self, x1, y1, x2, y2):
        """Reject tiles outside an ROI before costly model inference."""
        if self.roi_bounds is None:
            return True
        roi_x, roi_y, roi_width, roi_height = self.roi_bounds
        roi_x2, roi_y2 = roi_x + roi_width, roi_y + roi_height
        return x1 < roi_x2 and x2 > roi_x and y1 < roi_y2 and y2 > roi_y

    def detect_tile_batch(self, tiles, scale, flipped):
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
    def tile_positions(length, tile_size, step):
        if length <= tile_size:
            return [0]
        positions = list(range(0, length - tile_size + 1, step))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def merge_boxes(self, boxes, scores):
        """Remove duplicate boxes that came from tiles, scales, or a flipped image."""
        if not boxes:
            return [], []

        return self.run_nms(boxes, scores)

    def run_nms(self, boxes, scores):
        if not boxes:
            return [], []
        # NMS (non-maximum suppression) keeps the strongest box when several
        # boxes describe the same person. OpenCV runs this step efficiently.
        keep_indexes = self.nms_keep_indexes(boxes, scores)
        kept_boxes = [boxes[index] for index in keep_indexes]
        kept_scores = [scores[index] for index in keep_indexes]
        return self.merge_nearby_centers(kept_boxes, kept_scores)

    def nms_keep_indexes(self, boxes, scores):
        nms_boxes = [
            [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]
            for x1, y1, x2, y2 in boxes
        ]
        keep = cv2.dnn.NMSBoxes(nms_boxes, scores, self.conf, self.iou)
        return [] if len(keep) == 0 else np.asarray(keep).reshape(-1).tolist()

    def merge_nearby_centers(self, boxes, scores):
        if not boxes:
            return [], []
        if self.center_merge <= 0 and self.center_merge_px <= 0:
            return boxes, scores

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
    def geometry_cell(geometry, cell_size):
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
        """Draw one green rectangle and confidence score for every person."""
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
    def draw_count_overlay(frame, raw_count, adjusted_count, smoothed_count, reused=False):
        label = f"Crowd Count: {smoothed_count}"
        details = []
        if smoothed_count != adjusted_count or adjusted_count != raw_count:
            details.append(f"raw: {raw_count}")
        if reused:
            details.append("reused")
        if details:
            label = f"{label}  {'  '.join(details)}"
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

    @staticmethod
    def draw_status_overlay(frame, label):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_width, text_height = text_size
        cv2.rectangle(frame, (8, 8), (28 + text_width, 30 + text_height), (0, 0, 0), -1)
        cv2.putText(frame, label, (18, 22 + text_height), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    def draw_roi(self, frame):
        if self.roi_contour is None:
            return
        cv2.polylines(frame, [self.roi_contour.astype(np.int32)], True, (255, 180, 0), 2)

    def release(self):
        self.stop_event.set()
        self.new_frame_event.set()
        self.inference_event.set()
        if getattr(self, "capture_thread", None):
            self.capture_thread.join(timeout=1.0)
        if getattr(self, "inference_thread", None):
            self.inference_thread.join(timeout=1.0)
        self.cap.release()

    def performance_summary(self):
        if not self.average_detection_ms:
            return "detector timing unavailable"
        detector_fps = 1000.0 / self.average_detection_ms
        return f"detect={self.last_detection_ms:.0f}ms avg={self.average_detection_ms:.0f}ms ({detector_fps:.1f} FPS)"


def parse_args():
    """Read command-line options. Defaults provide normal CrowdHuman use."""
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
        default=str(default_weights_path()),
        help="Path to person-detector weights. Defaults to the local CrowdHuman YOLOv8n best checkpoint.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Detection confidence threshold. Matches the CrowdHuman YOLOv8 inference configuration.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold. Matches the CrowdHuman YOLOv8 inference configuration.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Matches the CrowdHuman YOLOv8n training and inference configuration.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum person detections per frame. Matches the CrowdHuman YOLOv8 inference configuration.",
    )
    parser.add_argument(
        "--tiled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use overlapping tiled inference. Disabled by default to match CrowdHuman YOLOv8 full-frame inference.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size for optional tiled inference. Set it to match --imgsz when enabling --tiled.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.20,
        help="Tile overlap ratio for tiled inference.",
    )
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=0,
        help="Tiles per YOLO call. 0 chooses a conservative batch from available device memory.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU threads for inference. 0 uses a responsive four-thread maximum; increase only after measuring throughput.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device: 0 for the first NVIDIA CUDA GPU or cpu. Defaults to CUDA when available.",
    )
    parser.add_argument("--camera-width", type=int, default=1920, help="Requested camera width; ignored for recorded video.")
    parser.add_argument("--camera-height", type=int, default=1080, help="Requested camera height; ignored for recorded video.")
    parser.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="Average the displayed count over this many frames.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=0.0,
        help="Ignore boxes smaller than this pixel area. 0 disables size filtering (recommended until validated).",
    )
    parser.add_argument(
        "--max-box-area",
        type=float,
        default=0.0,
        help="Ignore boxes larger than this pixel area. 0 disables the filter (recommended).",
    )
    parser.add_argument(
        "--min-box-height",
        type=float,
        default=0.0,
        help="Ignore boxes shorter than this many pixels. 0 disables the filter (recommended).",
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
        default=0.0,
        help="Ignore boxes narrower than this width/height ratio. 0 disables the filter (recommended).",
    )
    parser.add_argument(
        "--max-aspect",
        type=float,
        default=0.0,
        help="Ignore boxes wider than this width/height ratio. 0 disables the filter (recommended).",
    )
    parser.add_argument(
        "--center-merge",
        type=float,
        default=0.0,
        help="Merge boxes whose centers are close relative to box size. 0 disables this risky dense-crowd filter.",
    )
    parser.add_argument(
        "--center-merge-px",
        type=float,
        default=0.0,
        help="Minimum pixel distance used for center-based duplicate merging. 0 disables the pixel rule.",
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
        "--roi",
        default="",
        help='Optional ROI polygon "x1,y1;x2,y2;x3,y3". Detections with centers outside it are ignored.',
    )
    parser.add_argument(
        "--count-log-csv",
        default="",
        help="Optional CSV path for per-frame predicted counts and detector timing.",
    )
    parser.add_argument(
        "--dashboard-url",
        default="",
        help="Optional CrowdCount server URL, for example http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--camera-id",
        default="",
        help="Camera ID to update on the dashboard; required with --dashboard-url.",
    )
    parser.add_argument(
        "--dashboard-publish-interval",
        type=float,
        default=2.0,
        help="Seconds between dashboard updates when --dashboard-url is set.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        default="",
        help='Optional CSV with "frame,count" columns. Prints MAE/RMSE/bias after the video ends.',
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print count to terminal every N frames. Use 0 to disable.",
    )
    return parser.parse_args()


def parse_scales(scale_text):
    """Turn text such as '0.75,1.0' into a list of usable scale numbers."""
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


def parse_roi(roi_text):
    """Turn an ROI string into points that OpenCV can use as a polygon."""
    roi_text = roi_text.strip()
    if not roi_text:
        return None

    points = []
    for part in roi_text.split(";"):
        coordinates = [coordinate.strip() for coordinate in part.split(",")]
        if len(coordinates) != 2 or not coordinates[0] or not coordinates[1]:
            raise ValueError('ROI points must use "x,y;x,y;x,y" format.')
        points.append((float(coordinates[0]), float(coordinates[1])))

    if len(points) < 3:
        raise ValueError("ROI must contain at least three points.")
    return points


def validate_args(args):
    """Stop early with a clear message when an option value is invalid."""
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")
    if args.imgsz <= 0 or args.tile_size <= 0 or args.max_det <= 0:
        raise ValueError("--imgsz, --tile-size, and --max-det must be greater than 0.")
    if not 0.0 <= args.tile_overlap < 1.0:
        raise ValueError("--tile-overlap must be at least 0 and less than 1.")
    if args.tile_batch_size < 0 or args.cpu_threads < 0:
        raise ValueError("--tile-batch-size and --cpu-threads cannot be negative.")
    if args.process_every <= 0 or args.smooth <= 0:
        raise ValueError("--process-every and --smooth must be greater than 0.")
    if args.center_merge < 0 or args.center_merge_px < 0:
        raise ValueError("--center-merge and --center-merge-px cannot be negative.")
    if args.min_box_area < 0 or args.max_box_area < 0 or args.min_box_height < 0:
        raise ValueError("Box-area and box-height filters cannot be negative.")
    if args.min_aspect < 0 or args.max_aspect < 0:
        raise ValueError("--min-aspect and --max-aspect cannot be negative.")
    if args.min_aspect and args.max_aspect and args.min_aspect > args.max_aspect:
        raise ValueError("When both aspect filters are enabled, --min-aspect cannot exceed --max-aspect.")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("--camera-width and --camera-height must be greater than zero.")
    if bool(args.dashboard_url) != bool(args.camera_id):
        raise ValueError("--dashboard-url and --camera-id must be provided together.")
    if args.dashboard_publish_interval <= 0:
        raise ValueError("--dashboard-publish-interval must be greater than zero.")


def open_count_log(csv_path):
    """Create the optional CSV file used to save count results."""
    if not csv_path:
        return None, None

    log_path = Path(csv_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", newline="")
    fieldnames = [
        "frame",
        "count",
        "raw_count",
        "adjusted_count",
        "smoothed_count",
        "reused_detection",
        "last_detection_ms",
    ]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    return log_file, writer


def publish_dashboard_count(dashboard_url, camera_id, count):
    """Send one detector result to the running CrowdCount dashboard."""
    endpoint = dashboard_url.rstrip("/") + "/api/crowd-record/add/"
    payload = json.dumps({"camera_id": camera_id, "count": count}).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=2) as response:
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


def load_ground_truth_counts(csv_path):
    """Read expected counts from a CSV so final accuracy metrics can be shown."""
    if not csv_path:
        return {}

    ground_truth = {}
    with Path(csv_path).expanduser().open(newline="") as gt_file:
        reader = csv.DictReader(gt_file)
        field_lookup = {field.strip().lower(): field for field in reader.fieldnames or []}
        if "frame" not in field_lookup or "count" not in field_lookup:
            raise ValueError('Ground-truth CSV must contain "frame" and "count" columns.')

        frame_field = field_lookup["frame"]
        count_field = field_lookup["count"]
        for row in reader:
            frame_number = int(float(row[frame_field]))
            ground_truth[frame_number] = float(row[count_field])
    return ground_truth


def print_count_metrics(predictions, ground_truth):
    """Print simple count-error measurements when ground truth is available."""
    if not ground_truth:
        return

    matched = [
        (predictions[frame_number], expected_count)
        for frame_number, expected_count in ground_truth.items()
        if frame_number in predictions
    ]
    if not matched:
        print("[metrics] No overlapping frame numbers between predictions and ground truth.")
        return

    errors = [predicted - expected for predicted, expected in matched]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = (sum(error * error for error in errors) / len(errors)) ** 0.5
    bias = sum(errors) / len(errors)
    print(f"[metrics] frames={len(matched)} MAE={mae:.2f} RMSE={rmse:.2f} bias={bias:.2f}")


def choose_video_source(args):
    """Use the supplied source option or ask whether to use camera or video."""
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
    # 1. Read settings, 2. open the selected camera/video, 3. keep showing
    # frames until the video ends or the user presses q.
    args = parse_args()
    validate_args(args)
    scales = parse_scales(args.scales)
    roi_points = parse_roi(args.roi)
    ground_truth_counts = load_ground_truth_counts(args.ground_truth_csv)
    video_source, source_label = choose_video_source(args)
    print(f"[source] Using {source_label}")
    video_feed = VideoFeed(
        video_source,
        args.weights,
        args.conf,
        args.iou,
        args.imgsz,
        args.max_det,
        args.tiled,
        args.tile_size,
        args.tile_overlap,
        args.tile_batch_size,
        args.cpu_threads,
        args.device,
        args.camera_width,
        args.camera_height,
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
        roi_points,
    )
    frame_number = 0
    predictions = {}
    count_log_file, count_log_writer = open_count_log(args.count_log_csv)
    next_dashboard_publish_at = 0.0

    try:
        while True:
            frame, crowd_count = video_feed.get_frame()
            if frame is None:
                break
            frame_number += 1
            predictions[frame_number] = crowd_count
            if args.dashboard_url and time.monotonic() >= next_dashboard_publish_at:
                publish_dashboard_count(args.dashboard_url, args.camera_id, crowd_count)
                next_dashboard_publish_at = time.monotonic() + args.dashboard_publish_interval
            if count_log_writer:
                count_log_writer.writerow(
                    {
                        "frame": frame_number,
                        "count": crowd_count,
                        "raw_count": video_feed.last_raw_count,
                        "adjusted_count": video_feed.last_adjusted_count,
                        "smoothed_count": video_feed.last_smoothed_count,
                        "reused_detection": int(video_feed.reused_detection),
                        "last_detection_ms": f"{video_feed.last_detection_ms:.3f}",
                    }
                )
            if args.print_every and frame_number % args.print_every == 0:
                print(
                    f"[frame {frame_number}] Crowd Count: {crowd_count} | "
                    f"{video_feed.performance_summary()}"
                )
            cv2.imshow("Video Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if count_log_file:
            count_log_file.close()
        video_feed.release()
        cv2.destroyAllWindows()

    print_count_metrics(predictions, ground_truth_counts)
