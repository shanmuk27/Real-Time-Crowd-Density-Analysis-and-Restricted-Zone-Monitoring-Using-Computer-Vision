"""Live face detection and in-memory enrolled/unknown verification.

The trained YOLO model detects faces. ArcFace embeddings compare every detected
face with enrollment photos while this program is running. No camera frames,
face crops, CSV files, or enrollment embeddings are written by this engine.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, Lock, Thread
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from insightface.app import FaceAnalysis
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Default deployment model: paste the trained face_yolo6n best.pt at this path.
# Keeping the trained run under runs makes the normal engine command work
# without supplying --weights each time.
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "face_detection" / "face_yolo6n" / "weights" / "best.pt"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify faces from a live camera or video against enrolled photos.")
    parser.add_argument("--enrolled-faces", type=Path, help="Enrollment image or folder. If omitted, the program asks at runtime.")
    parser.add_argument("--source", help="0 for the default camera or a recorded-video path. If omitted, the program asks at runtime.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Trained YOLO face-detector best.pt.")
    parser.add_argument("--imgsz", type=int, default=0, help="0 = automatic (1024 with CUDA, responsive 512 CPU mode).")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum face-detector confidence, from 0 to 1.")
    parser.add_argument("--iou", type=float, default=0.45, help="Duplicate-box IoU threshold, from 0 to 1.")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum faces retained per frame.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Large-frame tile size for small-face accuracy.")
    parser.add_argument("--tile-overlap", type=float, default=0.25, help="Overlap for large-frame tiles, from 0 to less than 1.")
    parser.add_argument("--tiles", action="store_true", help="Force tiled inference, including on a CPU. This is slower.")
    parser.add_argument("--no-tiles", action="store_true", help="Disable tiled large-frame inference.")
    parser.add_argument("--tile-batch", type=int, default=-1, help="Tiles run together. -1 selects a safe value automatically.")
    parser.add_argument("--device", default=None, help="YOLO device: 0 for first CUDA GPU, or cpu. Defaults automatically.")
    parser.add_argument("--match-threshold", type=float, default=0.45, help="ArcFace cosine-match threshold. Higher is stricter; default favours safe UNKNOWN labels.")
    parser.add_argument("--face-margin", type=float, default=0.20, help="Extra crop around each detected face, as a fraction of face size.")
    parser.add_argument(
        "--max-verify-faces", type=int, default=-1,
        help="Maximum faces sent to ArcFace per processed frame. -1 selects 4 on CPU / 12 with CUDA; 0 disables the limit.",
    )
    parser.add_argument(
        "--verify-cache-seconds", type=float, default=5.0,
        help="Reuse a recent identity for a matching face for this long. 0 disables the cache.",
    )
    parser.add_argument(
        "--verify-cache-iou", type=float, default=0.45,
        help="Minimum overlap needed to reuse a recent identity, from greater than 0 to 1.",
    )
    parser.add_argument(
        "--recognition-det-size", type=int, default=0,
        help="ArcFace crop-detector size. 0 = automatic (160 CPU, 320 CUDA); lower is faster but less tolerant of difficult crops.",
    )
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested camera width; ignored for a recorded video.")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested camera height; ignored for a recorded video.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.weights.expanduser().is_file():
        raise FileNotFoundError(f"Face-detector weights were not found: {args.weights}")
    if args.imgsz < 0 or args.tile_size <= 0 or args.max_det <= 0:
        raise ValueError("--imgsz must be 0 or greater; --tile-size and --max-det must be greater than zero.")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("--conf and --iou must be between 0 and 1.")
    if not 0.0 <= args.tile_overlap < 1.0:
        raise ValueError("--tile-overlap must be at least 0 and less than 1.")
    if args.tile_batch == 0 or args.tile_batch < -1:
        raise ValueError("--tile-batch must be -1 (automatic) or a positive integer.")
    if not 0.0 < args.match_threshold <= 1.0:
        raise ValueError("--match-threshold must be greater than 0 and at most 1.")
    if not 0.0 <= args.face_margin <= 1.0:
        raise ValueError("--face-margin must be between 0 and 1.")
    if args.max_verify_faces < -1:
        raise ValueError("--max-verify-faces must be -1 (automatic) or greater.")
    if args.verify_cache_seconds < 0:
        raise ValueError("--verify-cache-seconds must be 0 or greater.")
    if not 0.0 < args.verify_cache_iou <= 1.0:
        raise ValueError("--verify-cache-iou must be greater than 0 and at most 1.")
    if args.recognition_det_size < 0:
        raise ValueError("--recognition-det-size must be 0 or greater.")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("--camera-width and --camera-height must be greater than zero.")
    if args.tiles and args.no_tiles:
        raise ValueError("Use either --tiles or --no-tiles, not both.")


def ask_for_enrollment_source() -> Path:
    while True:
        value = input("Enter the enrolled-face image or folder location: ").strip().strip('"')
        if not value:
            print("Please enter an enrollment location.")
            continue
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
        print(f"Location was not found: {path}")


def choose_video_source(requested_source: str | None) -> tuple[int | str, str]:
    if requested_source is not None:
        clean_source = requested_source.strip().strip('"')
        if clean_source.isdigit():
            return int(clean_source), f"camera {clean_source}"
        source_path = Path(clean_source).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(f"Video source was not found: {source_path}")
        return str(source_path.resolve()), str(source_path.resolve())

    while True:
        choice = input("Enter video source (0 = camera, 1 = recorded video): ").strip()
        if choice == "0":
            return 0, "camera 0"
        if choice == "1":
            value = input("Enter recorded-video location: ").strip().strip('"')
            source_path = Path(value).expanduser()
            if source_path.is_file():
                return str(source_path.resolve()), str(source_path.resolve())
            print(f"Video was not found: {source_path}")
            continue
        print("Enter 0 for camera or 1 for recorded video.")


def collect_enrollment_images(source: Path) -> tuple[list[Path], Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported enrollment image type: {source.suffix}")
        return [source], source.parent
    if not source.is_dir():
        raise FileNotFoundError(f"Enrollment location does not exist: {source}")
    images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No enrollment images were found in: {source}")
    return images, source


def enrollment_name(image_path: Path, source_root: Path, is_single_image: bool) -> str:
    if is_single_image:
        return image_path.stem
    relative = image_path.relative_to(source_root)
    # Recommended layout: enrolled_faces\\Alice\\photo1.jpg. The first folder is the name.
    return relative.parts[0] if len(relative.parts) > 1 else image_path.stem


def read_image(path: Path) -> np.ndarray | None:
    # Supports Windows file paths that contain non-ASCII characters.
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def choose_yolo_device(requested_device: str | None) -> str | int:
    if requested_device is not None:
        return requested_device
    return 0 if torch.cuda.is_available() else "cpu"


def configure_live_runtime(args: argparse.Namespace) -> None:
    """Select responsive CPU settings while retaining the high-accuracy GPU path."""
    args.resolved_device = choose_yolo_device(args.device)
    args.cpu_mode = str(args.resolved_device).lower() == "cpu"
    if args.imgsz == 0:
        args.imgsz = 512 if args.cpu_mode else 1024
    if args.max_verify_faces == -1:
        args.max_verify_faces = 4 if args.cpu_mode else 12
    if args.recognition_det_size == 0:
        args.recognition_det_size = 160 if args.cpu_mode else 320
    if args.cpu_mode and not args.tiles:
        args.no_tiles = True
    if args.cpu_mode:
        logical_cores = max(1, os.cpu_count() or 1)
        torch.set_num_threads(max(1, min(logical_cores - 1, 12)))
        try:
            cv2.setNumThreads(max(1, min(logical_cores - 1, 12)))
        except cv2.error:
            pass
        print(
            "[runtime] CPU live mode: 512px, no tiles, 160px ArcFace crops, and synchronized result frames. "
            "Only four new faces are verified per result."
        )
        print("[runtime] For maximum 1024px/tiled accuracy, run on the NVIDIA CUDA training PC.")


def choose_tile_batch(requested_batch: int, device: str | int) -> int:
    if requested_batch > 0:
        return requested_batch
    if str(device).lower() == "cpu":
        return 1
    try:
        memory_gib = torch.cuda.get_device_properties(int(device)).total_memory / (1024**3)
    except (AssertionError, RuntimeError, ValueError):
        return 1
    if memory_gib < 8:
        return 1
    if memory_gib < 12:
        return 2
    if memory_gib < 20:
        return 4
    return 8


def tile_positions(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, round(tile_size * (1.0 - overlap)))
    positions = list(range(0, length - tile_size + 1, step))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def make_tiles(image: np.ndarray, tile_size: int, overlap: float) -> list[tuple[np.ndarray, int, int]]:
    height, width = image.shape[:2]
    return [
        (image[y : min(y + tile_size, height), x : min(x + tile_size, width)], x, y)
        for y in tile_positions(height, tile_size, overlap)
        for x in tile_positions(width, tile_size, overlap)
    ]


def result_arrays(result) -> tuple[np.ndarray, np.ndarray]:
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()


def remove_duplicates(boxes: list[list[float]], scores: list[float], conf: float, iou: float, max_det: int) -> tuple[list[list[float]], list[float]]:
    if not boxes:
        return [], []
    nms_boxes = [[int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))] for x1, y1, x2, y2 in boxes]
    selected = cv2.dnn.NMSBoxes(nms_boxes, scores, conf, iou)
    if len(selected) == 0:
        return [], []
    indexes = np.asarray(selected).reshape(-1).tolist()
    indexes.sort(key=lambda index: scores[index], reverse=True)
    indexes = indexes[:max_det]
    return [boxes[index] for index in indexes], [scores[index] for index in indexes]


class FaceDetector:
    def __init__(self, weights: Path, args: argparse.Namespace):
        self.args = args
        self.device = args.resolved_device
        self.tile_batch = choose_tile_batch(args.tile_batch, self.device)
        self.model = YOLO(str(weights))
        print(f"[detector] Loaded trained weights: {weights}")
        print(f"[detector] device={self.device}, imgsz={args.imgsz}, tile_batch={self.tile_batch}, tiles={'off' if args.no_tiles else 'on'}")

    def detect(self, image: np.ndarray) -> tuple[list[list[float]], list[float]]:
        height, width = image.shape[:2]
        use_tiles = not self.args.no_tiles and (height > self.args.tile_size or width > self.args.tile_size)
        tiles = make_tiles(image, self.args.tile_size, self.args.tile_overlap) if use_tiles else [(image, 0, 0)]
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        for start in range(0, len(tiles), self.tile_batch):
            batch = tiles[start : start + self.tile_batch]
            results = self.model.predict(
                [tile for tile, _, _ in batch], imgsz=self.args.imgsz, conf=self.args.conf,
                iou=self.args.iou, max_det=self.args.max_det, device=self.device, verbose=False,
            )
            for (_, offset_x, offset_y), result in zip(batch, results):
                boxes, scores = result_arrays(result)
                for (x1, y1, x2, y2), score in zip(boxes, scores):
                    all_boxes.append([float(x1 + offset_x), float(y1 + offset_y), float(x2 + offset_x), float(y2 + offset_y)])
                    all_scores.append(float(score))
        return remove_duplicates(all_boxes, all_scores, self.args.conf, self.args.iou, self.args.max_det)


class FaceVerifier:
    def __init__(
        self, match_threshold: float, face_margin: float, recognition_det_size: int,
        model_pack: str = "buffalo_l",
    ):
        providers = ort.get_available_providers()
        self.uses_gpu = "CUDAExecutionProvider" in providers
        selected_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.uses_gpu else ["CPUExecutionProvider"]
        # Only detection and recognition are required. Avoid loading landmark,
        # pose, age, and gender models into the live pipeline.
        self.analyzer = FaceAnalysis(
            name=model_pack, allowed_modules=["detection", "recognition"], providers=selected_providers
        )
        self.analyzer.prepare(
            ctx_id=0 if self.uses_gpu else -1,
            det_size=(recognition_det_size, recognition_det_size),
        )
        self.match_threshold = match_threshold
        self.face_margin = face_margin
        self.model_pack = model_pack
        self.prototypes: dict[str, np.ndarray] = {}
        print(
            f"[recognition] model={model_pack}, provider={'CUDA' if self.uses_gpu else 'CPU'}, "
            f"crop detector={recognition_det_size}px, match threshold={match_threshold:.2f}"
        )

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    @staticmethod
    def largest_face(faces):
        if not faces:
            return None
        return max(faces, key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))

    def embedding_from_image(self, image: np.ndarray) -> np.ndarray | None:
        face = self.largest_face(self.analyzer.get(image))
        if face is None or getattr(face, "normed_embedding", None) is None:
            return None
        return self.normalize(face.normed_embedding)

    def load_enrollments(self, source: Path) -> None:
        images, source_root = collect_enrollment_images(source)
        embeddings_by_name: dict[str, list[np.ndarray]] = defaultdict(list)
        is_single_image = source.is_file()
        for image_path in images:
            image = read_image(image_path)
            if image is None:
                print(f"[enrollment] Skipped unreadable image: {image_path}")
                continue
            embedding = self.embedding_from_image(image)
            if embedding is None:
                print(f"[enrollment] No usable face found: {image_path}")
                continue
            name = enrollment_name(image_path, source_root, is_single_image)
            embeddings_by_name[name].append(embedding)

        for name, embeddings in embeddings_by_name.items():
            self.prototypes[name] = self.normalize(np.mean(np.stack(embeddings), axis=0))
            print(f"[enrollment] Loaded {len(embeddings)} face image(s) for {name}")
        if not self.prototypes:
            raise RuntimeError("No usable enrolled faces were loaded. Use clear photos where one face is visible.")
        print(f"[enrollment] Ready with {len(self.prototypes)} enrolled identity/identities. Nothing was saved to disk.")

    def crop_face(self, frame: np.ndarray, box: list[float]) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        margin_x = (x2 - x1) * self.face_margin
        margin_y = (y2 - y1) * self.face_margin
        left, top = max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))
        right, bottom = min(width, int(x2 + margin_x)), min(height, int(y2 + margin_y))
        return frame[top:bottom, left:right]

    def verify(self, frame: np.ndarray, box: list[float]) -> tuple[str, float, bool]:
        embedding = self.embedding_from_image(self.crop_face(frame, box))
        if embedding is None:
            return "UNKNOWN", 0.0, False
        best_name, best_score = max(
            ((name, float(np.dot(embedding, prototype))) for name, prototype in self.prototypes.items()),
            key=lambda candidate: candidate[1],
        )
        return (best_name, best_score, True) if best_score >= self.match_threshold else ("UNKNOWN", best_score, False)


@dataclass
class CachedFaceLabel:
    """A recently verified label that can follow the same detected face briefly."""

    box: tuple[float, float, float, float]
    label: tuple[str, float, bool]
    expires_at: float


def box_iou(
    first: list[float] | tuple[float, float, float, float],
    second: list[float] | tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-12)


class FaceLabelCache:
    """Avoids re-running ArcFace for the same face on every detector result."""

    def __init__(self, cache_seconds: float, match_iou: float, max_verifications: int):
        self.cache_seconds = cache_seconds
        self.match_iou = match_iou
        self.max_verifications = max_verifications
        self.entries: list[CachedFaceLabel] = []

    def label(
        self, frame: np.ndarray, boxes: list[list[float]], verifier: FaceVerifier
    ) -> tuple[list[tuple[str, float, bool]], int, int]:
        now = time.monotonic()
        available = [entry for entry in self.entries if entry.expires_at > now]
        used_entries: set[int] = set()
        next_entries: list[CachedFaceLabel] = []
        labels: list[tuple[str, float, bool]] = []
        verified_faces = 0
        cached_faces = 0

        for box in boxes:
            best_index: int | None = None
            best_iou = self.match_iou
            for index, entry in enumerate(available):
                if index in used_entries:
                    continue
                overlap = box_iou(box, entry.box)
                if overlap >= best_iou:
                    best_index = index
                    best_iou = overlap

            box_tuple = tuple(float(value) for value in box)
            if best_index is not None:
                entry = available[best_index]
                used_entries.add(best_index)
                labels.append(entry.label)
                next_entries.append(CachedFaceLabel(box_tuple, entry.label, entry.expires_at))
                cached_faces += 1
                continue

            if self.max_verifications and verified_faces >= self.max_verifications:
                # Preserve the detection box, but never let a dense scene turn
                # into hundreds of serial ArcFace passes.
                labels.append(("UNKNOWN", 0.0, False))
                continue

            label = verifier.verify(frame, box)
            labels.append(label)
            verified_faces += 1
            if self.cache_seconds:
                next_entries.append(CachedFaceLabel(box_tuple, label, now + self.cache_seconds))

        # Keep only faces present in the newest processed frame. This bounds
        # memory and limits the chance that an old label reaches a new person.
        self.entries = next_entries
        return labels, verified_faces, cached_faces


@dataclass(frozen=True)
class InferenceResult:
    """One frame and the results computed specifically for that frame."""

    sequence: int
    frame: np.ndarray
    boxes: list[list[float]]
    labels: list[tuple[str, float, bool]]
    fps: float
    elapsed_ms: float
    verified_faces: int
    cached_faces: int


class LiveInferenceWorker:
    """Processes only the newest available frame without creating a backlog.

    A completed result keeps its source frame. The UI must display that source
    frame with its boxes; drawing results on a newer frame makes boxes appear
    to trail moving faces even when capture itself is running smoothly.
    """

    def __init__(
        self,
        detector: FaceDetector,
        verifier: FaceVerifier,
        max_verify_faces: int,
        verify_cache_seconds: float,
        verify_cache_iou: float,
    ):
        self.detector = detector
        self.verifier = verifier
        self.label_cache = FaceLabelCache(verify_cache_seconds, verify_cache_iou, max_verify_faces)
        self.frames: Queue[tuple[int, np.ndarray]] = Queue(maxsize=1)
        self.stop_event = Event()
        self.result_lock = Lock()
        self.result: InferenceResult | None = None
        self.failure: str | None = None
        self.next_sequence = 0
        self.worker = Thread(target=self._run, name="face-inference", daemon=True)

    def start(self) -> None:
        self.worker.start()

    def submit(self, frame: np.ndarray) -> None:
        """Publish a frame without copying it; newer frames replace queued work."""
        self.next_sequence += 1
        item = (self.next_sequence, frame)
        while True:
            try:
                self.frames.put_nowait(item)
                return
            except Full:
                try:
                    self.frames.get_nowait()
                except Empty:
                    # The inference thread picked up the old frame. Retry so
                    # this newest frame is not accidentally discarded.
                    continue

    def snapshot(self) -> InferenceResult | None:
        """Return the immutable result reference; callers must not mutate its frame."""
        with self.result_lock:
            return self.result

    def error(self) -> str | None:
        with self.result_lock:
            return self.failure

    def _take_latest(self) -> tuple[int, np.ndarray]:
        sequence, frame = self.frames.get(timeout=0.1)
        while True:
            try:
                sequence, frame = self.frames.get_nowait()
            except Empty:
                return sequence, frame

    def _run(self) -> None:
        moving_seconds: float | None = None
        while not self.stop_event.is_set():
            try:
                sequence, frame = self._take_latest()
            except Empty:
                continue
            if self.stop_event.is_set():
                break
            try:
                started_at = time.perf_counter()
                boxes, _ = self.detector.detect(frame)
                labels, verified_faces, cached_faces = self.label_cache.label(frame, boxes, self.verifier)
                elapsed = time.perf_counter() - started_at
            except Exception as error:
                with self.result_lock:
                    self.failure = f"{type(error).__name__}: {error}"
                self.stop_event.set()
                return

            moving_seconds = elapsed if moving_seconds is None else (moving_seconds * 0.8) + (elapsed * 0.2)
            with self.result_lock:
                self.result = InferenceResult(
                    sequence=sequence,
                    frame=frame,
                    boxes=boxes,
                    labels=labels,
                    fps=1.0 / max(moving_seconds, 1e-6),
                    elapsed_ms=elapsed * 1000.0,
                    verified_faces=verified_faces,
                    cached_faces=cached_faces,
                )

    def close(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=2.0)


def draw_frame(
    frame: np.ndarray,
    boxes: list[list[float]],
    labels: list[tuple[str, float, bool]],
    fps: float,
    elapsed_ms: float = 0.0,
    verified_faces: int = 0,
    cached_faces: int = 0,
) -> np.ndarray:
    output = frame.copy()
    known_count = 0
    for box, (name, match_score, enrolled) in zip(boxes, labels):
        x1, y1, x2, y2 = (round(value) for value in box)
        color = (40, 210, 40) if enrolled else (30, 45, 235)
        if enrolled:
            known_count += 1
            label = f"ENROLLED: {name} ({match_score:.0%})"
        elif match_score:
            label = f"UNKNOWN ({match_score:.0%})"
        else:
            label = "UNKNOWN"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, max(23, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    header = (
        f"Faces: {len(boxes)} | Enrolled: {known_count} | {fps:.1f} inference FPS ({elapsed_ms:.0f} ms) | "
        f"ArcFace: {verified_faces} new, {cached_faces} cached | q = quit"
    )
    cv2.rectangle(output, (0, 0), (min(output.shape[1], max(450, len(header) * 12)), 38), (20, 20, 20), -1)
    cv2.putText(output, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def draw_waiting_frame(frame: np.ndarray) -> np.ndarray:
    """Show a live preview only until the first synchronized result is ready."""
    output = frame.copy()
    header = "Analyzing newest frame... | q = quit"
    cv2.rectangle(output, (0, 0), (min(output.shape[1], max(360, len(header) * 12)), 38), (20, 20, 20), -1)
    cv2.putText(output, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def open_capture(source: int | str, camera_width: int, camera_height: int) -> cv2.VideoCapture:
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture = cv2.VideoCapture(source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture
    return cv2.VideoCapture(source)


def ui_wait_ms(capture: cv2.VideoCapture, source: int | str) -> int:
    """Pace a file source so it cannot be consumed faster than its video rate."""
    if isinstance(source, int):
        return 1
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if 1.0 <= fps <= 120.0:
        return max(1, round(1000.0 / fps))
    return 1


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    configure_live_runtime(args)
    enrollment_source = args.enrolled_faces.expanduser().resolve() if args.enrolled_faces else ask_for_enrollment_source()
    video_source, source_label = choose_video_source(args.source)
    verifier = FaceVerifier(args.match_threshold, args.face_margin, args.recognition_det_size)
    verifier.load_enrollments(enrollment_source)
    detector = FaceDetector(args.weights.expanduser().resolve(), args)
    capture = open_capture(video_source, args.camera_width, args.camera_height)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {source_label}")

    print(f"[source] Live verification started from {source_label}. Press q in the video window to stop.")
    print("[runtime] Showing each completed frame with its own boxes; stale frames are dropped before inference.")
    wait_ms = ui_wait_ms(capture, video_source)
    worker = LiveInferenceWorker(
        detector,
        verifier,
        args.max_verify_faces,
        args.verify_cache_seconds,
        args.verify_cache_iou,
    )
    worker.start()
    displayed_sequence = -1
    displayed_frame: np.ndarray | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            worker.submit(frame)
            if failure := worker.error():
                raise RuntimeError(f"Live inference stopped: {failure}")
            result = worker.snapshot()
            if result is not None and result.sequence != displayed_sequence:
                # Never draw old coordinates on the current camera frame. The
                # result's input frame is the only frame those coordinates fit.
                displayed_frame = draw_frame(
                    result.frame,
                    result.boxes,
                    result.labels,
                    result.fps,
                    result.elapsed_ms,
                    result.verified_faces,
                    result.cached_faces,
                )
                displayed_sequence = result.sequence
            elif displayed_frame is None:
                displayed_frame = draw_waiting_frame(frame)
            cv2.imshow("Live Face Verification", displayed_frame)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
    finally:
        worker.close()
        capture.release()
        cv2.destroyAllWindows()
    print("[complete] Live face verification stopped. No camera frames, results, or face data were saved.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Face verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
