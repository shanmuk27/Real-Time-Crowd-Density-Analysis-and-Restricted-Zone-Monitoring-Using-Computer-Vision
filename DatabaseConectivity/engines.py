"""Lazy, in-process bridge between the dashboard and the vision engines.

The standalone engine scripts remain useful for local experimentation.  This
module adapts their model code for Django: a submitted camera JPEG is analysed
and the annotated JPEG is returned to the live-camera relay.  Models are loaded
only on the first frame so starting the web application stays fast.
"""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from threading import Event, Lock, Thread
import time

import cv2
import numpy as np
from django.conf import settings


class VisionEngines:
    def __init__(self) -> None:
        self._lock = Lock()
        self._crowd_model = None
        self._crowd_class_ids: list[int] = []
        self._face_detector = None
        self._face_verifier = None
        self._face_label_caches = {}
        self._last_face_analysis_at = {}
        self._workers = {}
        self._workers_lock = Lock()
        self._states = {
            "crowd": {"enabled": True, "ready": False, "message": "Waiting for first frame."},
            "face": {"enabled": True, "ready": False, "message": "Waiting for first frame."},
        }

    def status(self) -> dict:
        return {name: dict(state) for name, state in self._states.items()}

    @staticmethod
    def clean_video_noise(frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply mild denoising only when compression/low-light noise is present.

        A bilateral filter removes small chroma and sensor artifacts while
        preserving the face and person edges used by the two detectors. Heavy
        denoisers are deliberately avoided: they can erase small faces and
        reduce crowd-detection recall.
        """
        if not getattr(settings, 'VISION_DENOISE_ENABLED', True) or frame.size == 0:
            return frame, False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        baseline = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = cv2.absdiff(gray, baseline)
        noise_score = float(np.median(residual))
        if noise_score < float(getattr(settings, 'VISION_DENOISE_MIN_RESIDUAL', 4.0)):
            return frame, False
        cleaned = cv2.bilateralFilter(
            frame,
            int(getattr(settings, 'VISION_DENOISE_DIAMETER', 5)),
            float(getattr(settings, 'VISION_DENOISE_COLOR_SIGMA', 25)),
            float(getattr(settings, 'VISION_DENOISE_SPACE_SIGMA', 25)),
        )
        return cleaned, True

    @staticmethod
    def enhance_frame_clarity(frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Recover usable contrast in genuinely dark or washed-out frames.

        This never invents missing scene content. It adjusts only luminance,
        and only when a small sampled histogram shows insufficient brightness
        or contrast. The check is cheap and avoids extra work on normal video.
        """
        if not getattr(settings, 'VISION_CLARITY_ENABLED', True) or frame.size == 0:
            return frame, False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sample = gray[::8, ::8]
        average = float(sample.mean())
        low, high = np.percentile(sample, (5, 95))
        if average >= float(getattr(settings, 'VISION_CLARITY_DARK_MEAN', 65)) and \
                (high - low) >= float(getattr(settings, 'VISION_CLARITY_MIN_RANGE', 45)):
            return frame, False
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(getattr(settings, 'VISION_CLARITY_CLAHE_CLIP', 1.5)),
            tileGridSize=(8, 8),
        )
        enhanced_luminance = clahe.apply(luminance)
        if average < float(getattr(settings, 'VISION_SHADOW_LIFT_MEAN', 65)):
            # Gamma below 1 lifts shadows strongly while changing highlights
            # only slightly. It is applied to luminance alone so colours and
            # skin tones remain stable for face verification.
            gamma = float(getattr(settings, 'VISION_SHADOW_LIFT_GAMMA', 0.72))
            lookup = np.array([
                np.clip(((value / 255.0) ** gamma) * 255.0, 0, 255)
                for value in range(256)
            ], dtype=np.uint8)
            enhanced_luminance = cv2.LUT(enhanced_luminance, lookup)
        return cv2.cvtColor(cv2.merge((enhanced_luminance, channel_a, channel_b)), cv2.COLOR_LAB2BGR), True

    def submit(
        self, camera_id: str, image_bytes: bytes, analysis_mode: str,
        enrollments: list[dict], on_result,
    ) -> None:
        """Queue only the newest frame for a camera, like the desktop engine."""
        with self._workers_lock:
            worker = self._workers.get(camera_id)
            if worker is None:
                worker = _LatestFrameWorker(self, camera_id)
                self._workers[camera_id] = worker
            worker.submit(image_bytes, analysis_mode, enrollments, on_result)

    @staticmethod
    def _device() -> str:
        requested = str(getattr(settings, "VISION_ENGINE_DEVICE", "auto"))
        if requested.lower() != "auto":
            return requested
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_crowd(self) -> None:
        if self._crowd_model is not None or not self._states["crowd"]["enabled"]:
            return
        try:
            from ultralytics import YOLO

            weights = Path(settings.CROWD_ENGINE_WEIGHTS)
            if not weights.is_file():
                raise FileNotFoundError(f"Crowd weights not found: {weights}")
            model = YOLO(str(weights))
            classes = [
                int(index) for index, name in model.names.items()
                if str(name).strip().lower() in {"person", "head"}
            ]
            if not classes:
                raise RuntimeError("Crowd weights must include a person or head class.")
            self._crowd_model = model
            self._crowd_class_ids = classes
            self._states["crowd"].update(ready=True, message=f"Loaded {weights.name}.")
        except Exception as error:
            self._states["crowd"].update(enabled=False, message=f"Unavailable: {error}")

    def _load_face(self) -> None:
        if self._face_detector is not None or not self._states["face"]["enabled"]:
            return
        try:
            # Reuse the deployed face engine's detector, including its tiling
            # and duplicate-removal behaviour. Face identity verification is
            # deliberately not activated here: it needs an explicit enrolled
            # face directory and should never be enabled implicitly.
            from CroudCountEngine.face_detection.face_detection_engine import FaceDetector, FaceLabelCache, FaceVerifier

            weights = Path(settings.FACE_ENGINE_WEIGHTS)
            if not weights.is_file():
                raise FileNotFoundError(f"Face weights not found: {weights}")
            args = Namespace(
                resolved_device=self._device(), tile_batch=1, imgsz=settings.FACE_AUTH_IMAGE_SIZE,
                no_tiles=True, tile_size=1024, tile_overlap=0.25,
                conf=0.25, iou=0.45, max_det=1000,
            )
            self._face_detector = FaceDetector(weights, args)
            self._face_verifier = FaceVerifier(
                0.45, 0.20, 160, settings.FACE_RECOGNITION_MODEL,
            )
            self._face_label_cache_class = FaceLabelCache
            self._states["face"].update(
                ready=True,
                message=f"Loaded {weights.name} with {settings.FACE_RECOGNITION_MODEL}; ready for enrolled-face verification.",
            )
        except Exception as error:
            self._states["face"].update(enabled=False, message=f"Unavailable: {error}")

    def enroll_face(self, image_bytes: bytes) -> list[float]:
        """Create a normalized ArcFace embedding without retaining the photo."""
        with self._lock:
            self._load_face()
            if self._face_verifier is None:
                raise RuntimeError(self._states["face"]["message"])
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("The enrollment image could not be read.")
            embedding = self._face_verifier.embedding_from_image(image)
            if embedding is None:
                raise ValueError("No clear face was found in this enrollment image.")
            return [float(value) for value in embedding]

    def extract_face_candidates(self, image_bytes: bytes) -> list[tuple[list[float], bytes]]:
        """Detect usable faces and return embeddings plus small display crops."""
        with self._lock:
            self._load_face()
            if self._face_detector is None or self._face_verifier is None:
                raise RuntimeError(self._states["face"]["message"])
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("The selected image could not be read.")
            boxes, _ = self._face_detector.detect(image)
            candidates = []
            for box in boxes:
                crop = self._face_verifier.crop_face(image, box)
                if crop.size == 0 or min(crop.shape[:2]) < settings.FACE_AUTH_MIN_FACE_SIZE:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                # Poor enrollment photos make unreliable embeddings. Reject
                # them before they can be named and saved.
                if not 35 <= brightness <= 220 or blur_score < 45:
                    continue
                embedding = self._face_verifier.embedding_from_image(crop)
                if embedding is None or crop.size == 0:
                    continue
                thumbnail = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    candidates.append(([float(value) for value in embedding], encoded.tobytes()))
            return candidates

    def analyse_jpeg(
        self, image_bytes: bytes, analysis_mode: str, enrollments: list[dict] | None = None,
        camera_id: str | None = None,
    ) -> tuple[bytes, int | None, dict]:
        """Return an annotated JPEG and the detected crowd count.

        A non-blocking lock avoids building an inference queue when cameras
        submit frames faster than either model can process them.
        """
        if not self._lock.acquire(blocking=False):
            return image_bytes, None, self.status(), []
        try:
            raw = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                return image_bytes, None, self.status(), []
            frame, _denoised = self.clean_video_noise(frame)
            frame, _enhanced = self.enhance_frame_clarity(frame)
            if analysis_mode == "authentication" and camera_id:
                now = time.monotonic()
                last = self._last_face_analysis_at.get(camera_id, 0.0)
                if now - last < settings.FACE_AUTH_FRAME_INTERVAL_SECONDS:
                    # Keep the video smooth: recognition is deliberately
                    # sampled while the browser continues relaying live video.
                    return image_bytes, None, self.status(), []
                self._last_face_analysis_at[camera_id] = now
            if analysis_mode == "crowd":
                self._load_crowd()
            elif analysis_mode == "authentication":
                self._load_face()
            else:
                return image_bytes, None, self.status(), []
            output = frame.copy()
            count = None
            verified_count = 0
            auth_results = []

            if analysis_mode == "crowd" and self._crowd_model is not None:
                result = self._crowd_model.predict(
                    frame, imgsz=settings.CROWD_ENGINE_IMAGE_SIZE, conf=0.25, iou=0.45, max_det=2000,
                    classes=self._crowd_class_ids, device=self._device(), verbose=False,
                )[0]
                boxes = result.boxes
                count = len(boxes) if boxes is not None else 0
                if boxes is not None:
                    for xyxy, confidence in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                        x1, y1, x2, y2 = (int(value) for value in xyxy)
                        cv2.rectangle(output, (x1, y1), (x2, y2), (35, 210, 70), 2)
                        cv2.putText(output, f"person {confidence:.0%}", (x1, max(18, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 210, 70), 1, cv2.LINE_AA)

            face_count = 0
            if analysis_mode == "authentication" and self._face_detector is not None:
                face_boxes, face_scores = self._face_detector.detect(frame)
                face_count = len(face_boxes)
                grouped_embeddings = defaultdict(list)
                for enrollment in enrollments or []:
                    vector = np.asarray(enrollment.get("embedding", []), dtype=np.float32)
                    if vector.size:
                        grouped_embeddings[enrollment["name"]].append(vector)
                # A mean, normalized embedding represents every enrollment
                # photo for the identity. Previously only the final image of a
                # given name was used, which made matching unstable.
                self._face_verifier.prototypes = {
                    name: self._face_verifier.normalize(np.mean(vectors, axis=0))
                    for name, vectors in grouped_embeddings.items()
                }
                if self._face_verifier.prototypes:
                    label_cache = self._face_label_caches.setdefault(
                        camera_id or "default",
                        self._face_label_cache_class(
                            settings.FACE_AUTH_VERIFY_CACHE_SECONDS, 0.45,
                            settings.FACE_AUTH_MAX_VERIFY_FACES,
                        ),
                    )
                    labels, _, _ = label_cache.label(frame, face_boxes, self._face_verifier)
                else:
                    labels = [("UNKNOWN", 0.0, False) for _ in face_boxes]
                for box, confidence, (name, score, verified) in zip(face_boxes, face_scores, labels):
                    face_frame = None
                    face_quality = 0.0
                    if not verified:
                        crop = self._face_verifier.crop_face(frame, box)
                        if crop.size and min(crop.shape[:2]) >= settings.FACE_AUTH_MIN_FACE_SIZE:
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            brightness = float(gray.mean())
                            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                            # Retain unknown faces only when the crop is usable
                            # for a human administrator to review.
                            if 35 <= brightness <= 220 and blur_score >= 45:
                                # Keep enough detail for the dashboard's
                                # full-size incident-review panel.
                                thumbnail = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)
                                ok_crop, encoded_crop = cv2.imencode(
                                    ".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 88],
                                )
                                if ok_crop:
                                    face_frame = encoded_crop.tobytes()
                                    face_quality = blur_score
                    auth_results.append({
                        "name": name if verified else "Unknown",
                        "verified": bool(verified),
                        "face_frame": face_frame,
                        "face_quality": face_quality,
                    })
                    x1, y1, x2, y2 = (int(value) for value in box)
                    color = (35, 210, 70) if verified else (240, 160, 25)
                    label = f"ENROLLED: {name} {score:.0%}" if verified else "UNKNOWN"
                    verified_count += int(verified)
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output, label, (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            banner = (f"Crowd: {count if count is not None else 'unavailable'}"
                      if analysis_mode == "crowd" else f"Authentication: {verified_count}/{face_count} enrolled")
            cv2.rectangle(output, (0, 0), (min(output.shape[1], 390), 34), (18, 18, 18), -1)
            cv2.putText(output, banner, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 82])
            return (encoded.tobytes() if ok else image_bytes), count, self.status(), auth_results
        except Exception as error:
            # A bad frame must not take the camera relay down.  Surface the
            # failure in the status endpoint and keep showing the raw feed.
            self._states["crowd"].update(message=f"Inference error: {error}")
            return image_bytes, None, self.status(), []
        finally:
            self._lock.release()


engines = VisionEngines()


class _LatestFrameWorker:
    """A per-camera worker that drops stale browser frames before inference."""
    def __init__(self, engines: VisionEngines, camera_id: str) -> None:
        self.engines = engines
        self.camera_id = camera_id
        self._lock = Lock()
        self._event = Event()
        self._pending = None
        self._thread = Thread(target=self._run, name=f"vision-{camera_id}", daemon=True)
        self._thread.start()

    def submit(self, image_bytes, analysis_mode, enrollments, on_result) -> None:
        with self._lock:
            self._pending = (image_bytes, analysis_mode, enrollments, on_result)
        self._event.set()

    def _run(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()
            with self._lock:
                payload = self._pending
                self._pending = None
            if payload is None:
                continue
            image_bytes, analysis_mode, enrollments, on_result = payload
            output, count, status, auth_results = self.engines.analyse_jpeg(
                image_bytes, analysis_mode, enrollments, self.camera_id,
            )
            try:
                # A rate-limited authentication pass intentionally returns the
                # untouched input. Keep the last completed overlay visible
                # instead of replacing it with a raw in-between frame.
                if not (analysis_mode == "authentication" and count is None and output == image_bytes):
                    on_result(output, count, status, auth_results)
            except Exception as error:
                print(f"[web-engine] camera={self.camera_id} result publish failed: {error}")
            else:
                print(
                    f"[web-engine] camera={self.camera_id} mode={analysis_mode} "
                    f"count={count if count is not None else 'n/a'} "
                    f"faces={len(auth_results)} verified={sum(item['verified'] for item in auth_results)} "
                    f"face={status['face']['ready']} crowd={status['crowd']['ready']}"
                )
