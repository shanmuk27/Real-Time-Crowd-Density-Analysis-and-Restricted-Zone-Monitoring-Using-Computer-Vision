import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_VIDEO_PATH = Path(
    r"C:\infosys_crowdcount_october2025_Shanmuk\uploads\6387-191695740.mp4"
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
        smooth: int,
    ):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.count_history = deque(maxlen=max(1, smooth))
        print(f"[model] Loaded weights: {model_path}")
        print(
            f"[mode] {'Tiled accuracy mode' if tiled else 'Single-frame speed mode'} "
            f"conf={conf}, iou={iou}, imgsz={imgsz}"
        )

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, 0

        if self.tiled:
            boxes, scores = self.detect_tiled(frame)
            crowd_count = len(boxes)
            annotated_frame = frame.copy()
            self.draw_boxes(annotated_frame, boxes, scores)
        else:
            results = self.model(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
            result = results[0]
            crowd_count = 0 if result.boxes is None else len(result.boxes)
            annotated_frame = result.plot()

        self.count_history.append(crowd_count)
        smoothed_count = round(sum(self.count_history) / len(self.count_history))
        self.draw_count_overlay(annotated_frame, crowd_count, smoothed_count)
        return annotated_frame, crowd_count

    def detect_tiled(self, frame):
        height, width = frame.shape[:2]
        step = max(1, int(self.tile_size * (1.0 - self.tile_overlap)))
        all_boxes = []
        all_scores = []

        y_positions = self.tile_positions(height, self.tile_size, step)
        x_positions = self.tile_positions(width, self.tile_size, step)

        for y1 in y_positions:
            for x1 in x_positions:
                x2 = min(x1 + self.tile_size, width)
                y2 = min(y1 + self.tile_size, height)
                tile = frame[y1:y2, x1:x2]
                results = self.model(tile, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
                result = results[0]
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for box, score in zip(xyxy, confs):
                    bx1, by1, bx2, by2 = box
                    all_boxes.append([float(bx1 + x1), float(by1 + y1), float(bx2 + x1), float(by2 + y1)])
                    all_scores.append(float(score))

        return self.merge_boxes(all_boxes, all_scores)

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
        return [boxes[index] for index in keep_indexes], [scores[index] for index in keep_indexes]

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
    def draw_count_overlay(frame, crowd_count: int, smoothed_count: int):
        label = f"Crowd Count: {smoothed_count}"
        if smoothed_count != crowd_count:
            label = f"{label}  raw: {crowd_count}"
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
        "--weights",
        default=str(TRAINED_WEIGHTS_PATH if TRAINED_WEIGHTS_PATH.exists() else "yolov8n.pt"),
        help="Path to YOLO weights. Defaults to trained weights if available.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Detection confidence threshold. Lower values count more detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.55,
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
        action="store_true",
        help="Use overlapping tiled inference. Slower, but usually better for small dense heads.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=768,
        help="Tile size for tiled inference.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="Tile overlap ratio for tiled inference.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Average the displayed count over this many frames.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print count to terminal every N frames. Use 0 to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_feed = VideoFeed(
        args.video,
        args.weights,
        args.conf,
        args.iou,
        args.imgsz,
        args.tiled,
        args.tile_size,
        args.tile_overlap,
        args.smooth,
    )
    frame_number = 0

    while True:
        frame, crowd_count = video_feed.get_frame()
        if frame is None:
            break
        frame_number += 1
        if args.print_every and frame_number % args.print_every == 0:
            print(f"[frame {frame_number}] Crowd Count: {crowd_count}")
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_feed.release()
    cv2.destroyAllWindows()
