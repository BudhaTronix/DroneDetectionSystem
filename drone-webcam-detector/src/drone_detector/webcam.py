"""Run realtime drone detection from a webcam using a pretrained YOLO model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .download_model import DEFAULT_MODEL_PATH, PROJECT_ROOT, download_pretrained_model

WINDOW_NAME = "Drone Detector"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "runs" / "drone_webcam_output.mp4"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for webcam inference."""
    default_model = DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT)
    default_output = DEFAULT_OUTPUT_PATH.relative_to(PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description="Detect drones from a webcam using a pretrained YOLO checkpoint."
    )
    parser.add_argument("--model", default=str(default_model), help="Model checkpoint path.")
    parser.add_argument("--camera", default=0, type=int, help="Webcam index to open.")
    parser.add_argument(
        "--conf",
        default=0.35,
        type=float,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        default=640,
        type=int,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device, such as auto, cpu, 0, or cuda:0.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated webcam output.",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output video path used with --save.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run inference without opening an OpenCV display window.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths from the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_model_exists(model_path: Path) -> Path:
    """Download the pretrained checkpoint if the selected model path is missing."""
    if model_path.exists():
        print(f"Loading model from {model_path}")
        return model_path

    print(f"Model not found at {model_path}. Downloading pretrained checkpoint...")
    return download_pretrained_model(model_path)


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open a webcam and fail with a helpful message if it is unavailable."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open webcam index {camera_index}. "
            "Check camera permissions or try another index, for example --camera 1."
        )
    return capture


def camera_property(
    capture: cv2.VideoCapture, property_id: int, fallback: float
) -> float:
    """Read a positive OpenCV capture property or return a fallback value."""
    value = capture.get(property_id)
    if value and value > 0:
        return value
    return fallback


def create_video_writer(
    output_path: Path, width: int, height: int, fps: float
) -> cv2.VideoWriter:
    """Create a video writer for annotated MP4 output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video writer for {output_path}.")
    print(f"Saving annotated video to {output_path}")
    return writer


def annotate_detections(
    frame: np.ndarray,
    result: object,
) -> list[tuple[str, float]]:
    """Draw YOLO detections on a frame and return labels with confidences."""
    detections: list[tuple[str, float]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    names = getattr(result, "names", {}) or {}
    for box in boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy()
        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
        confidence = float(box.conf[0].detach().cpu().item())
        class_id = int(box.cls[0].detach().cpu().item())
        label = str(names.get(class_id, f"class_{class_id}"))
        detections.append((label, confidence))

        text = f"{label} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        text_y = max(y1 - 8, 20)
        cv2.putText(
            frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    return detections


def add_fps_overlay(frame: np.ndarray, fps: float) -> None:
    """Draw the current frames-per-second estimate on the frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_detection(args: argparse.Namespace) -> None:
    """Run YOLO inference on webcam frames until the user quits."""
    model_path = ensure_model_exists(resolve_project_path(args.model))
    output_path = resolve_project_path(args.output)

    print("Loading YOLO model...")
    model = YOLO(str(model_path))
    capture = open_camera(args.camera)
    writer: cv2.VideoWriter | None = None

    try:
        width = int(camera_property(capture, cv2.CAP_PROP_FRAME_WIDTH, 640))
        height = int(camera_property(capture, cv2.CAP_PROP_FRAME_HEIGHT, 480))
        source_fps = camera_property(capture, cv2.CAP_PROP_FPS, 30.0)
        if args.save:
            writer = create_video_writer(output_path, width, height, source_fps)

        print("Starting webcam detection. Press 'q' in the display window to quit.")
        if args.no_display:
            print("Display disabled. Press Ctrl+C in the terminal to stop.")

        previous_time = time.perf_counter()
        predict_kwargs = {
            "conf": args.conf,
            "imgsz": args.imgsz,
            "verbose": False,
        }
        if args.device.lower() != "auto":
            predict_kwargs["device"] = args.device

        while True:
            ok, frame = capture.read()
            if not ok:
                print("No frame received from webcam. Stopping detection.")
                break

            results = model.predict(frame, **predict_kwargs)
            result = results[0] if results else None

            detections: list[tuple[str, float]] = []
            if result is not None:
                detections = annotate_detections(frame, result)

            now = time.perf_counter()
            elapsed = max(now - previous_time, 1e-6)
            previous_time = now
            add_fps_overlay(frame, 1.0 / elapsed)

            for label, confidence in detections:
                print(f"Detected {label}: {confidence:.2f}")

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit requested. Exiting.")
                    break

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("Released webcam and closed OpenCV windows.")


def main() -> int:
    """Command-line entrypoint."""
    args = parse_args()
    try:
        run_detection(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting cleanly.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
