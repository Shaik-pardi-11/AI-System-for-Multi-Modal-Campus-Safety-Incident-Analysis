import cv2
import time
import threading
from collections import Counter
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CAMERA_URL = "http://100.75.50.187:8080/video"
MODEL_PATH = ROOT / "yolo26n_openvino_model"

INFER_SIZE = 512
CONFIDENCE = 0.35
DISPLAY_WIDTH = 960
INFER_EVERY = 1

def open_camera(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError("Cannot open Phone 1 stream.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

class LatestFrameReader:
    def __init__(self, url):
        self.cap = open_camera(url)
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.02)

    def read(self):
        with self.lock:
            return (self.frame is not None,
                    None if self.frame is None else self.frame.copy())

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"OpenVINO model not found: {MODEL_PATH}\n"
            "Run: python src/export_yolo_openvino.py"
        )

    print("Loading YOLO26n OpenVINO model...")
    model = YOLO(str(MODEL_PATH))
    print("Target device: Intel GPU (OpenVINO)")
    print("Connecting to Phone 1...")
    camera = LatestFrameReader(CAMERA_URL)

    last_view = None
    last_counts = Counter()
    inference_ms = 0.0
    fps = 0.0
    frames = 0
    fps_timer = time.perf_counter()
    frame_id = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]
            if w > DISPLAY_WIDTH:
                frame = cv2.resize(
                    frame,
                    (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)),
                    interpolation=cv2.INTER_AREA
                )

            frame_id += 1
            frames += 1

            if frame_id % INFER_EVERY == 0 or last_view is None:
                t0 = time.perf_counter()
                result = model.predict(
                    source=frame,
                    imgsz=INFER_SIZE,
                    conf=CONFIDENCE,
                    device="intel:gpu",
                    half=True,
                    verbose=False
                )[0]
                inference_ms = (time.perf_counter() - t0) * 1000
                last_view = result.plot()

                counts = Counter()
                if result.boxes is not None:
                    for class_id in result.boxes.cls.cpu().numpy():
                        counts[model.names[int(class_id)]] += 1
                last_counts = counts

            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1:
                fps = frames / elapsed
                frames = 0
                fps_timer = time.perf_counter()

            display = last_view.copy()
            people = last_counts.get("person", 0)
            vehicles = sum(last_counts.get(k, 0)
                           for k in ("car", "motorcycle", "bus", "truck"))

            cv2.rectangle(display, (8, 8), (590, 116), (25, 25, 55), -1)
            cv2.putText(display, f"PEOPLE: {people}  VEHICLES: {vehicles}",
                        (20, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (255, 255, 255), 2)
            cv2.putText(display, f"FPS: {fps:.1f}  INFERENCE: {inference_ms:.1f} ms",
                        (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (100, 255, 180), 2)
            cv2.putText(display, "PHONE 1 | YOLO26n | OPENVINO INTEL GPU",
                        (20, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (220, 220, 255), 1)

            cv2.imshow("I-Mage | Fast Intel GPU Live Detection", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
