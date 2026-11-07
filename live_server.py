"""
I-Mage low-latency live detection server.

Run from the project root:
    python live_server.py

Then run Streamlit in a second terminal:
    python -m streamlit run app.py
"""
import os
import time
import threading
import json
import uuid
from datetime import datetime, timezone
from collections import Counter, deque
from collections import defaultdict
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "yolo26n_openvino_model"
FIGHT_CHECKPOINT = ROOT / "models" / "videomae_best.pth"
VIDEOMAE_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_VIDEO_FRAMES = 16
VIDEO_FRAME_SIZE = 224
FIGHT_SAMPLE_INTERVAL = float(os.getenv("FIGHT_SAMPLE_INTERVAL", "0.25"))
FIGHT_INFERENCE_INTERVAL = float(os.getenv("FIGHT_INFERENCE_INTERVAL", "1.5"))
FIGHT_THRESHOLD = float(os.getenv("FIGHT_THRESHOLD", "0.60"))
FIGHT_CONFIRMATIONS = int(os.getenv("FIGHT_CONFIRMATIONS", "3"))
ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))
EVIDENCE_PRE_SECONDS = float(os.getenv("EVIDENCE_PRE_SECONDS", "10"))
EVIDENCE_POST_SECONDS = float(os.getenv("EVIDENCE_POST_SECONDS", "10"))
CAMERA_ID_DEFAULT = os.getenv("CAMERA_ID", "phone1")
INCIDENT_DIR = ROOT / "incidents"
EVIDENCE_DIR = INCIDENT_DIR / "evidence"
INCIDENTS_FILE = INCIDENT_DIR / "incidents.json"
INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
_incidents_lock = threading.Lock()


def load_incidents():
    if not INCIDENTS_FILE.exists():
        return []
    try:
        data = json.loads(INCIDENTS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_incidents(items):
    INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
    temp = INCIDENTS_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(items, indent=2), encoding="utf-8")
    temp.replace(INCIDENTS_FILE)


def persist_incident(incident):
    with _incidents_lock:
        items = load_incidents()
        items.insert(0, incident)
        save_incidents(items)


def write_evidence_video(frames, output_path, fps=4.0):
    if not frames:
        return False
    height, width = frames[0].shape[:2]
    if width > 640:
        new_height = int(height * 640 / width)
        size = (640, new_height)
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
        height, width = size[1], size[0]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0

# The current Phone 1 URL supplied in the latest dashboard output.
DEFAULT_CAMERA_URL = os.getenv(
    "IMAGE_CAMERA_URL",
    "http://10.22.195.48:8080/video"
)

IMG_SIZE = int(os.getenv("IMAGE_SIZE", "512"))
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.45"))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU", "0.50"))
MAX_FRAME_WIDTH = int(os.getenv("MAX_FRAME_WIDTH", "640"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "78"))
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "0.0"))

app = FastAPI(title="I-Mage Live Detection API")

_model = None
_fight_model = None
_fight_device = None
_worker = None
_worker_lock = threading.Lock()


class StartRequest(BaseModel):
    camera_url: str
    camera_id: str = CAMERA_ID_DEFAULT


class LatestCamera:
    """Continuously drains the camera stream and retains only the newest frame."""
    def __init__(self, url: str):
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Cannot open camera stream. Check the IP Webcam server, URL and network."
            )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while not self.stopped.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1.5)
        self.cap.release()


class DetectionWorker:
    def __init__(self, camera_url: str, model: YOLO, camera_id: str = CAMERA_ID_DEFAULT):
        self.camera = LatestCamera(camera_url)
        self.camera_url = camera_url
        self.camera_id = camera_id
        self.model = model
        self.fight_model, self.fight_device = load_fight_model()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.jpeg: Optional[bytes] = None
        self.counts = {}
        self.people = 0
        self.vehicles = 0
        self.inference_ms = 0.0
        self.inference_fps = 0.0
        self.error = None
        self.last_inference_at = None
        self.fight_frames = deque(maxlen=NUM_VIDEO_FRAMES)
        self.last_fight_sample = 0.0
        self.last_fight_inference = 0.0
        self.fight_label = "Collecting frames"
        self.fight_confidence = 0.0
        self.fight_probability = 0.0
        self.fight_inference_ms = 0.0
        self.fight_error = None
        self.fight_alert = False
        self.fight_alert_at = None
        self.fight_streak = 0
        self.last_alert_monotonic = 0.0
        self.incident_active = None
        self.incident_post_frames = []
        self.incident_post_started = None
        self.last_incident_id = None
        self.pre_event_frames = deque(
            maxlen=max(1, int(EVIDENCE_PRE_SECONDS / max(FIGHT_SAMPLE_INTERVAL, 0.01)))
        )
        self.fight_thread = threading.Thread(target=self._fight_loop, daemon=True)
        self.fight_thread.start()
        self.thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.thread.start()

    def _infer_loop(self):
        previous = None
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            frame = self.camera.get()
            if frame is None:
                time.sleep(0.01)
                continue

            sample_now = time.monotonic()
            if sample_now - self.last_fight_sample >= FIGHT_SAMPLE_INTERVAL:
                with self.lock:
                    sampled = frame.copy()
                    self.fight_frames.append(sampled)
                    self.pre_event_frames.append(sampled)
                    if self.incident_active is not None:
                        self.incident_post_frames.append(sampled)
                        if sample_now - self.incident_post_started >= EVIDENCE_POST_SECONDS:
                            self._finalize_incident_locked()
                self.last_fight_sample = sample_now

            try:
                h, w = frame.shape[:2]
                if w > MAX_FRAME_WIDTH:
                    new_h = max(1, int(h * MAX_FRAME_WIDTH / w))
                    frame = cv2.resize(
                        frame, (MAX_FRAME_WIDTH, new_h),
                        interpolation=cv2.INTER_AREA
                    )

                t0 = time.perf_counter()
                result = self.model.predict(
                    source=frame,
                    imgsz=IMG_SIZE,
                    conf=CONFIDENCE,
                    iou=IOU_THRESHOLD,
                    max_det=50,
                    device="intel:gpu",
                    verbose=False,
                )[0]
                latency = (time.perf_counter() - t0) * 1000.0

                counts = Counter()
                if result.boxes is not None:
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for class_id in class_ids:
                        counts[self.model.names[int(class_id)]] += 1

                people = counts.get("person", 0)
                vehicles = sum(
                    counts.get(k, 0)
                    for k in ("car", "motorcycle", "bus", "truck")
                )

                annotated = result.plot()
                ok, encoded = cv2.imencode(
                    ".jpg", annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed.")

                now = time.perf_counter()
                current_fps = 0.0 if previous is None else (
                    1.0 / max(now - previous, 1e-6)
                )
                previous = now

                with self.lock:
                    self.jpeg = encoded.tobytes()
                    self.counts = dict(counts)
                    self.people = people
                    self.vehicles = vehicles
                    self.inference_ms = latency
                    self.inference_fps = current_fps
                    self.error = None
                    self.last_inference_at = time.time()

            except Exception as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.15)

            if INFERENCE_INTERVAL > 0:
                elapsed = time.perf_counter() - loop_start
                wait = INFERENCE_INTERVAL - elapsed
                if wait > 0:
                    self.stop_event.wait(wait)

    def _confirm_incident_locked(self, fight_probability):
        now_mono = time.monotonic()
        if now_mono - self.last_alert_monotonic < ALERT_COOLDOWN_SECONDS:
            return
        if self.incident_active is not None:
            return
        incident_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        now_utc = datetime.now(timezone.utc)
        self.incident_active = {
            "incident_id": incident_id,
            "camera_id": self.camera_id,
            "timestamp_utc": now_utc.isoformat(),
            "fight_probability": round(float(fight_probability), 4),
            "threshold": FIGHT_THRESHOLD,
            "status": "POTENTIAL_FIGHT_REVIEW_REQUIRED",
            "review_status": "pending",
            "evidence_file": None,
            "evidence_status": "recording"
        }
        self.incident_post_frames = []
        self.incident_post_started = now_mono
        self.last_alert_monotonic = now_mono
        self.last_incident_id = incident_id
        self.fight_alert_at = now_utc.timestamp()

    def _finalize_incident_locked(self):
        incident = dict(self.incident_active)
        frames = list(self.pre_event_frames) + list(self.incident_post_frames)
        filename = f"{incident['incident_id']}.mp4"
        output = EVIDENCE_DIR / filename
        # Release the worker lock while encoding to avoid blocking status/frame access.
        self.incident_active = None
        self.incident_post_frames = []
        self.incident_post_started = None
        def save_job():
            ok = write_evidence_video(frames, output, fps=1.0 / max(FIGHT_SAMPLE_INTERVAL, 0.01))
            incident["evidence_file"] = filename if ok else None
            incident["evidence_status"] = "saved" if ok else "failed"
            persist_incident(incident)
        threading.Thread(target=save_job, daemon=True).start()

    def _fight_loop(self):
        """Classify a rolling 16-frame clip independently of the YOLO/video worker."""
        while not self.stop_event.is_set():
            now = time.monotonic()
            with self.lock:
                frames = list(self.fight_frames)
                ready = len(frames) == NUM_VIDEO_FRAMES
                due = now - self.last_fight_inference >= FIGHT_INFERENCE_INTERVAL
            if not ready or not due:
                self.stop_event.wait(0.08)
                continue
            try:
                clip = preprocess_live_clip(frames).to(self.fight_device)
                start = time.perf_counter()
                with torch.inference_mode():
                    logits = self.fight_model(pixel_values=clip).logits
                    probabilities = torch.softmax(logits, dim=1)[0]
                if self.fight_device.type == "xpu":
                    torch.xpu.synchronize()
                elif self.fight_device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000.0
                nonfight_prob = float(probabilities[0].detach().cpu())
                fight_prob = float(probabilities[1].detach().cpu())
                is_fight = fight_prob >= FIGHT_THRESHOLD
                with self.lock:
                    self.fight_probability = fight_prob
                    self.fight_confidence = max(nonfight_prob, fight_prob)
                    self.fight_label = "Fight" if is_fight else "Non-Fight"
                    self.fight_inference_ms = elapsed
                    self.fight_error = None
                    self.last_fight_inference = time.monotonic()
                    self.fight_streak = self.fight_streak + 1 if is_fight else 0
                    self.fight_alert = self.fight_streak >= FIGHT_CONFIRMATIONS
                    if self.fight_streak >= FIGHT_CONFIRMATIONS:
                        self._confirm_incident_locked(fight_prob)
            except Exception as exc:
                with self.lock:
                    self.fight_error = f"{type(exc).__name__}: {exc}"
                    self.fight_label = "Classification error"
                self.stop_event.wait(0.5)

    def snapshot(self):
        with self.lock:
            return {
                "jpeg": self.jpeg,
                "camera_url": self.camera_url,
                "counts": dict(self.counts),
                "people": self.people,
                "vehicles": self.vehicles,
                "total_objects": sum(self.counts.values()),
                "inference_ms": round(self.inference_ms, 2),
                "inference_fps": round(self.inference_fps, 2),
                "error": self.error,
                "last_inference_at": self.last_inference_at,
                "fight_label": self.fight_label,
                "fight_confidence": round(self.fight_confidence, 4),
                "fight_probability": round(self.fight_probability, 4),
                "fight_inference_ms": round(self.fight_inference_ms, 2),
                "fight_frames_ready": len(self.fight_frames),
                "fight_alert": self.fight_alert,
                "fight_alert_at": self.fight_alert_at,
                "fight_error": self.fight_error,
                "fight_threshold": FIGHT_THRESHOLD,
                "fight_device": str(self.fight_device),
                "camera_id": self.camera_id,
                "fight_streak": self.fight_streak,
                "fight_confirmations_required": FIGHT_CONFIRMATIONS,
                "incident_active": self.incident_active is not None,
                "last_incident_id": self.last_incident_id,
                "evidence_pre_seconds": EVIDENCE_PRE_SECONDS,
                "evidence_post_seconds": EVIDENCE_POST_SECONDS,
            }

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.fight_thread.join(timeout=2.0)
        self.camera.close()


def load_model():
    global _model
    if _model is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"OpenVINO model directory not found: {MODEL_DIR}"
            )
        _model = YOLO(str(MODEL_DIR))
    return _model


def load_fight_model():
    """Load the trained two-class VideoMAE checkpoint on Intel XPU when available."""
    global _fight_model, _fight_device
    if _fight_model is not None:
        return _fight_model, _fight_device
    if not FIGHT_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"VideoMAE checkpoint not found: {FIGHT_CHECKPOINT}. "
            "Place models/videomae_best.pth in the project root."
        )
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = VideoMAEForVideoClassification.from_pretrained(
        VIDEOMAE_ID,
        num_labels=2,
        ignore_mismatched_sizes=True,
        label2id={"NonFight": 0, "Fight": 1},
        id2label={0: "NonFight", 1: "Fight"},
    )
    checkpoint = torch.load(FIGHT_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    _fight_model, _fight_device = model, device
    print(f"VideoMAE loaded on {device}; checkpoint epoch={checkpoint.get('epoch', 'unknown')}")
    return _fight_model, _fight_device


def preprocess_live_clip(frames):
    """Match videomae_dataset.py: 16 RGB frames, 224x224, ImageNet normalization."""
    resized = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (VIDEO_FRAME_SIZE, VIDEO_FRAME_SIZE), interpolation=cv2.INTER_AREA)
        resized.append(rgb)
    arr = np.stack(resized).astype(np.float32) / 255.0  # [T,H,W,C]
    tensor = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2)  # [T,C,H,W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)  # [1,T,C,H,W]


@app.get("/")
def health():
    return {
        "service": "I-Mage Live Detection",
        "status": "online",
        "model_directory_exists": MODEL_DIR.exists(),
    }


@app.post("/start")
def start_detection(request: StartRequest):
    global _worker
    camera_url = request.camera_url.strip()
    if not camera_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Camera URL must start with http:// or https://")

    with _worker_lock:
        if _worker is not None:
            _worker.close()
            _worker = None
        try:
            model = load_model()
            load_fight_model()
            _worker = DetectionWorker(camera_url, model, request.camera_id.strip() or CAMERA_ID_DEFAULT)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return {"started": True, "camera_url": camera_url}


@app.post("/stop")
def stop_detection():
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.close()
            _worker = None
    return {"stopped": True}


@app.get("/status")
def status():
    current = _worker
    if current is None:
        return {"running": False, "camera_url": DEFAULT_CAMERA_URL}
    data = current.snapshot()
    data.pop("jpeg", None)
    data["running"] = True
    return data


@app.get("/incidents")
def incidents():
    with _incidents_lock:
        return {"incidents": load_incidents()}


@app.get("/incidents/{incident_id}/evidence")
def incident_evidence(incident_id: str):
    # Restrict lookup to a filename derived from the incident ID.
    if not incident_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in incident_id):
        raise HTTPException(status_code=400, detail="Invalid incident ID")
    path = EVIDENCE_DIR / f"{incident_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evidence clip is not available yet")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


def mjpeg_generator():
    boundary = b"--frame\r\n"
    last_jpeg = None
    while True:
        current = _worker
        if current is None:
            time.sleep(0.1)
            continue
        data = current.snapshot()
        jpeg = data["jpeg"]
        if jpeg is None:
            time.sleep(0.02)
            continue
        if jpeg is last_jpeg:
            time.sleep(0.015)
            continue
        last_jpeg = jpeg
        yield (
            boundary
            + b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            + jpeg
            + b"\r\n"
        )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    print("I-Mage live server starting on http://127.0.0.1:8000")
    print("YOLO target: OpenVINO Intel GPU")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
