import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import tempfile
import subprocess
import imageio_ffmpeg
import hashlib
import html
import time
import threading
from collections import Counter
from pathlib import Path
from torchvision.models.video import r3d_18
from transformers import VideoMAEForVideoClassification

# ============================================================
# I-MAGE CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parent
R3D_CHECKPOINT = ROOT / "models" / "r3d18_best.pth"
VIDEOMAE_CHECKPOINT = ROOT / "models" / "videomae_best.pth"
VIDEOMAE_MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_FRAMES = 16
WINDOW_STRIDE = 8
CLASS_NAMES = ["NonFight", "Fight"]

R3D_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(1, 3, 1, 1)
R3D_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(1, 3, 1, 1)
VMAE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
VMAE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# LIVE OBJECT DETECTION CONFIGURATION
# ============================================================
YOLO_OPENVINO_DIR = ROOT / "yolo26n_openvino_model"
DEFAULT_CAMERA_URL = "http://100.75.50.187:8080/video"
LIVE_INFER_SIZE = 512
LIVE_CONFIDENCE = 0.35


class LatestFrameReader:
    """Continuously read the newest camera frame without building a frame queue."""

    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                "Could not open the IP Webcam stream. Check that the phone's "
                "server is running, the IP/port are correct, and the laptop can "
                "reach the phone."
            )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.03)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.cap.release()


@st.cache_resource
def load_yolo_openvino_model():
    from ultralytics import YOLO

    if not YOLO_OPENVINO_DIR.exists():
        raise FileNotFoundError(
            f"OpenVINO model folder not found: {YOLO_OPENVINO_DIR}. "
            "Export it first using export_yolo_openvino.py."
        )
    return YOLO(str(YOLO_OPENVINO_DIR))


def count_yolo_objects(result, model_names):
    counts = Counter()
    if result.boxes is not None:
        for class_id in result.boxes.cls.cpu().numpy():
            counts[model_names[int(class_id)]] += 1
    people = counts.get("person", 0)
    vehicles = sum(counts.get(name, 0) for name in ("car", "motorcycle", "bus", "truck"))
    return counts, people, vehicles


# ============================================================
# PAGE SETUP + STYLING
# ============================================================
st.set_page_config(
    page_title="I-Mage | Intelligent Video Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --ink:#f5f7ff; --muted:#b6c3e6; --purple:#9a83ff; --cyan:#6ce8f0;
  --pink:#ff8ac4; --green:#66e8bd; --red:#ff8298;
  --glass:rgba(255,255,255,.075); --glass-strong:rgba(255,255,255,.115);
  --glass-line:rgba(218,230,255,.23); --glass-highlight:rgba(255,255,255,.32);
}
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {
 color:var(--ink);
 background:
   radial-gradient(ellipse at 12% 8%,rgba(126,94,255,.25),transparent 34%),
   radial-gradient(ellipse at 88% 12%,rgba(44,208,224,.17),transparent 30%),
   radial-gradient(ellipse at 70% 100%,rgba(239,86,177,.14),transparent 34%),
   linear-gradient(135deg,#080e21 0%,#101a3b 48%,#11152e 100%);
 background-attachment:fixed;
}
.block-container {max-width:1760px;padding:1.25rem 2rem 2.5rem;}
header[data-testid="stHeader"] {background:rgba(8,14,33,.38);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);}
[data-testid="stSidebar"] {
 background:linear-gradient(180deg,rgba(9,15,37,.78),rgba(28,31,76,.72));
 border-right:1px solid var(--glass-line);
 backdrop-filter:blur(28px);-webkit-backdrop-filter:blur(28px);
}
[data-testid="stSidebar"] > div:first-child {padding-top:1.25rem;}
[data-testid="stSidebar"] * {color:#f1f4ff;}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {gap:8px;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
 padding:11px 12px;border-radius:14px;
 background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.055);
 backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
 transition:all .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
 background:rgba(153,133,255,.17);border-color:rgba(201,190,255,.28);
 box-shadow:inset 0 1px rgba(255,255,255,.15),0 8px 24px rgba(0,0,0,.12);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {font-weight:650;font-size:.9rem;}
[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.15);margin:1.3rem 0;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
 color:#aebbe0!important;letter-spacing:.12em;font-size:.66rem;font-weight:800;
}
[data-testid="stSidebar"] code {
 background:rgba(151,132,255,.18);border:1px solid rgba(194,181,255,.32);
 border-radius:9px;color:#fff!important;padding:4px 8px;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {color:#fff;}

.hero {
 position:relative;overflow:hidden;isolation:isolate;
 background:linear-gradient(115deg,rgba(91,61,185,.58),rgba(47,103,160,.43) 55%,rgba(168,74,157,.35));
 border:1px solid rgba(255,255,255,.28);border-radius:26px;
 padding:34px 38px;margin:0 0 20px;color:#fff;
 box-shadow:0 18px 50px rgba(0,0,0,.2),inset 0 1px 0 rgba(255,255,255,.34),inset 0 -1px 0 rgba(255,255,255,.08);
 backdrop-filter:blur(26px) saturate(160%);-webkit-backdrop-filter:blur(26px) saturate(160%);
}
.hero:before {
 content:"";position:absolute;z-index:-1;width:340px;height:340px;right:-75px;top:-220px;
 border:1px solid rgba(255,255,255,.3);border-radius:50%;
 box-shadow:0 0 0 30px rgba(255,255,255,.055),0 0 0 66px rgba(255,255,255,.04),0 0 0 105px rgba(255,255,255,.025);
}
.hero:after {
 content:"";position:absolute;z-index:-1;width:180px;height:180px;right:28%;bottom:-145px;
 border-radius:50%;background:rgba(255,255,255,.2);filter:blur(10px);
}
.hero h1 {
 margin:0;font-family:'Space Grotesk',sans-serif;font-size:clamp(2.3rem,4vw,3.15rem);
 line-height:1.04;font-weight:700;letter-spacing:-1.8px;color:#fff;
 text-shadow:0 3px 18px rgba(15,10,62,.2);
}
.hero p {margin:11px 0 0;color:#e8eaff;font-size:.97rem;max-width:800px;}

.panel,.kpi,.stepper,[data-testid="stMetric"],div[data-testid="stExpander"] {
 background:linear-gradient(135deg,rgba(255,255,255,.13),rgba(255,255,255,.045))!important;
 border:1px solid var(--glass-line)!important;border-radius:18px!important;
 box-shadow:0 12px 32px rgba(0,0,0,.16),inset 0 1px 0 rgba(255,255,255,.22),inset 0 -1px 0 rgba(255,255,255,.06)!important;
 backdrop-filter:blur(24px) saturate(155%)!important;
 -webkit-backdrop-filter:blur(24px) saturate(155%)!important;
 color:#f5f7ff!important;
}
.panel {padding:19px 20px;margin-bottom:15px;}
.panel-title {
 display:flex;align-items:center;gap:8px;font-family:'Space Grotesk',sans-serif;
 font-size:1.02rem;font-weight:700;color:#fff;margin:0 0 15px;
}
.kpi {position:relative;overflow:hidden;padding:15px 16px;min-height:98px;margin-bottom:9px;}
.kpi:after {
 content:"";position:absolute;width:100px;height:100px;right:-48px;top:-55px;
 border-radius:50%;background:rgba(190,181,255,.15);filter:blur(1px);
}
.kpi-label {color:#c5d0ed;font-size:.75rem;font-weight:700;letter-spacing:.025em;margin-bottom:7px;}
.kpi-value {color:#fff;font-family:'Space Grotesk',sans-serif;font-size:1.7rem;line-height:1.1;font-weight:700;}
.kpi-note {color:#aebbe0;font-size:.71rem;margin-top:5px;}

.status-good,.status-bad,.status-safe {
 display:block;border-radius:14px;padding:12px 14px;font-size:.83rem;font-weight:800;margin:7px 0 13px;
 border:1px solid rgba(255,255,255,.22);
 box-shadow:inset 0 1px rgba(255,255,255,.2),0 8px 20px rgba(0,0,0,.12);
}
.status-good {display:inline-block;background:rgba(24,190,145,.25);color:#a5ffe2;border-radius:30px;padding:7px 12px;font-size:.74rem;}
.status-bad {color:#ffe0e7;background:rgba(237,64,106,.25);}
.status-safe {color:#c5ffeb;background:rgba(24,190,145,.23);}
.section-label {color:#f4f5ff;font-family:'Space Grotesk',sans-serif;font-size:1.04rem;font-weight:700;margin:2px 0 12px;}

.stepper {
 display:flex;align-items:center;justify-content:space-between;gap:12px;
 padding:15px 21px;margin:8px 0 17px;
}
.step {display:flex;align-items:center;gap:11px;color:#fff;font-weight:700;}
.step-num {
 flex:0 0 auto;width:38px;height:38px;border-radius:13px;display:flex;align-items:center;justify-content:center;
 color:white;background:linear-gradient(135deg,rgba(255,138,196,.9),rgba(130,111,255,.85));
 border:1px solid rgba(255,255,255,.35);font-weight:800;
 box-shadow:0 6px 18px rgba(130,77,255,.28),inset 0 1px rgba(255,255,255,.35);
}
.step-sub {display:block;color:#bdc9e8;font-size:.7rem;font-weight:500;margin-top:2px;}

div.stButton > button,.stDownloadButton > button,a[data-testid="stLinkButton"] {
 border-radius:13px!important;min-height:43px;font-weight:700;color:#f7f8ff!important;
 border:1px solid rgba(255,255,255,.25)!important;
 background:rgba(255,255,255,.09)!important;
 box-shadow:inset 0 1px rgba(255,255,255,.2),0 8px 20px rgba(0,0,0,.12)!important;
 backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
 transition:transform .15s ease,box-shadow .15s ease;
}
div.stButton > button:hover,.stDownloadButton > button:hover {
 transform:translateY(-1px);background:rgba(255,255,255,.16)!important;
 border-color:rgba(210,202,255,.55)!important;box-shadow:0 10px 24px rgba(0,0,0,.18)!important;
}
div.stButton > button[kind="primary"] {
 background:linear-gradient(105deg,rgba(125,91,255,.88),rgba(190,91,223,.76),rgba(255,113,176,.72))!important;
 border:1px solid rgba(255,255,255,.36)!important;color:#fff!important;min-height:47px;
 box-shadow:0 9px 24px rgba(125,91,255,.25),inset 0 1px rgba(255,255,255,.35)!important;
}
div.stButton > button[kind="primary"]:hover {
 background:linear-gradient(105deg,rgba(142,112,255,.95),rgba(205,111,234,.88),rgba(255,135,190,.85))!important;
}
div[data-testid="stFileUploader"] {
 background:linear-gradient(135deg,rgba(255,255,255,.12),rgba(255,255,255,.045))!important;
 border:1px dashed rgba(211,222,255,.48)!important;border-radius:16px!important;padding:10px;
 box-shadow:inset 0 1px rgba(255,255,255,.2),0 10px 25px rgba(0,0,0,.12);
 backdrop-filter:blur(22px) saturate(150%);-webkit-backdrop-filter:blur(22px) saturate(150%);
 color:#fff!important;
}
div[data-testid="stFileUploader"]:hover {border-color:rgba(217,183,255,.8)!important;background:rgba(255,255,255,.15)!important;}
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span {color:#e8edff!important;}
[data-testid="stMetric"] {padding:12px 14px!important;}
[data-testid="stMetric"] label,[data-testid="stMetric"] [data-testid="stMetricLabel"] {color:#cbd5f1!important;}
[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#fff!important;}
[data-testid="stDataFrame"] {
 border:1px solid var(--glass-line);border-radius:14px;overflow:hidden;
 background:rgba(255,255,255,.06);backdrop-filter:blur(18px);
}
[data-testid="stProgressBar"] > div {background:rgba(255,255,255,.12);border-radius:20px;}
[data-testid="stProgressBar"] > div > div {background:linear-gradient(90deg,#ff8ac4,#9a83ff,#6ce8f0);border-radius:20px;}

.stSelectbox > div > div,.stTextInput > div > div,.stNumberInput > div > div,
.stMultiSelect > div > div,.stDateInput > div > div,.stTimeInput > div > div {
 border-radius:12px!important;border:1px solid rgba(218,230,255,.27)!important;
 background:rgba(255,255,255,.085)!important;color:#fff!important;
 box-shadow:inset 0 1px rgba(255,255,255,.13),0 7px 18px rgba(0,0,0,.09);
 backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
}
.stSelectbox label,.stTextInput label,.stFileUploader label,.stNumberInput label,
.stMultiSelect label,.stDateInput label,.stTimeInput label {color:#dce5ff!important;font-weight:650;}
.stSelectbox [data-baseweb="select"] *, .stTextInput input,.stNumberInput input {color:#fff!important;}
.stSelectbox svg {fill:#e5eaff!important;}
.stSelectbox [data-baseweb="select"] > div {background:transparent!important;}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {
 background:rgba(25,35,75,.96)!important;color:#fff!important;
 border:1px solid rgba(218,230,255,.25)!important;border-radius:13px!important;
 backdrop-filter:blur(24px)!important;
}
[role="option"] {color:#f3f5ff!important;background:transparent!important;}
[role="option"]:hover,[aria-selected="true"] {background:rgba(144,120,255,.3)!important;}
.stSlider [data-baseweb="slider"] div {color:#fff;}
.stSlider [data-baseweb="slider"] [role="slider"] {background:#c5b7ff;border:2px solid white;box-shadow:0 0 0 5px rgba(154,131,255,.18);}
.stCheckbox label,.stRadio label {color:#f4f6ff!important;}
div[data-testid="stAlert"] {
 border-radius:14px!important;border:1px solid rgba(218,230,255,.22)!important;
 background:rgba(255,255,255,.09)!important;color:#f4f6ff!important;
 backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
}
div[data-testid="stExpander"] {overflow:hidden;}
div[data-testid="stExpander"] summary {color:#fff!important;}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {color:#e2e9ff!important;}
.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stMarkdownContainer"] li {color:#dce5ff;}
.stApp [data-testid="stMarkdownContainer"] strong {color:#fff;}
.stApp hr {border-color:rgba(218,230,255,.18);margin:1.2rem 0;}
.stApp [data-testid="stCaptionContainer"] p {color:#aebde1;}
.stApp [data-testid="stRadio"] label p {color:#f4f6ff;}
.stApp input {color:#fff;}
.stApp [data-testid="stSelectbox"] svg {color:#fff;}
footer {visibility:hidden;}
@media(max-width:900px) {
 .block-container {padding:1rem .8rem 2rem;}
 .hero {padding:24px 22px;border-radius:19px;}
 .stepper {flex-wrap:wrap;padding:13px;}
 .step {min-width:42%;}
 .panel {padding:15px;}
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_r3d_model():
    if not R3D_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {R3D_CHECKPOINT}")
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(R3D_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_videomae_model():
    if not VIDEOMAE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {VIDEOMAE_CHECKPOINT}")
    model = VideoMAEForVideoClassification.from_pretrained(
        VIDEOMAE_MODEL_ID,
        num_labels=2,
        ignore_mismatched_sizes=True,
        label2id={"NonFight": 0, "Fight": 1},
        id2label={0: "NonFight", 1: "Fight"},
    )
    checkpoint = torch.load(VIDEOMAE_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


# ============================================================
# VIDEO UTILITIES
# ============================================================
def prepare_video_preview(video_bytes, filename):
    ext = Path(filename).suffix.lower()
    if ext in [".mp4", ".mov", ".m4v"]:
        return video_bytes
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{ext or '.avi'}"
        dst = Path(tmp) / "browser_preview.mp4"
        src.write_bytes(video_bytes)
        command = [
            ffmpeg, "-y", "-i", str(src), "-c:v", "libx264",
            "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(dst)
        ]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("Preview conversion failed: " + proc.stderr[-1200:])
        return dst.read_bytes()


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError("Invalid video information: FPS or frame count is zero.")
    return fps, total_frames


def get_window_starts(total_frames):
    if total_frames <= NUM_FRAMES:
        return [0]
    starts = list(range(0, total_frames - NUM_FRAMES + 1, WINDOW_STRIDE))
    last_start = total_frames - NUM_FRAMES
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def read_video_window(video_path, start_frame, total_frames, image_size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to read video.")
    end_frame = min(start_frame + NUM_FRAMES, total_frames)
    indices = np.linspace(start_frame, max(start_frame, end_frame - 1), NUM_FRAMES, dtype=int)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()
        if not success:
            cap.release()
            if frames:
                frames.append(frames[-1].copy())
                continue
            raise RuntimeError(f"Unable to read frame {index}.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))
        frames.append(frame)
    cap.release()
    if len(frames) != NUM_FRAMES:
        raise RuntimeError("Incorrect number of sampled frames.")
    video = np.stack(frames).astype(np.float32) / 255.0
    return np.transpose(video, (0, 3, 1, 2))  # [T,C,H,W]


def prepare_r3d_input(frames):
    frames = (frames - R3D_MEAN) / R3D_STD
    frames = np.transpose(frames, (1, 0, 2, 3))  # [C,T,H,W]
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float().unsqueeze(0)
    return tensor.to(DEVICE)


def prepare_videomae_input(frames):
    frames = (frames - VMAE_MEAN) / VMAE_STD
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float().unsqueeze(0)  # [B,T,C,H,W]
    return tensor.to(DEVICE)


def run_detection(model, model_name, video_path, progress_bar, status_text):
    fps, total_frames = get_video_info(video_path)
    duration = total_frames / fps
    starts = get_window_starts(total_frames)
    rows = []
    image_size = 112 if model_name == "R3D-18" else 224

    for window_number, start_frame in enumerate(starts, 1):
        frames = read_video_window(video_path, start_frame, total_frames, image_size)
        inputs = prepare_r3d_input(frames) if model_name == "R3D-18" else prepare_videomae_input(frames)
        with torch.inference_mode():
            logits = model(inputs) if model_name == "R3D-18" else model(pixel_values=inputs).logits
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        predicted_class = int(np.argmax(probabilities))
        start_time = start_frame / fps
        end_time = min((start_frame + NUM_FRAMES) / fps, duration)
        rows.append({
            "Window": window_number,
            "Start Time (s)": round(start_time, 2),
            "End Time (s)": round(end_time, 2),
            "Prediction": CLASS_NAMES[predicted_class],
            "Confidence (%)": round(float(probabilities[predicted_class]) * 100, 2),
            "Fight Probability (%)": round(float(probabilities[1]) * 100, 2),
            "NonFight Probability (%)": round(float(probabilities[0]) * 100, 2),
        })
        progress_bar.progress(window_number / len(starts))
        status_text.caption(f"Analysing temporal window {window_number} of {len(starts)}")
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(rows), fps, duration


# ============================================================
# SESSION STATE + HELPERS
# ============================================================
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_key" not in st.session_state:
    st.session_state.analysis_key = None


def save_analysis(results, filename, model_name, duration, fps):
    safe_stem = Path(filename).stem.replace(" ", "_")
    model_tag = model_name.lower().replace("-", "")
    out = OUTPUT_DIR / f"{safe_stem}_{model_tag}_results.csv"
    results.to_csv(out, index=False)
    st.session_state.analysis_result = {
        "results": results,
        "filename": filename,
        "model": model_name,
        "duration": duration,
        "fps": fps,
        "output": str(out),
    }


def kpi(label, value, note=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{html.escape(str(label))}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        f'<div class="kpi-note">{html.escape(str(note))}</div></div>',
        unsafe_allow_html=True,
    )


def render_results(data):
    results = data["results"]
    total = len(results)
    fight_count = int((results["Prediction"] == "Fight").sum())
    safe_count = total - fight_count
    fight_pct = (fight_count / total * 100) if total else 0
    avg_fight_prob = float(results["Fight Probability (%)"].mean()) if total else 0
    max_fight_prob = float(results["Fight Probability (%)"].max()) if total else 0

    left, right = st.columns([1.02, 1.0], gap="medium")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Detection Summary</div>', unsafe_allow_html=True)
        if fight_count:
            st.markdown(
                f'<div class="status-bad">⚠ FIGHT ACTIVITY FLAGGED<br>'
                f'<span style="font-size:.78rem;font-weight:500">Potential fight activity in {fight_count} of {total} analysed windows. Review the timestamps.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-safe">✓ NO FIGHT WINDOWS FLAGGED<br>'
                '<span style="font-size:.78rem;font-weight:500">The model classified all analysed windows as NonFight.</span></div>',
                unsafe_allow_html=True,
            )
        a, b = st.columns(2)
        with a:
            kpi("Video duration", f'{data["duration"]:.2f} sec')
            kpi("Fight windows", fight_count, f"{fight_pct:.1f}% of all windows")
        with b:
            kpi("Total windows", total, f'{data["fps"]:.1f} FPS source video')
            kpi("NonFight windows", safe_count, f"{100-fight_pct:.1f}% of all windows")
        st.markdown("**Fight-window distribution**")
        st.progress(fight_pct / 100)
        st.caption(f"{fight_pct:.2f}% Fight predictions · {100-fight_pct:.2f}% NonFight predictions")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">Model Confidence</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            kpi("Average fight probability", f"{avg_fight_prob:.1f}%")
        with b:
            kpi("Peak fight probability", f"{max_fight_prob:.1f}%")
        chart = results[["Start Time (s)", "Fight Probability (%)"]].set_index("Start Time (s)")
        st.markdown("**Fight probability over time**")
        st.line_chart(chart, height=205)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Window-by-Window Predictions</div>', unsafe_allow_html=True)
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fight Probability (%)": st.column_config.ProgressColumn(
                "Fight Probability", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

    fight_rows = results[results["Prediction"] == "Fight"]
    if not fight_rows.empty:
        st.markdown('<div class="panel"><div class="panel-title">Potential Fight Timestamps</div>', unsafe_allow_html=True)
        st.dataframe(
            fight_rows[["Window", "Start Time (s)", "End Time (s)", "Confidence (%)", "Fight Probability (%)"]],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "↓  Download Results (CSV)",
        data=csv_data,
        file_name=f'{Path(data["filename"]).stem}_{data["model"].lower().replace("-", "")}_results.csv',
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f'Analysis CSV saved locally: {data["output"]}')
    st.caption("Predictions are model outputs for review; they are not a definitive determination that violence occurred.")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 4px 22px 4px">
          <div style="font-family:Space Grotesk,sans-serif;font-size:1.8rem;font-weight:700;letter-spacing:-1px">⬡ I-Mage</div>
          <div style="font-size:.82rem;color:#c6ccef;margin-top:8px;line-height:1.5">
            Intelligent Violence and<br>Non-Violence Detection
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page_selection = st.radio(
        "Navigation",
        ["🎞️  Video Detection", "📡  Live Detection", "📊  Model Comparison", "🗂️  Results History", "ⓘ  About"],
        label_visibility="collapsed",
    )
    page = page_selection.split("  ", 1)[-1]
    st.markdown("---")
    st.caption("SYSTEM STATUS")
    st.markdown(f"**Execution device**  \n`{str(DEVICE).upper()}`")
    st.markdown(f"**Input frames**  \n`{NUM_FRAMES}`")
    st.markdown(f"**Window stride**  \n`{WINDOW_STRIDE}`")
    st.markdown("---")
    st.caption("I-Mage · Video Intelligence Module")


# ============================================================
# PAGE: VIDEO DETECTION
# ============================================================
if page == "Video Detection":
    st.markdown(
        """
        <div class="hero">
          <h1>I-Mage</h1>
          <div style="font-size:1.25rem;font-weight:750;margin-top:2px">
            Intelligent Violence and Non-Violence Detection System
          </div>
          <p>Video-based fight detection using deep learning and temporal analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stepper">
          <div class="step"><span class="step-num">1</span><span>Upload Video<span class="step-sub">Choose a video file</span></span></div>
          <span style="color:#a5acc4;font-size:1.4rem">›</span>
          <div class="step"><span class="step-num" style="background:#aab3d0">2</span><span>Analysis<span class="step-sub">AI model processes the video</span></span></div>
          <span style="color:#a5acc4;font-size:1.4rem">›</span>
          <div class="step"><span class="step-num" style="background:#aab3d0">3</span><span>Results<span class="step-sub">View detection results and insights</span></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    settings_col, video_col, result_col = st.columns([0.88, 1.48, 1.12], gap="small")

    with settings_col:
        st.markdown('<div class="panel"><div class="panel-title">⚙ Detection Settings</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Selected Model", ["VideoMAE", "R3D-18"])
        st.selectbox("Execution Device", [str(DEVICE).upper()], disabled=True)
        st.selectbox("Input Frames", [NUM_FRAMES], disabled=True)
        st.markdown("**Detection classes**")
        st.markdown("🟢 **NonFight**  \n🔴 **Fight**")
        st.markdown("---")
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv", "m4v"],
            help="Supported formats: MP4, AVI, MOV, MKV and M4V.",
            key="video_uploader",
        )
        if uploaded_file:
            st.caption(f"✓ {uploaded_file.name} · {uploaded_file.size / (1024*1024):.2f} MB")
        analyse = st.button(
            "▷  Analyse Video",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with video_col:
        st.markdown('<div class="panel"><div class="panel-title">▣ Uploaded Video</div>', unsafe_allow_html=True)
        if uploaded_file:
            video_bytes = uploaded_file.getvalue()
            digest = hashlib.sha256(video_bytes).hexdigest()[:16]
            try:
                preview = prepare_video_preview(video_bytes, uploaded_file.name)
                st.video(preview)
            except Exception as e:
                st.warning(f"Browser preview could not be prepared: {e}")
            st.caption(f"File: {uploaded_file.name}")
            if st.session_state.analysis_result and st.session_state.analysis_key == (digest, model_choice):
                st.markdown('<span class="status-good">✓ Analysis completed</span>', unsafe_allow_html=True)
            else:
                st.info("Video ready for analysis. Select a model and click Analyse Video.")
        else:
            st.markdown(
                """
                <div style="height:275px;border:1px dashed rgba(218,230,255,.48);border-radius:16px;background:linear-gradient(135deg,rgba(255,255,255,.13),rgba(255,255,255,.045));backdrop-filter:blur(24px);
                            display:flex;flex-direction:column;align-items:center;justify-content:center;color:#dce5ff">
                  <div style="font-size:3rem;color:#6849e9">▷</div>
                  <div style="font-weight:750;color:#ffffff">Video preview</div>
                  <div style="font-size:.8rem;color:#b8c6e8">Upload a video to preview it here</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with result_col:
        st.markdown('<div class="panel"><div class="panel-title">▥ Detection Results</div>', unsafe_allow_html=True)
        current = st.session_state.analysis_result
        if current:
            rr = current["results"]
            fc = int((rr["Prediction"] == "Fight").sum())
            if fc:
                st.markdown('<div class="status-bad">⚠ FIGHT DETECTED<br><span style="font-size:.76rem;font-weight:500">Potential fight activity detected in this analysis.</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-safe">✓ NO FIGHT WINDOWS FLAGGED<br><span style="font-size:.76rem;font-weight:500">No Fight predictions in this analysis.</span></div>', unsafe_allow_html=True)
            x, y = st.columns(2)
            with x:
                kpi("Video duration", f'{current["duration"]:.2f} sec')
                kpi("Fight windows", fc)
            with y:
                kpi("Total windows", len(rr))
                kpi("NonFight windows", len(rr)-fc)
            st.markdown("**Fight window distribution**")
            st.progress(fc / len(rr))
            st.caption(f"Fight {fc/len(rr)*100:.1f}%  ·  NonFight {(len(rr)-fc)/len(rr)*100:.1f}%")
        else:
            st.info("Detection results will appear here after analysis.")
            x, y = st.columns(2)
            with x:
                kpi("Video duration", "—")
                kpi("Fight windows", "—")
            with y:
                kpi("Total windows", "—")
                kpi("NonFight windows", "—")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyse and uploaded_file:
        video_bytes = uploaded_file.getvalue()
        digest = hashlib.sha256(video_bytes).hexdigest()[:16]
        temp_path = None
        try:
            suffix = Path(uploaded_file.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_bytes)
                temp_path = Path(tmp.name)

            with st.spinner(f"Loading {model_choice} model on {str(DEVICE).upper()}..."):
                model = load_r3d_model() if model_choice == "R3D-18" else load_videomae_model()

            st.markdown("### Analysing video")
            progress = st.progress(0)
            status = st.empty()
            results, fps, duration = run_detection(model, model_choice, temp_path, progress, status)
            save_analysis(results, uploaded_file.name, model_choice, duration, fps)
            st.session_state.analysis_key = (digest, model_choice)
            st.success("Video analysis completed.")
            st.rerun()
        except Exception as error:
            st.error(f"Detection failed: {error}")
            st.exception(error)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    if st.session_state.analysis_result:
        st.markdown("---")
        data = st.session_state.analysis_result
        st.markdown(f"## Analysis Report · {html.escape(data['filename'])}")
        render_results(data)


# ============================================================
# PAGE: LIVE DETECTION
# ============================================================
elif page == "Live Detection":
    import requests

    st.markdown(
        '<div class="hero"><h1>Live Camera Detection</h1>'
        '<p>Low-latency live stream with YOLO26n person detection and VideoMAE fight classification.</p></div>',
        unsafe_allow_html=True,
    )

    LIVE_API = "http://127.0.0.1:8000"
    if "live_server_url" not in st.session_state:
        st.session_state.live_server_url = "http://100.77.216.203:8080/video"

    st.markdown(
        """
        <style>
        .live-video-wrap {background:#10152b;border:1px solid #30365d;border-radius:14px;
            padding:8px;overflow:hidden}
        .live-video-wrap img {width:100%;height:auto;display:block;object-fit:contain;
            border-radius:9px;background:#080b16}
        .live-note {color:#8992b3;font-size:12px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    control_col, status_col = st.columns([1.5, 1], gap="medium")
    with control_col:
        st.markdown('<div class="panel"><div class="panel-title">Camera Connection</div>', unsafe_allow_html=True)
        camera_url = st.text_input(
            "Phone 1 IP Webcam URL",
            value=st.session_state.live_server_url,
            key="live_server_url_input",
            help="Example: http://PHONE-IP:8080/video",
        )
        a, b = st.columns(2)
        with a:
            start_camera = st.button("Start Detection", type="primary", use_container_width=True)
        with b:
            stop_camera = st.button("Stop Detection", use_container_width=True)
        st.caption("Video is streamed directly from the FastAPI MJPEG server; Streamlit no longer redraws every frame.")
        st.markdown("</div>", unsafe_allow_html=True)

    with status_col:
        st.markdown('<div class="panel"><div class="panel-title">Runtime Status</div>', unsafe_allow_html=True)
        status_placeholder = st.empty()
        st.markdown('<div class="live-note">YOLO and VideoMAE run in separate workers. Video streaming remains independent of fight classification.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if start_camera:
        try:
            response = requests.post(
                f"{LIVE_API}/start",
                json={"camera_url": camera_url.strip()},
                timeout=180,
            )
            response.raise_for_status()
            st.success("Camera and VideoMAE fight-classification workers started.")
        except Exception as exc:
            st.error(f"Could not start the live server. Start live_server.py first. Details: {exc}")

    if stop_camera:
        try:
            requests.post(f"{LIVE_API}/stop", timeout=5).raise_for_status()
            st.success("Camera worker stopped.")
        except Exception as exc:
            st.error(f"Could not contact live server: {exc}")

    def fetch_live_status():
        response = requests.get(f"{LIVE_API}/status", timeout=2)
        response.raise_for_status()
        return response.json()

    try:
        initial_status = fetch_live_status()
        running = initial_status.get("running", False)

        video_col, metrics_col = st.columns([1.8, 1], gap="medium")
        with video_col:
            st.markdown('<div class="panel"><div class="panel-title">Phone 1 · Low-Latency Live Feed</div>', unsafe_allow_html=True)
            if running:
                st.markdown(
                    f'<div class="live-video-wrap"><img src="{LIVE_API}/video_feed?ts={int(time.time())}" '
                    'alt="Live YOLO detection stream"></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Start the camera to display the live feed.")
            st.caption(f"Source: {html.escape(initial_status.get('camera_url') or camera_url)}")
            st.markdown("</div>", unsafe_allow_html=True)

        with metrics_col:
            # This fragment refreshes only the metric panel every second.
            # The MJPEG <img> above remains outside the fragment and is not redrawn.
            @st.fragment(run_every=1)
            def live_metrics_panel():
                try:
                    live_status = fetch_live_status()
                except requests.RequestException as exc:
                    st.error(f"Cannot read live metrics: {exc}")
                    return

                is_running = live_status.get("running", False)
                st.markdown('<div class="panel"><div class="panel-title">Detection Metrics</div>', unsafe_allow_html=True)
                st.markdown(
                    '<span class="status-good">● DETECTION RUNNING</span>' if is_running
                    else '<span class="status-bad">● DETECTION STOPPED</span>',
                    unsafe_allow_html=True,
                )
                a, b = st.columns(2)
                with a:
                    kpi("People", live_status.get("people", 0), "person class")
                with b:
                    kpi("Vehicles", live_status.get("vehicles", 0), "car / motorcycle / bus / truck")
                kpi("Inference latency", f'{live_status.get("inference_ms", 0):.1f} ms', "YOLO model")
                kpi("Inference FPS", f'{live_status.get("inference_fps", 0):.1f}', "completed inferences")
                kpi("Detected objects", live_status.get("total_objects", 0), "all detected classes")
                st.markdown("---")
                st.markdown("**VideoMAE · Fight classification**")
                fight_label = live_status.get("fight_label", "Waiting")
                fight_prob = float(live_status.get("fight_probability", 0.0))
                frames_ready = int(live_status.get("fight_frames_ready", 0))
                if fight_label == "Fight":
                    st.error(f"POTENTIAL FIGHT · {fight_prob * 100:.1f}% fight probability")
                elif fight_label == "Non-Fight":
                    st.success(f"NON-FIGHT · {fight_prob * 100:.1f}% fight probability")
                else:
                    st.info(f"{fight_label} · {frames_ready}/16 frames collected")
                kpi("VideoMAE latency", f'{live_status.get("fight_inference_ms", 0):.1f} ms', live_status.get("fight_device", "device pending"))
                st.caption(
                    f"Fight threshold: {float(live_status.get('fight_threshold', 0.6)):.0%} · "
                    f"Rolling 16-frame window · Buffer: {frames_ready}/16"
                )
                st.caption(
                    f"Fight probability: {fight_prob:.1%} · "
                    f"Predicted-class confidence: {float(live_status.get('fight_confidence', 0.0)):.1%}"
                )
                if live_status.get("fight_alert"):
                    st.warning("Potential fight alert is active. Review the source footage.")
                if live_status.get("fight_error"):
                    st.warning(f"VideoMAE: {live_status['fight_error']}")
                counts = live_status.get("counts", {})
                if counts:
                    st.markdown("**Current object counts**")
                    st.write(" · ".join(f"{name.title()}: {count}" for name, count in counts.items()))
                if live_status.get("error"):
                    st.error(live_status["error"])
                st.markdown("</div>", unsafe_allow_html=True)

            live_metrics_panel()

        # Incident history is refreshed separately so it does not redraw the live MJPEG feed.
        @st.fragment(run_every=2)
        def incident_history_panel():
            st.markdown(
                '<div class="panel"><div class="panel-title">Incident Alerts & Evidence</div>',
                unsafe_allow_html=True,
            )
            try:
                incident_response = requests.get(f"{LIVE_API}/incidents", timeout=3)
                incident_response.raise_for_status()
                incident_items = incident_response.json().get("incidents", [])
            except requests.RequestException as exc:
                st.warning(f"Could not load incident history: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            if not incident_items:
                st.info("No incidents recorded yet. Confirmed alerts will appear here after the evidence clip is saved.")
            else:
                st.caption(f"{len(incident_items)} recorded incident(s) · newest first")
                for incident in incident_items:
                    incident_id = str(incident.get("incident_id", ""))
                    camera_id = str(incident.get("camera_id", "Unknown camera"))
                    timestamp = str(incident.get("timestamp_utc", "Timestamp unavailable"))
                    probability = float(incident.get("fight_probability", 0.0))
                    status_text = str(incident.get("status", "Review required"))
                    evidence_status = str(incident.get("evidence_status", "pending"))
                    heading = f"{camera_id} · {probability:.1%} fight probability · {timestamp}"
                    with st.expander(heading, expanded=False):
                        st.write(f"**Incident ID:** `{incident_id}`")
                        st.write(f"**Status:** {status_text}")
                        st.write(f"**Review:** {str(incident.get('review_status', 'pending')).title()}")
                        st.write(f"**Evidence:** {evidence_status.title()}")
                        if incident_id and evidence_status == "saved":
                            st.link_button(
                                "Open evidence video",
                                f"{LIVE_API}/incidents/{incident_id}/evidence",
                                use_container_width=False,
                            )
                        elif evidence_status == "recording":
                            st.caption("Evidence clip is still being recorded.")
                        elif evidence_status == "failed":
                            st.warning("Evidence video could not be encoded. Review the live server logs.")
                        else:
                            st.caption("Evidence video is not available yet.")
            st.markdown("</div>", unsafe_allow_html=True)

        incident_history_panel()

        st.caption(
            "Metrics automatically refresh every second. The MJPEG video connection remains independent "
            "and streams frames directly from FastAPI."
        )

    except requests.RequestException:
        st.error("Live server is offline. Start it with: python live_server.py")


# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.markdown('<div class="hero"><h1>Model Comparison</h1><p>Validation performance of the trained I-Mage models.</p></div>', unsafe_allow_html=True)
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.markdown('<div class="panel"><div class="panel-title">Model Performance Metrics</div>', unsafe_allow_html=True)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        numeric = comparison.select_dtypes(include="number")
        if not numeric.empty and "Model" in comparison.columns:
            plot_df = comparison.set_index("Model")
            preferred = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "accuracy", "precision", "recall", "f1_score", "roc_auc"] if c in plot_df.columns]
            if preferred:
                st.markdown('<div class="panel"><div class="panel-title">Metric Comparison</div>', unsafe_allow_html=True)
                st.bar_chart(plot_df[preferred])
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No model_comparison.csv found in the outputs folder. Run src/compare_models.py first.")
    report = OUTPUT_DIR / "I-Mage_Model_Evaluation_Report.txt"
    if report.exists():
        with st.expander("View evaluation report"):
            st.text(report.read_text(encoding="utf-8", errors="replace"))


# ============================================================
# PAGE: RESULTS HISTORY
# ============================================================
elif page == "Results History":
    st.markdown('<div class="hero"><h1>Results History</h1><p>Previously saved video analysis outputs.</p></div>', unsafe_allow_html=True)
    csv_files = sorted(OUTPUT_DIR.glob("*_results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        st.info("No saved detection results yet. Run a video analysis first.")
    else:
        selected = st.selectbox("Select an analysis file", csv_files, format_func=lambda p: p.name)
        history = pd.read_csv(selected)
        st.markdown(f'<div class="panel"><div class="panel-title">{html.escape(selected.name)}</div>', unsafe_allow_html=True)
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button("Download selected CSV", history.to_csv(index=False).encode("utf-8"),
                           file_name=selected.name, mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE: ABOUT
# ============================================================
else:
    st.markdown('<div class="hero"><h1>About I-Mage</h1><p>Intelligent Violence and Non-Violence Detection System.</p></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">System Overview</div>
          I-Mage analyses uploaded video clips using deep-learning video classifiers. The video is
          sampled into overlapping temporal windows, and each window receives a Fight or NonFight
          prediction with class probabilities.
        </div>
        <div class="panel">
          <div class="panel-title">Available Models</div>
          <b>VideoMAE</b> — video transformer classifier. 16 sampled frames at 224 × 224 pixels.<br><br>
          <b>R3D-18</b> — 3D convolutional video classifier. 16 sampled frames at 112 × 112 pixels.
        </div>
        <div class="panel">
          <div class="panel-title">Processing Configuration</div>
          <ul>
            <li>Temporal window: 16 frames</li>
            <li>Window stride: 8 frames</li>
            <li>Execution device: Intel XPU when available, otherwise CUDA or CPU</li>
            <li>Output: per-window predictions, probability chart, timestamps and CSV report</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="text-align:center;color:#9299ad;font-size:.75rem;padding:18px 0 4px 0">'
    'I-Mage · Intelligent Video Analysis · Review model predictions alongside the source footage.'
    '</div>',
    unsafe_allow_html=True,
)
