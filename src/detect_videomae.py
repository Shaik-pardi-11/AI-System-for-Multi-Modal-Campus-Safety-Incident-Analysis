
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from transformers import VideoMAEForVideoClassification


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "videomae_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 16
IMAGE_SIZE = 224
WINDOW_STRIDE = 8

CLASS_NAMES = {
    0: "NonFight",
    1: "Fight"
}

# ImageNet normalization used during training
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================
# 2. DEVICE CONFIGURATION
# ============================================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    print("Intel Arc GPU detected.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("NVIDIA GPU detected.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU.")

print("Selected device:", DEVICE)


# ============================================================
# 3. LOAD TRAINED VIDEOMAE MODEL
# ============================================================

def load_model():

    print("\nLoading VideoMAE model...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        ignore_mismatched_sizes=True,
        label2id={
            "NonFight": 0,
            "Fight": 1
        },
        id2label={
            0: "NonFight",
            1: "Fight"
        }
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    print("VideoMAE checkpoint loaded successfully.")
    print("Model device:", DEVICE)

    return model


# ============================================================
# 4. READ VIDEO WINDOW
# ============================================================

def read_window(video_path, start_frame, total_frames):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    # Read 16 frames from the selected window
    end_frame = min(
        start_frame + NUM_FRAMES,
        total_frames
    )

    frame_indices = np.linspace(
        start_frame,
        max(start_frame, end_frame - 1),
        NUM_FRAMES,
        dtype=int
    )

    frames = []

    for frame_index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        success, frame = cap.read()

        if not success:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
                continue

            cap.release()
            raise RuntimeError(
                f"Unable to read frame {frame_index}"
            )

        # OpenCV BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize to VideoMAE input size
        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        frames.append(frame)

    cap.release()

    if len(frames) != NUM_FRAMES:
        raise RuntimeError(
            f"Expected {NUM_FRAMES} frames, got {len(frames)}"
        )

    # Convert frames to tensor
    video = np.stack(frames)

    video = torch.from_numpy(video).float() / 255.0

    # [T, H, W, C] -> [T, C, H, W]
    video = video.permute(0, 3, 1, 2)

    # Normalize
    mean = MEAN.to(video.device)
    std = STD.to(video.device)

    video = (video - mean) / std

    # IMPORTANT:
    # VideoMAE expects [T, C, H, W].
    # Do NOT change this to [C, T, H, W].
    return video.contiguous()


# ============================================================
# 5. PROCESS VIDEO USING SLIDING WINDOWS
# ============================================================

def detect_video(model, video_path):

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(
            "Invalid video FPS or frame count."
        )

    duration = total_frames / fps

    print("\n================================")
    print("VIDEO INFORMATION")
    print("================================")
    print("File:", video_path.name)
    print("FPS:", round(fps, 2))
    print("Total frames:", total_frames)
    print("Duration:", round(duration, 2), "seconds")

    if total_frames < NUM_FRAMES:
        window_starts = [0]
    else:
        window_starts = list(
            range(
                0,
                total_frames - NUM_FRAMES + 1,
                WINDOW_STRIDE
            )
        )

        # Include the final possible window
        last_start = total_frames - NUM_FRAMES

        if window_starts[-1] != last_start:
            window_starts.append(last_start)

    results = []

    print("\nStarting VideoMAE detection...\n")

    for window_number, start_frame in enumerate(
        window_starts,
        start=1
    ):

        # Read video frames
        video = read_window(
            video_path,
            start_frame,
            total_frames
        )

        # [T, C, H, W] -> [B, T, C, H, W]
        video = video.unsqueeze(0).to(DEVICE)

        # Validate input tensor layout
        if window_number == 1:
            print("VideoMAE input shape:", tuple(video.shape))

        assert video.shape == (
            1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE
        ), f"Incorrect input shape: {video.shape}"

        # Run model inference
        with torch.inference_mode():

            logits = model(
                pixel_values=video
            ).logits

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        probabilities = probabilities[0].cpu().numpy()

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        fight_probability = float(
            probabilities[1]
        )

        nonfight_probability = float(
            probabilities[0]
        )

        start_time = start_frame / fps
        end_time = min(
            (start_frame + NUM_FRAMES) / fps,
            duration
        )

        prediction = CLASS_NAMES[predicted_class]

        result = {
            "window": window_number,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "fight_probability": round(
                fight_probability, 4
            ),
            "nonfight_probability": round(
                nonfight_probability, 4
            )
        }

        results.append(result)

        print(
            f"Window {window_number:03d} | "
            f"{start_time:.2f}s - {end_time:.2f}s | "
            f"{prediction} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

    return results


# ============================================================
# 6. SAVE DETECTION RESULTS
# ============================================================

def save_results(results, video_path):

    if not results:
        print("No detection results to save.")
        return

    df = pd.DataFrame(results)

    video_name = Path(video_path).stem

    output_file = (
        OUTPUT_DIR /
        f"{video_name}_videomae_predictions.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    total_windows = len(df)

    fight_windows = int(
        (df["prediction"] == "Fight").sum()
    )

    nonfight_windows = int(
        (df["prediction"] == "NonFight").sum()
    )

    fight_percentage = (
        fight_windows / total_windows
    ) * 100

    print("\n================================")
    print("DETECTION SUMMARY")
    print("================================")

    print("Total windows:", total_windows)
    print("Fight windows:", fight_windows)
    print("NonFight windows:", nonfight_windows)

    print(
        "Fight window percentage:",
        round(fight_percentage, 2),
        "%"
    )

    if fight_windows > 0:
        print("\nPotential fight windows:")

        for _, row in df[
            df["prediction"] == "Fight"
        ].iterrows():

            print(
                f"{row['start_time']}s - "
                f"{row['end_time']}s | "
                f"Fight probability: "
                f"{row['fight_probability'] * 100:.2f}%"
            )

    print("\nCSV saved successfully:")
    print(output_file)


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    print("\n================================")
    print("I-MAGE - VIDEOMAE DETECTION")
    print("================================")

    model = load_model()

    print("\nEnter the complete video path.")

    video_input = input(
        "Video path: "
    ).strip().strip('"')

    # Handle Windows extended-length path prefix
    if video_input.startswith("\\\\?\\"):
        video_input = video_input[4:]

    video_path = Path(video_input)

    results = detect_video(
        model,
        video_path
    )

    save_results(
        results,
        video_path
    )

    print("\nDetection completed successfully.")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nDetection stopped by user.")

    except Exception as error:
        print("\nERROR:", error)
        raise
