
import cv2
from pathlib import Path
from config import (
    TRAIN_FIGHT,
    TRAIN_NONFIGHT,
    VAL_FIGHT,
    VAL_NONFIGHT
)

folders = {
    "Train Fight": TRAIN_FIGHT,
    "Train NonFight": TRAIN_NONFIGHT,
    "Validation Fight": VAL_FIGHT,
    "Validation NonFight": VAL_NONFIGHT
}

total_videos = 0

for name, folder in folders.items():

    videos = list(folder.glob("*.avi"))

    print(f"\n{name}: {len(videos)} videos")
    total_videos += len(videos)

    readable = 0

    for video in videos[:10]:

        cap = cv2.VideoCapture(str(video))

        success, frame = cap.read()

        if success and frame is not None:
            readable += 1

        cap.release()

    print(f"First 10 readable: {readable}/10")

print("\nTotal videos:", total_videos)
