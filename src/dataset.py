
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from config import (
    TRAIN_FIGHT,
    TRAIN_NONFIGHT,
    VAL_FIGHT,
    VAL_NONFIGHT,
    NUM_FRAMES,
    IMAGE_SIZE,
    CLASS_TO_IDX
)


class ViolenceDataset(Dataset):

    def __init__(self, split="train"):

        self.samples = []

        if split == "train":
            folders = [
                (TRAIN_NONFIGHT, CLASS_TO_IDX["NonFight"]),
                (TRAIN_FIGHT, CLASS_TO_IDX["Fight"])
            ]

        elif split == "val":
            folders = [
                (VAL_NONFIGHT, CLASS_TO_IDX["NonFight"]),
                (VAL_FIGHT, CLASS_TO_IDX["Fight"])
            ]

        else:
            raise ValueError("split must be train or val")

        for folder, label in folders:
            for video_path in sorted(folder.glob("*.avi")):
                self.samples.append((video_path, label))

        print(f"{split} dataset loaded: {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        video_path, label = self.samples[index]

        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened() or total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_indices = np.linspace(
            0, total_frames - 1, NUM_FRAMES
        ).astype(int)

        frames = []

        for frame_index in frame_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = cap.read()

            if not success:
                cap.release()
                raise RuntimeError(
                    f"Cannot decode frame {frame_index}: {video_path}"
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(
                frame,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            frames.append(frame)

        cap.release()

        # T, H, W, C -> T, C, H, W
        video = np.stack(frames)
        video = torch.from_numpy(video.copy()).float() / 255.0
        video = video.permute(0, 3, 1, 2)

        # Kinetics pretrained model normalization
        mean = torch.tensor(
            [0.43216, 0.394666, 0.37645]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.22803, 0.22145, 0.216989]
        ).view(1, 3, 1, 1)

        video = (video - mean) / std

        # T, C, H, W -> C, T, H, W
        video = video.permute(1, 0, 2, 3).contiguous()

        return video, torch.tensor(label, dtype=torch.long)
