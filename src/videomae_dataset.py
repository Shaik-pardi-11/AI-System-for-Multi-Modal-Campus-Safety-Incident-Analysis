
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

from config import (
    TRAIN_FIGHT,
    TRAIN_NONFIGHT,
    VAL_FIGHT,
    VAL_NONFIGHT
)


class VideoMAEDataset(Dataset):

    def __init__(self, split="train", max_samples=None):

        self.samples = []

        if split == "train":
            folders = [
                (TRAIN_NONFIGHT, 0),
                (TRAIN_FIGHT, 1)
            ]
        elif split == "val":
            folders = [
                (VAL_NONFIGHT, 0),
                (VAL_FIGHT, 1)
            ]
        else:
            raise ValueError("Invalid split")

        for folder, label in folders:
            for path in sorted(folder.glob("*.avi")):
                self.samples.append((path, label))

        if max_samples is not None:
            rng = np.random.default_rng(42)

            # Keep the subset balanced across both classes
            selected = []

            for label in [0, 1]:
                class_samples = [
                    sample for sample in self.samples
                    if sample[1] == label
                ]

                count = min(max_samples // 2, len(class_samples))
                indices = rng.choice(
                    len(class_samples),
                    size=count,
                    replace=False
                )

                selected.extend(
                    class_samples[int(i)] for i in indices
                )

            self.samples = selected

        print(f"{split} videos: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        video_path, label = self.samples[index]

        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if not cap.isOpened() or total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Cannot open: {video_path}")

        indices = np.linspace(
            0, total_frames - 1, 16
        ).astype(int)

        frames = []

        for frame_index in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = cap.read()

            if not success:
                cap.release()
                raise RuntimeError(
                    f"Cannot decode: {video_path}"
                )

            frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame, (224, 224)
            )

            frames.append(frame)

        cap.release()

        # [T, H, W, C] -> [T, C, H, W]
        video = np.stack(frames)

        video = torch.from_numpy(
            video.copy()
        ).float() / 255.0

        video = video.permute(0, 3, 1, 2)

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        video = (video - mean) / std

        return video, torch.tensor(
            label, dtype=torch.long
        )
