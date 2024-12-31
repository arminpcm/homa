import numpy as np
import torch

from src.diffusion.datasets.base_dataset import BaseDataset


class SimulatedDataset(BaseDataset):
    def __init__(self, num_samples=1000, image_size=64):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random shapes (e.g., circles, squares)
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        label = np.random.choice(["circle", "square"])

        if label == "circle":
            cx, cy = np.random.randint(10, self.image_size - 10, size=2)
            radius = np.random.randint(5, 15)
            for x in range(self.image_size):
                for y in range(self.image_size):
                    if (x - cx) ** 2 + (y - cy) ** 2 < radius**2:
                        image[x, y] = 1
        elif label == "square":
            x1, y1 = np.random.randint(10, self.image_size - 10, size=2)
            side = np.random.randint(5, 15)
            x2, y2 = x1 + side, y1 + side
            image[x1:x2, y1:y2] = 1

        return torch.tensor(image).unsqueeze(0), label
