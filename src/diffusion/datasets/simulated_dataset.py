from typing import List, Tuple

import numpy as np
import torch

from src.diffusion.datasets.base_dataset import BaseDataset
from src.diffusion.datasets.colors import Color
from src.diffusion.datasets.shapes import Circle, Rectangle, Shape, Triangle


class SimulatedDataset(BaseDataset):
    """
    Generate random scenes with multiple shapes at different distances.
    Distances are in meters.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 64,
        max_shapes: int = 3,
        min_distance: float = 0.5,
        max_distance: float = 10.0,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_shapes = max_shapes

        if min_distance <= 0:
            raise ValueError("min_distance must be positive")
        if max_distance <= min_distance:
            raise ValueError("max_distance must be greater than min_distance")

        self.min_distance = min_distance
        self.max_distance = max_distance

    def __len__(self) -> int:
        return self.num_samples

    def _generate_random_shape(self) -> Shape:
        # Calculate center region bounds (±1/2 of image size from center)
        quarter_size = self.image_size // 4  # 1/4 of image size
        center = self.image_size // 2  # Image center point

        # Random center within central region
        c_x = np.random.randint(
            center - 2 * quarter_size,  # left bound: center - 1/2 image size
            center + 2 * quarter_size,  # right bound: center + 1/2 image size
        )
        c_y = np.random.randint(
            center - 2 * quarter_size,  # upper bound: center - 1/2 image size
            center + 2 * quarter_size,  # lower bound: center + 1/2 image size
        )

        # Random distance between min_distance and max_distance meters
        distance = np.random.uniform(self.min_distance, self.max_distance)

        # Random color - convert to array first for numpy's random choice
        colors = np.array(list(Color))
        color = np.random.choice(colors)

        shape_type = np.random.choice(["circle", "rectangle", "triangle"])

        # Base size is half of image size
        base_size = self.image_size // 2

        if shape_type == "circle":
            # Radius is exactly half of image size
            radius = base_size
            return Circle(c_x, c_y, radius, distance, color)

        elif shape_type == "rectangle":
            # Length is image size, width is random between 25% and 100% of image size
            length = self.image_size
            width = np.random.randint(self.image_size // 4, self.image_size)
            # Randomly choose orientation
            if np.random.random() > 0.5:
                length, width = width, length
            return Rectangle(c_x, c_y, length, width, distance, color)

        else:  # triangle
            # Size is exactly image size
            size = self.image_size

            # Choose random triangle type
            triangle_type = np.random.choice(
                [
                    "equilateral",
                    "isosceles",
                    "right",
                    "random",  # Keep some randomness but with constraints
                ]
            )

            if triangle_type == "equilateral":
                # All angles are 60 degrees
                angles = (60.0, 60.0, 60.0)

            elif triangle_type == "isosceles":
                # Two angles are equal, third one makes sum 180
                equal_angle = np.random.uniform(30, 75)  # Reasonable range
                third_angle = 180 - 2 * equal_angle
                angles = (equal_angle, equal_angle, third_angle)

            elif triangle_type == "right":
                # One angle is 90, others split remaining 90 degrees
                remaining = np.random.uniform(20, 70)  # Avoid too sharp angles
                angles = (90.0, remaining, 90.0 - remaining)

            else:  # random but constrained
                # Use dirichlet but ensure no angle is too small or too large
                while True:
                    angles = (
                        np.random.dirichlet([2, 2, 2]) * 180
                    )  # Alpha=2 gives more balanced distribution
                    if all(
                        20 <= angle <= 120 for angle in angles
                    ):  # Reasonable angle constraints
                        break
                angles = tuple(angles)

            return Triangle(c_x, c_y, angles, size, distance, color)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        # Initialize a black RGB image: (3, H, W)
        image = np.zeros(shape=(self.image_size, self.image_size, 3), dtype=np.float32)

        # Generate random number of shapes
        num_shapes = np.random.randint(1, self.max_shapes + 1)
        shapes: List[Shape] = []

        # Generate shapes
        for _ in range(num_shapes):
            try:
                shape = self._generate_random_shape()
                shapes.append(shape)
                image = shape.draw(image)
            except ValueError as e:
                print(f"Skipping invalid shape: {e}")
                continue

        # Sort shapes by distance for caption generation
        shapes.sort(key=lambda x: x.distance)

        # Generate caption by combining all shape descriptions
        caption = " ".join(shape.get_caption() for shape in shapes)

        # Convert image to Torch tensor (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.tensor(image)

        return image_tensor, caption
