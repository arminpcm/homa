from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np

from src.diffusion.datasets.colors import Color


class Shape(ABC):
    def __init__(self, c_x: int, c_y: int, distance: float, color: Color):
        self.validate_parameters(c_x, c_y, distance)
        self.c_x = c_x
        self.c_y = c_y
        self.distance = distance
        self.color = color

    def validate_parameters(self, c_x: int, c_y: int, distance: float) -> None:
        if distance <= 0:
            raise ValueError("Distance must be positive")
        if not isinstance(c_x, (int, np.integer)) or not isinstance(
            c_y, (int, np.integer)
        ):
            raise ValueError("Center coordinates must be integers")

    @abstractmethod
    def draw(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_caption(self) -> str:
        pass

    def _get_base_caption(self) -> str:
        return (
            f"{self.__class__.__name__} with color {self.color.name} "
            f"is at distance {self.distance:.1f} and centered at "
            f"({self.c_x}, {self.c_y})"
        )

    @property
    def color_rgb(self) -> Tuple[float, float, float]:
        """Convert color to format expected by cv2"""
        return self.color.rgb

    @property
    def color_rgb_int(self) -> Tuple[int, int, int]:
        """Convert color to format expected by cv2 (RGB integers 0-255)"""
        r, g, b = self.color.rgb
        return (int(r * 255), int(g * 255), int(b * 255))

    @property
    def color_rgb_float(self) -> Tuple[float, float, float]:
        """Convert RGB integers back to floats for cv2.fillPoly"""
        r, g, b = self.color_rgb_int
        return (float(r), float(g), float(b))


class Circle(Shape):
    def __init__(self, c_x: int, c_y: int, radius: int, distance: float, color: Color):
        super().__init__(c_x, c_y, distance, color)
        self.validate_radius(radius)
        self.radius = radius

    def validate_radius(self, radius: int) -> None:
        if not isinstance(radius, (int, np.integer)) or radius <= 0:
            raise ValueError("Radius must be a positive integer")

    def draw(self, image: np.ndarray) -> np.ndarray:
        scaled_radius = int(self.radius / self.distance)
        cv2.circle(
            image,
            center=(self.c_x, self.c_y),
            radius=scaled_radius,
            color=self.color_rgb_int,
            thickness=-1,
        )
        return image

    def get_caption(self) -> str:
        return self._get_base_caption()


class Rectangle(Shape):
    def __init__(
        self, c_x: int, c_y: int, length: int, width: int, distance: float, color: Color
    ):
        super().__init__(c_x, c_y, distance, color)
        self.validate_dimensions(length, width)
        self.length = length
        self.width = width

    def validate_dimensions(self, length: int, width: int) -> None:
        if not isinstance(length, (int, np.integer)) or length <= 0:
            raise ValueError("Length must be a positive integer")
        if not isinstance(width, (int, np.integer)) or width <= 0:
            raise ValueError("Width must be a positive integer")

    def draw(self, image: np.ndarray) -> np.ndarray:
        scaled_length = int(self.length / self.distance)
        scaled_width = int(self.width / self.distance)

        x1 = self.c_x - scaled_width // 2
        y1 = self.c_y - scaled_length // 2
        x2 = x1 + scaled_width
        y2 = y1 + scaled_length

        # Create rectangle points in counter-clockwise order
        points = [
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2],  # bottom-left
        ]
        pts = np.array([points], dtype=np.int32)  # shape: (1, 4, 2)

        cv2.fillPoly(
            image,
            pts=[pts],
            color=self.color_rgb_int,
        )
        return image

    def get_caption(self) -> str:
        return self._get_base_caption()


class Triangle(Shape):
    def __init__(
        self,
        c_x: int,
        c_y: int,
        angles: Tuple[float, float, float],
        size: int,
        distance: float,
        color: Color,
    ):
        super().__init__(c_x, c_y, distance, color)
        self.validate_angles(angles)
        self.validate_size(size)
        self.angles = angles
        self.size = size

    def validate_angles(self, angles: Tuple[float, float, float]) -> None:
        if len(angles) != 3:
            raise ValueError("Must provide exactly 3 angles")
        if not np.isclose(sum(angles), 180):
            raise ValueError("Angles must sum to 180 degrees")
        if any(angle <= 0 for angle in angles):
            raise ValueError("All angles must be positive")

    def validate_size(self, size: int) -> None:
        if not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError("Size must be a positive integer")

    def draw(self, image: np.ndarray) -> np.ndarray:
        scaled_size = int(self.size / self.distance)

        # Convert angles to radians
        angles_rad = np.array(self.angles) * np.pi / 180

        # Calculate vertices as 2D points
        points = []
        current_angle = 0
        for i in range(3):
            x = self.c_x + scaled_size * np.cos(current_angle)
            y = self.c_y + scaled_size * np.sin(current_angle)
            points.append([int(x), int(y)])
            current_angle += angles_rad[i]

        # Format points as required by cv2.fillPoly
        pts = np.array([points], dtype=np.int32)  # shape: (1, 3, 2)

        cv2.fillPoly(
            image,
            pts=[pts],
            color=self.color_rgb_int,
        )
        return image

    def get_caption(self) -> str:
        return self._get_base_caption()
