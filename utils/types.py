"""
Centralize cross-module type aliases.

This module defines canonical aliases for image inputs, array payloads, camera
types, and geometric primitives used throughout the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from PIL import Image

# Image input types: can be path, PIL Image, numpy array, or bytes
ImageInput = str | Path | Image.Image | npt.NDArray[np.uint8] | bytes

# Numpy array representing an image: shape (H, W, 3) or (H, W), dtype uint8
ImageArray = npt.NDArray[np.uint8]

# BioCLIP embedding: shape (512,), dtype float32, L2-normalized
EmbeddingArray = npt.NDArray[np.float32]

# Camera type literal
CameraType = Literal["reconyx", "boly", "unknown"]

# Bounding box format: (x1, y1, x2, y2) in pixel coordinates
BBox = tuple[int, int, int, int]

# Timestamp parsing order: maps regex groups to (year, month, day, hour, minute, second)
TimestampGroupOrder = tuple[int, int, int, int, int, int]
