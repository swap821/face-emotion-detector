"""
utils.py — Helpers for Face Emotion Detection
"""

import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image


def decode_base64_image(base64_string):
    """Decode base64 string to numpy array."""
    img_data = base64.b64decode(base64_string.split(',')[1] if ',' in base64_string else base64_string)
    img = Image.open(BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def encode_image_base64(frame):
    """Encode numpy array to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)