import cv2
import numpy as np


def get_template(frame, bbox):
    x, y, w, h = map(int, bbox)
    template = frame[y:y+h, x:x+w]
    return template


def get_search_region(frame, bbox, scale=2.0):
    x, y, w, h = map(int, bbox)

    cx = x + w // 2
    cy = y + h // 2

    new_w = int(w * scale)
    new_h = int(h * scale)

    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)

    x2 = min(frame.shape[1], x1 + new_w)
    y2 = min(frame.shape[0], y1 + new_h)

    search = frame[y1:y2, x1:x2]
    return search


def preprocess_image(img, size=(224, 224)):
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img