import cv2
import numpy as np
import sys

MIN_LOGO_MATCHES = 25
MIN_AREA_RATIO   = 0.05
MAX_AREA_RATIO   = 0.95
APPROX_EPSILON   = 0.02

def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def detect_poster_corners(frame):
    h, w = frame.shape[:2]
    area = w * h

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0

    for cnt in contours:
        a = cv2.contourArea(cnt)

        if not (area * MIN_AREA_RATIO < a < area * MAX_AREA_RATIO):
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPSILON * peri, True)

        if len(approx) == 4 and a > best_area:
            best_area = a
            best = order_corners(approx)

    return best

def detect_poster(image_path):

    frame = cv2.imread(image_path)

    if frame is None:
        return None

    corners = detect_poster_corners(frame)

    return corners