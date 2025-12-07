import cv2
import numpy as np

class BallDetector:
    def __init__(
        self,
        scale: float = 0.5,
        min_area: int = 15,
        max_area: int = 1500,
        min_radius: int = 3,
        max_radius: int = 35,
    ):
        self.scale = float(scale)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.min_radius = int(min_radius)
        self.max_radius = int(max_radius)

    def _color_mask(self, frame_bgr_small: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr_small, cv2.COLOR_BGR2HSV)

        # White-ish (bright): low saturation, high value
        lower_white = np.array([0, 0, 210], dtype=np.uint8)
        upper_white = np.array([180, 70, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lower_orange1 = np.array([0, 140, 100], dtype=np.uint8)
        upper_orange1 = np.array([25, 255, 255], dtype=np.uint8)
        lower_orange2 = np.array([160, 140, 100], dtype=np.uint8)
        upper_orange2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_orange1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
        mask_orange2 = cv2.inRange(hsv, lower_orange2, upper_orange2)

        mask = cv2.bitwise_or(mask_white, cv2.bitwise_or(mask_orange1, mask_orange2))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask

    def detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        s = self.scale
        frame_small = cv2.resize(frame_bgr, (int(w * s), int(h * s)))

        mask = self._color_mask(frame_small)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_center = None
        best_area = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < self.min_radius or radius > self.max_radius:
                continue
            if area > best_area:
                best_area = area
                best_center = (x, y)

        if best_center is None:
            return None

        cx_small, cy_small = best_center
        cx = int(cx_small / s)
        cy = int(cy_small / s)
        return (cx, cy)
