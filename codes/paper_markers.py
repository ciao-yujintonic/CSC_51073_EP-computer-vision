import cv2
import numpy as np

import layout as L

MIN_MARKERS = 3  # markers needed for a homography (3 markers = 12 corner points)
HOLD_FRAMES = 15  # keep using the last homography for this many frames after the markers are lost
ALPHA = 0.6  # smoothing weight of the newest corner positions (1 = no smoothing, lower = steadier but laggier)
RANSAC_PX = 3.0  # reprojection error (px) above which a corner is treated as an outlier


class ZoneTracker:
    """Tracks one zone (its markers) and converts between image pixels and paper millimetres."""

    def __init__(self, ids, zone_center):
        self.ids = list(ids)  # marker ids around this zone
        self.center = zone_center  # zone center in paper mm (must lie inside the detected markers' hull)
        self.smoothed = {}  # id -> (4, 2) smoothed image corners
        self.H = None  # paper (mm) -> image (px), None = zone not tracked
        self.H_inv = None  # image (px) -> paper (mm)
        self.lost = 0  # consecutive frames without a fresh homography
        self.n_used = 0  # markers used in the latest fresh homography

    @property
    def valid(self):
        return self.H is not None

    # Input: found = {id: corners (4, 2) px} / Output: True if a homography is available (fresh or held)
    def update(self, found):
        seen = [i for i in self.ids if i in found]
        for i in list(self.smoothed):  # forget markers that disappeared (a reappearing marker starts unsmoothed)
            if i not in seen:
                del self.smoothed[i]
        for i in seen:
            old = self.smoothed.get(i)
            self.smoothed[i] = found[i] if old is None else ALPHA * found[i] + (1 - ALPHA) * old

        if len(seen) >= MIN_MARKERS and self._encloses(seen):
            src = np.array([p for i in seen for p in L.marker_corners(i)], np.float32)  # (4k, 2) paper mm
            dst = np.vstack([self.smoothed[i] for i in seen]).astype(np.float32)  # (4k, 2) image px
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
            if H is not None:
                self.H, self.H_inv = H, np.linalg.inv(H)
                self.lost, self.n_used = 0, len(seen)
                return True

        self.lost += 1
        if self.lost > HOLD_FRAMES:  # held homography is too old
            self.H = self.H_inv = None
        return self.valid

    def _encloses(self, seen):  # detected markers must surround the zone center, else extrapolation is unreliable
        centers = np.array([L.MARKERS[i] for i in seen], np.float32)
        hull = cv2.convexHull(centers)
        return cv2.pointPolygonTest(hull, tuple(map(float, self.center)), False) >= 0

    def to_paper(self, pt_px):  # image px (x, y) -> paper mm (x, y)
        return cv2.perspectiveTransform(np.array([[pt_px]], np.float32), self.H_inv)[0, 0]

    def to_image(self, pts_mm):  # paper mm (N, 2) -> image px (N, 2)
        pts = np.asarray(pts_mm, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
