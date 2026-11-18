import math
import cv2

# Physical layout of the printed sheet. All lengths in millimetres.
# Sheet coordinates: origin = top-left corner of the sheet, x to the right, y downward (same axes as an image).
# The marker generator and (later) the homography code both read this file, so they always agree.

ARUCO_DICT = cv2.aruco.DICT_4X4_50
SHEET_W, SHEET_H = 297, 210  # A4 landscape
MARKER = 18  # marker side (outer edge of the black border)
CLEARANCE = 5  # white gap between a zone and the nearest marker edge (must stay >= 1 marker bit = MARKER/6 = 3)

# Donut zone (blank circle on paper; the safe zone and the pie pieces are drawn on screen)
DONUT_C = (92, 105)  # center
OUTER_R = 66
INNER_RATIO = 0.4  # safe zone radius = INNER_RATIO * OUTER_R
N_DONUT_MARKERS = 6

# Strum zone
RECT_C = (237, 105)  # center
RECT_W, RECT_H = 50, 130


def donut_marker_centers():  # 6 markers on a ring around the donut, starting at 12 o'clock, clockwise
    h = MARKER / 2
    pts = []
    for k in range(N_DONUT_MARKERS):
        a = math.radians(90 - 360 * k / N_DONUT_MARKERS)
        c, s = math.cos(a), math.sin(a)
        d = OUTER_R + CLEARANCE + h * (abs(c) + abs(s))  # axis-aligned square: keep the same clearance in every direction
        pts.append((DONUT_C[0] + d * c, DONUT_C[1] - d * s))
    return pts


def rect_marker_centers():  # 4 corners + 2 long-side midpoints
    h = MARKER / 2
    dx = RECT_W / 2 + CLEARANCE + h
    dy = RECT_H / 2 + CLEARANCE + h
    offsets = [(-dx, -dy), (dx, -dy), (dx, 0), (dx, dy), (-dx, dy), (-dx, 0)]
    return [(RECT_C[0] + ox, RECT_C[1] + oy) for ox, oy in offsets]


# marker id -> center (mm). Donut ids 0..5, strum zone ids 6..11
MARKERS = {i: c for i, c in enumerate(donut_marker_centers() + rect_marker_centers())}
DONUT_IDS = range(0, N_DONUT_MARKERS)
RECT_IDS = range(N_DONUT_MARKERS, N_DONUT_MARKERS + 6)


def marker_corners(marker_id):  # 4 corners (mm) in ArUco order: top-left, top-right, bottom-right, bottom-left
    cx, cy = MARKERS[marker_id]
    h = MARKER / 2
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
