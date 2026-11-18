import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import layout as L

DPI = 300
PX_PER_MM = DPI / 25.4
DRAW_GUIDES = True  # faint outlines of the donut / strum zone (set False to print markers only)
OUT_DIR = Path(__file__).parent.parent / "print"


def px(mm):  # mm -> pixels (int)
    return int(round(mm * PX_PER_MM))


def main():
    sheet = np.full((px(L.SHEET_H), px(L.SHEET_W)), 255, np.uint8)  # white A4 landscape

    # 1) markers
    dictionary = cv2.aruco.getPredefinedDictionary(L.ARUCO_DICT)
    side = px(L.MARKER)
    for marker_id, (cx, cy) in L.MARKERS.items():
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, side)  # (side, side) uint8, black border included
        x0, y0 = px(cx - L.MARKER / 2), px(cy - L.MARKER / 2)
        sheet[y0:y0 + side, x0:x0 + side] = img

    # 2) optional guides (gray, thin) + a 50 mm bar to check the printed scale
    if DRAW_GUIDES:
        gray, t = 150, 3
        c = (px(L.DONUT_C[0]), px(L.DONUT_C[1]))
        cv2.circle(sheet, c, px(L.OUTER_R), gray, t)
        cv2.circle(sheet, c, px(L.OUTER_R * L.INNER_RATIO), gray, t)
        x0, y0 = px(L.RECT_C[0] - L.RECT_W / 2), px(L.RECT_C[1] - L.RECT_H / 2)
        x1, y1 = px(L.RECT_C[0] + L.RECT_W / 2), px(L.RECT_C[1] + L.RECT_H / 2)
        cv2.rectangle(sheet, (x0, y0), (x1, y1), gray, t)
    cv2.line(sheet, (px(15), px(202)), (px(65), px(202)), 0, 3)
    cv2.putText(sheet, "50 mm", (px(68), px(203.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

    # 3) save: PNG for checking, PDF with the exact A4 size for printing
    OUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "markers_a4.png"), sheet)
    Image.fromarray(sheet).save(OUT_DIR / "markers_a4.pdf", resolution=DPI)
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
