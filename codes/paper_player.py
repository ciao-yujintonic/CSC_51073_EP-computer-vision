import argparse
import time
import cv2
import numpy as np
import sounddevice as sd
from pathlib import Path
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

import layout as L
from detect_markers import create_detector, detect, open_camera
from paper_markers import ZoneTracker
from pie_player import (SAMPLE_RATE, CHORDS, N_SECTORS, make_chord, sector_at, StrumDetector)
from track_hands import create_landmarker, detect_hands, draw_hand, to_pixel

INK, GREEN = (40, 40, 40), (0, 200, 0)  # dark overlay lines: they must be visible on white paper


# ---------- drawing: shapes are defined in paper mm, projected to the image with the zone's homography ----------
def arc(center, r, deg0, deg1, n=24):  # points on a circle, angle = degrees clockwise from 12 o'clock / (n, 2) mm
    th = np.radians(np.linspace(deg0, deg1, n))
    return np.stack([center[0] + r * np.sin(th), center[1] - r * np.cos(th)], axis=1)


def poly(zone, pts_mm):  # paper mm -> int32 pixel polygon for cv2
    return zone.to_image(pts_mm).astype(np.int32)


def blend_fill(frame, polygon, color, alpha=0.4):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_donut(frame, zone, active):  # active: highlighted sector (-1 = none)
    c, R = L.DONUT_C, L.OUTER_R
    r = R * L.INNER_RATIO
    step = 360 / N_SECTORS
    if active >= 0:
        piece = np.vstack([arc(c, R, active * step, (active + 1) * step), arc(c, r, (active + 1) * step, active * step)])
        blend_fill(frame, poly(zone, piece), GREEN)
    for rad in (R, r):
        cv2.polylines(frame, [poly(zone, arc(c, rad, 0, 360, 72))], True, INK, 2)
    for i in range(N_SECTORS):
        if N_SECTORS > 1:
            p0, p1 = poly(zone, np.vstack([arc(c, r, i * step, i * step, 1), arc(c, R, i * step, i * step, 1)]))
            cv2.line(frame, tuple(p0), tuple(p1), INK, 2)
        mid = poly(zone, arc(c, (R + r) / 2, (i + 0.5) * step, (i + 0.5) * step, 1))[0]
        cv2.putText(frame, CHORDS[i][0], (mid[0] - 10, mid[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, INK, 2)


def rect_corners():  # strum zone corners in paper mm: TL, TR, BR, BL
    x0, y0 = L.RECT_C[0] - L.RECT_W / 2, L.RECT_C[1] - L.RECT_H / 2
    x1, y1 = L.RECT_C[0] + L.RECT_W / 2, L.RECT_C[1] + L.RECT_H / 2
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], np.float32)


def draw_strum(frame, zone, active, volume):  # active: a hand is inside / volume: last strum volume (0..1)
    corners = rect_corners()
    if active:
        blend_fill(frame, poly(zone, corners), GREEN, 0.2)
    cv2.polylines(frame, [poly(zone, corners)], True, INK, 2)
    tl = poly(zone, corners[:1])[0]
    cv2.putText(frame, "STRUM", (tl[0] + 10, tl[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 2)
    (x0, y0), (x1, y1) = corners[0], corners[2]
    top = y1 - 2 - volume * (L.RECT_H - 4)  # volume bar along the right edge, inside the rectangle
    bar = np.array([(x1 - 6, y1 - 2), (x1 - 2, y1 - 2), (x1 - 2, top), (x1 - 6, top)], np.float32)
    cv2.fillPoly(frame, [poly(zone, bar)], GREEN)


# ---------- hit tests in paper mm ----------
def donut_sector(hands, W, H, zone):  # sector under any index fingertip, -1 if none
    inner = L.OUTER_R * L.INNER_RATIO
    for hand in hands:
        p = zone.to_paper(to_pixel(hand[HandLandmark.INDEX_FINGER_TIP], W, H))
        s = sector_at(p, L.DONUT_C, inner, L.OUTER_R, N_SECTORS)
        if s >= 0:
            return s
    return -1


def strum_y(hands, W, H, zone):  # fingertip y inside the strum zone as 0 (top) .. 1 (bottom) of the rectangle, else None
    for hand in hands:
        x, y = zone.to_paper(to_pixel(hand[HandLandmark.INDEX_FINGER_TIP], W, H))
        t = (y - (L.RECT_C[1] - L.RECT_H / 2)) / L.RECT_H
        if abs(x - L.RECT_C[0]) <= L.RECT_W / 2 and 0 <= t <= 1:
            return float(t)  # in rect-heights, so StrumDetector speeds are rect-heights / s
    return None


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    landmarker = create_landmarker(Path(__file__).parent.parent / "models" / "hand_landmarker.task")
    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        raise SystemExit(f"camera {args.camera} not opened (try: python detect_markers.py --probe)")
    cam_index = args.camera

    detector = create_detector()
    donut = ZoneTracker(L.DONUT_IDS, L.DONUT_C)
    strum_zone = ZoneTracker(L.RECT_IDS, L.RECT_C)
    chord_waves = [make_chord(freqs) for _, freqs in CHORDS]
    strum = StrumDetector()
    last_volume = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]  # NOT mirrored: ArUco markers are not detected in a flipped image

            found = detect(frame, detector)
            donut.update(found)
            strum_zone.update(found)
            result = detect_hands(landmarker, frame)
            hands = result.hand_landmarks

            hover, sy = -1, None
            if donut.valid:
                hover = donut_sector(hands, W, H, donut)
            if strum_zone.valid:
                sy = strum_y(hands, W, H, strum_zone)
            volume = strum.update(sy, time.monotonic())
            if volume is not None and hover >= 0:  # a stroke while a chord is held -> play it at that volume
                sd.play(chord_waves[hover] * volume, SAMPLE_RATE)
                last_volume = volume

            if donut.valid:
                draw_donut(frame, donut, hover)
            if strum_zone.valid:
                draw_strum(frame, strum_zone, sy is not None, last_volume)
            for hand in hands:
                draw_hand(frame, hand, "")

            # status: tracking state per zone
            for k, (name, z) in enumerate((("donut", donut), ("strum", strum_zone))):
                state = "tracking" if z.lost == 0 else ("held" if z.valid else "LOST")
                color = GREEN if z.lost == 0 else ((0, 200, 255) if z.valid else (0, 0, 255))
                cv2.putText(frame, f"{name}: {state} ({z.n_used} markers)", (10, 30 + 32 * k),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"camera {cam_index}  (keys: 1 -> cam 0, 2 -> cam 1, q: quit)",
                        (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("paper player", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if ord('1') <= key <= ord('9'):  # number key k -> camera index k - 1
                new_index = key - ord('1')
                if new_index != cam_index:
                    new_cap = open_camera(new_index, args.width, args.height)
                    if new_cap is None:
                        print(f"camera {new_index} not available, staying on camera {cam_index}")
                    else:
                        cap.release()
                        cap, cam_index = new_cap, new_index
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
