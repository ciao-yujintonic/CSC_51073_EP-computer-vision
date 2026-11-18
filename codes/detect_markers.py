import argparse
import time
import cv2
import numpy as np

import layout as L


def create_detector():  # ArUco detector for the dictionary used on the printed sheet
    dictionary = cv2.aruco.getPredefinedDictionary(L.ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # sub-pixel corners -> steadier homography later
    return cv2.aruco.ArucoDetector(dictionary, params)


# Input: frame (H, W, 3) BGR, detector / Output: {id: corners (4, 2) float32 px}, corners ordered TL, TR, BR, BL
def detect(frame, detector):
    corners, ids, _ = detector.detectMarkers(frame)  # corners: list of (1, 4, 2), ids: (N, 1) or None
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            found[int(i)] = c.reshape(4, 2)
    return found


def open_camera(index, width, height):  # returns an opened VideoCapture, or None if it cannot be opened
    c = cv2.VideoCapture(index)
    if not c.isOpened():
        c.release()
        return None
    c.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return c


# Input: frame (H, W, 3) BGR, detector / Output: annotated frame (drawn in place), {id: corners (4, 2)}
def detect_and_draw(frame, detector):
    found = detect(frame, detector)
    if found:
        cv2.aruco.drawDetectedMarkers(frame, [c.reshape(1, 4, 2) for c in found.values()],
                                      np.array(list(found), dtype=np.int32).reshape(-1, 1))

    # per-zone count + average marker side in pixels (aim for >= 40 px for stable detection)
    lines = []
    for name, zone_ids in (("donut", L.DONUT_IDS), ("strum", L.RECT_IDS)):
        n = sum(1 for i in zone_ids if i in found)
        lines.append((f"{name}: {n}/{len(zone_ids)}", n))
    sides = [np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).mean() for c in found.values()]
    side_text = f"marker side: {np.mean(sides):.0f}px" if sides else "marker side: -"

    y = 30
    for text, n in lines:
        color = (0, 200, 0) if n >= 3 else (0, 0, 255)  # green if enough markers for a homography (>= 3)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        y += 32
    cv2.putText(frame, side_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return found


def probe(max_index=5):  # find which index is the iPhone: print the resolution each camera gives
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        print(f"camera {idx}: " + (f"{frame.shape[1]}x{frame.shape[0]}" if ok else "not available"))
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="camera index (iPhone Continuity Camera is often 0 or 1)")
    ap.add_argument("--probe", action="store_true", help="list camera indices and their resolution, then exit")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    if args.probe:
        probe()
        return

    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        raise SystemExit(f"camera {args.camera} not opened (try --probe)")
    cam_index = args.camera

    detector = create_detector()
    prev_t = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detect_and_draw(frame, detector)  # NOT mirrored: ArUco markers are not detected in a flipped image

            now = time.monotonic()
            cv2.putText(frame, f"{1 / (now - prev_t):.0f} fps  {frame.shape[1]}x{frame.shape[0]}",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            prev_t = now
            cv2.putText(frame, f"camera {cam_index}  (keys: 1 -> cam 0, 2 -> cam 1, q: quit)",
                        (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("markers", frame)
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
