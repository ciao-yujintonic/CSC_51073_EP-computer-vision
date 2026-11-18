import cv2
import math  # atan2, hypot
import time  # stroke speed (dt)
import numpy as np  # chord synthesis
import sounddevice as sd  # audio playback
from pathlib import Path
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

from track_hands import create_landmarker, detect_hands, get_labels, draw_hand, to_pixel

# CONSTANTS
SAMPLE_RATE = 44100
INNER_RATIO = 0.4  # safe zone radius = INNER_RATIO * outer radius (hovering inside it plays nothing)
# one (name, frequencies in Hz) per sector, clockwise from 12 o'clock
CHORDS = [("C", (261.63, 329.63, 392.00)),  # C major: C4 E4 G4
          ("D", (293.66, 369.99, 440.00)),  # D major: D4 F#4 A4
          ("E", (329.63, 415.30, 493.88))]  # E major: E4 G#4 B4
N_SECTORS = len(CHORDS)  # pie pieces (= number of chords of the song)
CIRCLE_CX = 0.28  # donut center x as a fraction of W (moved to the left)
STRUM_RECT = (0.62, 0.15, 0.95, 0.85)  # strum zone (x0, y0, x1, y1) as fractions of (W, H)
V_MIN = 0.5  # stroke speed (frame-heights / s) needed to trigger a strum
V_MAX = 2.5  # speed at which the volume is maximal
VOL_MIN = 0.15  # volume of the softest strum


# P-1. make_chord  # Input: freqs (tuple of Hz), duration (s) / Output: float32 wave (N,) in [-1, 1]
def make_chord(freqs, duration=1.5):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE  # (N,) time axis in seconds
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs)  # sum of sine waves, one per note
    wave = wave * np.exp(-2.0 * t)  # decaying envelope (avoids a constant drone)
    attack = np.minimum(t / 0.01, 1.0)  # 10 ms fade-in (avoids a click)
    wave = wave * attack
    return (0.8 * wave / np.max(np.abs(wave))).astype(np.float32)  # normalize


# P-2. sector_at  # Input: point (x, y) px, center (x, y) px, inner/outer radius px, n / Output: sector 0..n-1, or -1
def sector_at(point, center, inner, outer, n):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    r = math.hypot(dx, dy)
    if r < inner or r > outer:  # in the safe zone (hole) or outside the donut
        return -1
    angle = math.atan2(dx, -dy) % (2 * math.pi)  # clockwise angle from 12 o'clock (image y points down)
    return int(angle / (2 * math.pi / n))  # which of the n equal pie pieces


# P-3. draw_pie  # draws the donut, n sectors with names, highlights sector `active` (-1 = none)
def draw_pie(frame, center, inner, outer, n, active):
    step = 360 / n
    if active >= 0:  # highlight only the donut part of the active sector (blended)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.ellipse(mask, center, (outer, outer), -90, active * step, (active + 1) * step, 255, -1)
        cv2.circle(mask, center, inner, 0, -1)  # cut out the safe zone
        m = mask > 0
        frame[m] = (0.6 * frame[m] + 0.4 * np.array((0, 200, 0))).astype(np.uint8)
    cv2.circle(frame, center, outer, (255, 255, 255), 2)  # outer outline
    cv2.circle(frame, center, inner, (255, 255, 255), 2)  # safe zone outline
    for i in range(n):
        a = math.radians(i * step - 90)  # separator angle (0 deg = 3 o'clock, so shift -90 to start at 12)
        p0 = (int(center[0] + inner * math.cos(a)), int(center[1] + inner * math.sin(a)))
        p1 = (int(center[0] + outer * math.cos(a)), int(center[1] + outer * math.sin(a)))
        if n > 1:
            cv2.line(frame, p0, p1, (255, 255, 255), 2)
        mid = math.radians((i + 0.5) * step - 90)  # chord name at the middle of the piece
        rm = (inner + outer) / 2
        pos = (int(center[0] + rm * math.cos(mid)) - 10, int(center[1] + rm * math.sin(mid)) + 10)
        cv2.putText(frame, CHORDS[i][0], pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


# P-4. hovered_sector  # Input: hands (list of 21 landmarks), W, H, center, inner, outer, n / Output: sector index or -1
def hovered_sector(hands, W, H, center, inner, outer, n):
    for hand in hands:
        tip = to_pixel(hand[HandLandmark.INDEX_FINGER_TIP], W, H)
        s = sector_at(tip, center, inner, outer, n)
        if s >= 0:
            return s
    return -1


# P-5. strum zone helpers
def rect_px(W, H):  # fractions -> pixel rectangle (x0, y0, x1, y1)
    x0, y0, x1, y1 = STRUM_RECT
    return int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)


def in_rect(point, rect):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def draw_strum_zone(frame, rect, active, volume):  # active: a hand is inside / volume: last strum volume (0..1)
    if active:  # light highlight while a hand is in the zone
        overlay = frame.copy()
        cv2.rectangle(overlay, rect[:2], rect[2:], (0, 200, 0), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.rectangle(frame, rect[:2], rect[2:], (255, 255, 255), 2)
    cv2.putText(frame, "STRUM", (rect[0] + 10, rect[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    bar_h = int((rect[3] - rect[1] - 20) * volume)  # volume bar on the right edge of the zone
    cv2.rectangle(frame, (rect[2] - 20, rect[3] - 10 - bar_h), (rect[2] - 8, rect[3] - 10), (0, 200, 0), -1)


# P-6. StrumDetector: vertical speed of the fingertip -> volume
class StrumDetector:
    def __init__(self):
        self.prev_y = None  # previous normalized fingertip y (0..1), None if no hand in the zone
        self.prev_t = None  # previous timestamp (s)
        self.armed = True  # True when a new stroke may trigger (re-armed once the hand slows down)

    # Input: y (normalized tip y or None), t (s) / Output: volume in [VOL_MIN, 1] on a new stroke, else None
    def update(self, y, t):
        if y is None:  # hand left the zone
            self.prev_y = None
            self.armed = True
            return None
        volume = None
        if self.prev_y is not None and t > self.prev_t:
            speed = abs(y - self.prev_y) / (t - self.prev_t)  # frame-heights per second, up or down
            if speed < V_MIN:
                self.armed = True
            elif self.armed:  # fast enough and not already triggered by this stroke
                level = min((speed - V_MIN) / (V_MAX - V_MIN), 1.0)
                volume = VOL_MIN + (1 - VOL_MIN) * level
                self.armed = False
        self.prev_y, self.prev_t = y, t
        return volume


# ============================================================
def main():
    model_path = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
    landmarker = create_landmarker(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("cap not opened")

    chord_waves = [make_chord(freqs) for _, freqs in CHORDS]  # pre-synthesize once
    strum = StrumDetector()
    last_volume = 0.0  # volume of the latest strum (for display)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror (selfie view) BEFORE detection, so labels/coordinates match the display
            result = detect_hands(landmarker, frame)
            labels = get_labels(result)
            H, W = frame.shape[:2]

            center = (int(W * CIRCLE_CX), H // 2)
            outer = min(H, W) // 3
            inner = int(outer * INNER_RATIO)
            hover = hovered_sector(result.hand_landmarks, W, H, center, inner, outer, N_SECTORS)

            # strumming hand: index fingertip inside the rectangle -> its vertical speed
            rect = rect_px(W, H)
            strum_y = None
            for hand in result.hand_landmarks:
                tip = hand[HandLandmark.INDEX_FINGER_TIP]
                if in_rect(to_pixel(tip, W, H), rect):
                    strum_y = tip.y
                    break
            volume = strum.update(strum_y, time.monotonic())
            if volume is not None and hover >= 0:  # a stroke while a chord is held -> play it at that volume
                sd.play(chord_waves[hover] * volume, SAMPLE_RATE)
                last_volume = volume

            draw_pie(frame, center, inner, outer, N_SECTORS, hover)
            draw_strum_zone(frame, rect, strum_y is not None, last_volume)
            for i, hand in enumerate(result.hand_landmarks):
                draw_hand(frame, hand, labels[i])

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
