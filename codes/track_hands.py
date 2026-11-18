import cv2  # camera/display
import mediapipe as mp  # hand tracking
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
import numpy as np  # unused yet
from pathlib import Path  # model path
import time  # timestamp

# CONSTANTS
# per-finger (B, G, R) direction: index f = 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky # color = direction * brightness
FINGER_DIRS = ((0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1))
PINCH_THRESHOLD = 20


# ============================================================
# F-1. create_landmarker # only once before the video loop
def create_landmarker(model_path):  # Input: model_path(Path) # Output: HandLandmarker(VIDEO  mode, max 2 hands)
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))  # which model file to load
    options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                                                    num_hands=2)  # running mode (VIDEO) + max number of hands
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options=options)  # create the landmarker object
    return landmarker


# F-2. detect_hands
def detect_hands(landmarker,
                 frame):  # Input: landmarekr, frame (H, W, 3) uint8 BGR # Output: result (HandLandmarkerResult)
    # F-2-1. Preprocess:: BGR -> MediaPipe input format
    # frame(H, W, 3, BGR) -> rgb(H, W, 3) using cv2.cvtColor(..., cv2.COLORBGR2RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # wrap with mp.Image(image_foramt = mp.ImageFormat.SRGB, data = rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # Make integer timestep:
    timestamp_ms = int(time.monotonic() * 1000)

    # F-2-2. Detect
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    # count =len(result.hand_landmarks) # number of hands (0 or 1 or 2) # each element is a list of length 21
    return result  # result.hand_landmarks[i]: 21 landmarks of hand i # result.handedness[i][0] : its label


# F-3. get_labels
def get_labels(result):  # Input: result / Output: labels (list[str]), e.g. ["Left", "Right"], [] if no hand
    labels = []  # one label per detected hand (same order as result.hand_landmarks)
    for hand in result.handedness:  # hand: list[Category] of one hand
        labels.append(hand[0].category_name)  # hand[0] = top1 Category, caegory_name is "Left" or "Right"
    return labels  # empty list when no hand is detected


# F-4. to_pixel
def to_pixel(lm, W, H):  # lm: NormalizedLandmark # lm.x, lm.y: 0~1 value
    x_px = int(lm.x * W)
    y_px = int(lm.y * H)
    # return (x_px, y_px): integer tuple
    return x_px, y_px


# F-5. draw_joint
def draw_joint(frame, hand, color,
               joint_index):  # frame: (H, W, 3) BGR data # hand: 21 landmarks list # color: (B, G, R) tuple
    H, W = frame.shape[:2]  # frame.shape is (H, W, 3) -> H first
    center = to_pixel(hand[joint_index], W, H)  # pixel position (x, y) of this joint
    cv2.circle(frame, center, 10, color, -1)  # filled dot, radius 10
    cv2.putText(frame, str(int(joint_index)), (center[0] + 12, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                1)  # joint number, 12 px right of the dot


# F-6. draw_hand
def draw_hand(frame, hand,
              label):  # Input: frame, hand (21 landmarks), label ("Left"/"Right") # Output: None (draws on frame)
    # 1) draw all 21 joints, color by finger
    for joint in HandLandmark:
        if joint == 0:  # wrist
            color = (0, 0, 0)
        else:
            f = (joint - 1) // 4  # finger index: 0 thumb ... 4 pinky
            k = (joint - 1) % 4  # position in finger: 0 base ... 3 tip
            brightness = min(64 * (k + 1), 255)  # 64, 128, 192, 255 (base -> tip)
            d = FINGER_DIRS[f]  # direction of this finger, e.g. (0, 1, 1)
            color = tuple(c * brightness for c in d)  # component-wise: direction * brightness -> (B, G, R)
        draw_joint(frame, hand, color, joint)
    # 2) hand label text near the wrist
    H, W = frame.shape[:2]
    wrist_coordinate = to_pixel(hand[HandLandmark.WRIST], W, H)
    hand_label_coordinate = (wrist_coordinate[0] - 15, wrist_coordinate[1] + 40)
    cv2.putText(frame, label, hand_label_coordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


# F-7. pinch detection
def tip_distance(hand, W, H):
    thumb_tip = to_pixel(hand[HandLandmark.THUMB_TIP], W, H)  # pixel (x, y) of thumb tip (4)
    index_tip = to_pixel(hand[HandLandmark.INDEX_FINGER_TIP], W, H)  # pixel (x, y) of index tip (8)
    # Euclidean distance: sqrt(dx^2 + dy^2) in pixels
    dx = index_tip[0] - thumb_tip[0]
    dy = index_tip[1] - thumb_tip[1]
    distance = (dx ** 2 + dy ** 2) ** 0.5
    return distance


# ============================================================
# MAIN
def main():
    # ============================================================
    # 0. CONSTANTS / STATE
    # ============================================================
    # 1. ENV CHECK
    print(mp.__version__)  # mediapipe version check

    # ============================================================
    # 2. Model PATH
    model_path = Path(
        __file__).parent.parent / "models" / "hand_landmarker.task"  # codes/ folder as ref -> find .task file
    print(f"Model Path Check: {model_path.exists()}")  # check whether such model_path file exists

    # ============================================================
    # 3. Hand Landmarker setup # only once before the video loop
    landmarker = create_landmarker(model_path)

    # ============================================================
    # 4. Camera setup
    cap = cv2.VideoCapture(0)  # open cap (video capturer)
    if not cap.isOpened():  # if cap is not opened,
        raise SystemExit("cap not opened")  # print error message and exit the program

    # ============================================================
    # 5. Main loop : repeat per each frame
    try:  # camera and landmarker are released even if an error occurs
        while True:  # 5-1. Frame read
            ok, frame = cap.read()  # To read data at the same frame
            if not ok:  # if fail to read,
                break  # break the loop

            # 5-3. Detect
            result = detect_hands(landmarker, frame)
            # 5-4. Labels
            labels = get_labels(result)

            # 5-5. Draw
            for i, hand in enumerate(result.hand_landmarks):
                draw_hand(frame, hand, labels[i])

            # 5-7. Pinch Detection
            H, W = frame.shape[:2]
            for i, hand in enumerate(result.hand_landmarks):
                d = tip_distance(hand, W, H)
                if d <= PINCH_THRESHOLD:
                    print(f"{labels[i]} is PINCHING")

            # 5-8. DISPLAY
            cv2.imshow('frame', frame)

            # 5-9. EXIT with exit key 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"q key pressed: program shutting down")
                break

    finally:
        # ============================================================
        # 6. CLEANUP
        #    - cap.release(), landmarker.close(), destroyAllWindows()
        cap.release()
        landmarker.close()  # call landmarker.close() after loop is finished
        cv2.destroyAllWindows()


# ============================================================

if __name__ == "__main__":  # run main() only when this file is executed directly
    main()
