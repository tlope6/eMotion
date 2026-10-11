"""Record gesture clips (short sequences of hand landmarks) for training."""
import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from features import CLASSES, CLIP_SECONDS, SLOT, frame_to_vector, resample

COUNTDOWN = 3.0
DATA_DIR = "data"

parser = argparse.ArgumentParser()
parser.add_argument("--person", default="p1", help="who is performing (lets training test on unseen people)")
args = parser.parse_args()
person = args.person.replace("_", "-")

for c in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)
counts = {c: len([f for f in os.listdir(os.path.join(DATA_DIR, c)) if f.endswith(".npy")]) for c in CLASSES}

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
current = 0
state = "idle" 
phase_start = 0.0
times, vecs = [], []
last_saved = None  
message = "Pick a gesture, press SPACE, then perform it when recording starts"

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Camera frame not available")
        break
    frame = cv2.flip(frame, 1)  
    h, w = frame.shape[:2]
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    vec = frame_to_vector(results, w / h)
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    now = time.time()
    banner = ""
    if state == "countdown":
        remaining = COUNTDOWN - (now - phase_start)
        if remaining <= 0:
            state, phase_start, times, vecs = "recording", now, [], []
        else:
            banner = "Get ready... %d" % (int(remaining) + 1)
    if state == "recording":
        times.append(now)
        vecs.append(vec)
        banner = "RECORDING - do the gesture now"
        if now - phase_start >= CLIP_SECONDS:
            state = "idle"
            name = CLASSES[current]
            seen = np.mean([(v[0] > 0.5) or (v[SLOT] > 0.5) for v in vecs])
            if name != "None" and seen < 0.7:
                message = "Hand lost in too many frames - clip discarded, try again"
            else:
                path = os.path.join(DATA_DIR, name, "%s_%d.npy" % (person, int(now * 1000)))
                np.save(path, resample(times, vecs))
                last_saved = (name, path)
                counts[name] += 1
                message = "Saved clip for %s" % name

    cv2.putText(frame, banner or message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if state == "idle" else (0, 0, 255), 2)
    for i, c in enumerate(CLASSES):
        color = (0, 255, 255) if i == current else (255, 255, 255)
        cv2.putText(frame, "%d %s: %d" % (i, c, counts[c]), (10, 70 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    cv2.imshow("Record clips", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if state != "idle":
        continue
    if key == ord(" "):
        state, phase_start = "countdown", now
    elif key == ord("n"):
        current = (current + 1) % len(CLASSES)
    elif key == ord("p"):
        current = (current - 1) % len(CLASSES)
    elif ord("0") <= key <= ord("9") and key - ord("0") < len(CLASSES):
        current = key - ord("0")
    elif key == ord("u") and last_saved and os.path.exists(last_saved[1]):
        os.remove(last_saved[1])
        counts[last_saved[0]] -= 1
        message = "Removed last clip of %s" % last_saved[0]
        last_saved = None

cap.release()
cv2.destroyAllWindows()