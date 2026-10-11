"""Live gesture -> emotion recognition from the webcam. Press q to quit."""
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from features import CLIP_SECONDS, SLOT, clip_features, display_name, frame_to_vector, resample

CONFIDENCE = 0.6      
PREDICT_EVERY = 0.1  
SMOOTHING = 5         

bundle = joblib.load("model.joblib")
model, classes = bundle["model"], bundle["classes"]
model.n_jobs = 1 

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
buffer = deque()              
votes = deque(maxlen=SMOOTHING)  
last_predict = 0.0
shown, shown_conf = None, 0.0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Camera frame not available")
        break
    frame = cv2.flip(frame, 1)  
    h, w = frame.shape[:2]
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    now = time.time()
    buffer.append((now, frame_to_vector(results, w / h)))
    while buffer and now - buffer[0][0] > CLIP_SECONDS:
        buffer.popleft()

    window_full = len(buffer) > 5 and now - buffer[0][0] >= CLIP_SECONDS * 0.9
    if window_full and now - last_predict >= PREDICT_EVERY:
        last_predict = now
        times = [t for t, _ in buffer]
        window = np.array([v for _, v in buffer])
        seen = np.mean((window[:, 0] > 0.5) | (window[:, SLOT] > 0.5))
        if seen < 0.5:
            votes.append((None, 0.0))
        else:
            probs = model.predict_proba([clip_features(resample(times, window))])[0]
            best = int(np.argmax(probs))
            label = int(model.classes_[best])
            if probs[best] >= CONFIDENCE and classes[label] != "None":
                votes.append((label, float(probs[best])))
            else:
                votes.append((None, 0.0))

        top, n = Counter(l for l, _ in votes).most_common(1)[0]
        shown = top if (top is not None and n > len(votes) / 2) else None
        shown_conf = float(np.mean([c for l, c in votes if l == shown])) if shown is not None else 0.0

    if shown is not None:
        text, color = "%s (%d%%)" % (display_name(classes[shown]), round(100 * shown_conf)), (0, 255, 0)
    else:
        text, color = "...", (200, 200, 200)
    cv2.putText(frame, "Detected: " + text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.imshow("eMotion", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()