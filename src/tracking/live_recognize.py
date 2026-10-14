"""Live gesture -> emotion recognition from the webcam. Press q to quit."""
import time
from collections import Counter, deque

import os
import sys
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from features import CLIP_SECONDS, SLOT, clip_features, display_name, frame_to_vector, resample


# setting up the look for the layout of the camera
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)) #lets the file import src/app

from app import ui
from app.feedback import Feedback
from app.state_machine import StateMachine



CONFIDENCE = 0.6      
PREDICT_EVERY = 0.1  
SMOOTHING = 5      



bundle = joblib.load("model.joblib")
model, classes = bundle["model"], bundle["classes"]
model.n_jobs = 1 


NON_EMOTIONS = {"Starting", "Ending", "Confirm", "Deny", "None"}  # holder for controls that aren't emotions
EMOTIONS = [n for n in dict.fromkeys(display_name(c) for c in classes)
            if n not in NON_EMOTIONS]                            
machine = StateMachine(EMOTIONS)                                 
feedback = Feedback(os.path.join(HERE, "feedback.csv"))           
scores = {}                                                       # handles the scoring of the emotions
last_frame = time.time()                                          
show_debug = False                                                
 

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1)
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
    dt, last_frame = now - last_frame, now #going to handle the opeartion from feedback to get more info on the emotions when capturing
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
            scores = {} 
        else:
            probs = model.predict_proba([clip_features(resample(times, window))])[0]
            scores = {} #keeping score of the overall predicting from the model and displaying the emotion based on what is being shown on the screen

            for p, lab in zip(probs, model.classes_):
                name = display_name(classes[int(lab)])
                scores[name] = scores.get(name, 0.0) + float(p)
            
            best = int(np.argmax(probs))
            label = int(model.classes_[best])
            if probs[best] >= CONFIDENCE and classes[label] != "None":
                votes.append((label, float(probs[best])))
            else:
                votes.append((None, 0.0))

        top, n = Counter(l for l, _ in votes).most_common(1)[0]
        shown = top if (top is not None and n > len(votes) / 2) else None
        shown_conf = float(np.mean([c for l, c in votes if l == shown])) if shown is not None else 0.0


    #setting up a different way of updating the steps in order to align with the UI 
    shown_name = display_name(classes[shown]) if shown is not None else None
    machine.update(shown_name, shown_conf, dt)
    debug = {
        "state": machine.state,
        "smoothed": shown_name or "-",
        "confidence": f"{shown_conf * 100:.0f}%",
        "lock-in": f"{machine.progress * 100:.0f}%",
        "votes": f"{len(votes)}/{SMOOTHING}",
    } if show_debug else None
    canvas = ui.draw(frame, machine, feedback, EMOTIONS, scores,
                     hands_now=bool(results.multi_hand_landmarks), debug=debug)
    cv2.imshow("eMotion", canvas)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("d"):                                          
        show_debug = not show_debug                           
        machine.answer_key(True)                               
    if key == ord("x"):                                           
        machine.answer_key(False)                                
    if machine.answer is not None:                                # saves the OK / X answer
        feedback.save(machine.locked, machine.locked_conf,
                      machine.answer, machine.answered_with)     



    # if shown is not None:
    #     text, color = "%s (%d%%)" % (display_name(classes[shown]), round(100 * shown_conf)), (0, 255, 0)
    # else:
    #     text, color = "...", (200, 200, 200)
    # cv2.putText(frame, "Detected: " + text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    # cv2.imshow("eMotion", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

cap.release()
cv2.destroyAllWindows()