"""
Saves each OK / X answer to a CSV file and keeps the session score.

feedback.csv columns: time, predicted, confidence, correct, answered_with
Will be able to open this through Excel/Numbers to see which emotions the model gets wrong most often, adn then record more clips of those with record_clips.py

"""


import csv
import os
import time



class Feedback:
    def __init__(self, path="feedback.csv") :
        self.path = path
        self.answered = 0 #answered per session
        self.correct = 0 #correct answers this session


    def save(self, emotion, confidence, correct, answered_with):
        new_files = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_files:
                writer.writerow(["time", "predicted", "confidence", "correct", "answered_with"])

            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), emotion, f"{confidence:.2f}", correct, answered_with])
        self.answered += 1
        self.correct += int(correct)
        print(f"[answer] {emotion} -> {'correct' if correct else 'wrong'} "
                f"(via {answered_with}) session: {self.correct}/{self.answered}")


    def reset(self):
        """Clear the session score when a new session starts (the CSV file keeps everything)."""
        self.answered = 0
        self.correct = 0