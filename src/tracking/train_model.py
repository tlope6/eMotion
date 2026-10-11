"""Train the gesture classifier from the clips in ./data and save model.joblib."""
import glob
import os
 
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
 
from features import CLASSES, clip_features
 
X, y, groups = [], [], []
for idx, name in enumerate(CLASSES):
    for path in glob.glob(os.path.join("data", name, "*.npy")):
        X.append(clip_features(np.load(path)))
        y.append(idx)
        groups.append(os.path.basename(path).split("_")[0])  
 
if not X:
    raise SystemExit("No clips found in ./data - run record_clips.py first.")
X, y, groups = np.array(X), np.array(y), np.array(groups)
 
counts = np.bincount(y, minlength=len(CLASSES))
for name, n in zip(CLASSES, counts):
    print("%-12s %3d clips" % (name, n))
missing = [CLASSES[i] for i in range(len(CLASSES)) if counts[i] == 0]
if missing:
    print("Warning: no clips yet for: " + ", ".join(missing))
present = [i for i in range(len(CLASSES)) if counts[i] > 0]
if counts[present].min() < 5:
    raise SystemExit("Need at least 5 clips for every gesture you have started recording.")
 
 
def make_model():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)
 
 
people = sorted(set(groups))
if len(people) >= 2:
    cv = LeaveOneGroupOut()
    print("\nTesting on unseen people (train on all but one, test on the one left out): " + ", ".join(people))
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print("\nOnly one person recorded, so this test is optimistic: clips from the same person look")
    print("alike. Record at least one more person and rerun for a realistic accuracy number.")
 
pred = cross_val_predict(make_model(), X, y, cv=cv, groups=groups)
 
names = [CLASSES[i] for i in present]
print("\nAccuracy: %.1f%%\n" % (100 * accuracy_score(y, pred)))
print(classification_report(y, pred, labels=present, target_names=names, zero_division=0))
print("Confusion matrix (rows = true gesture, columns = predicted), order: " + ", ".join(names))
print(confusion_matrix(y, pred, labels=present))
 
model = make_model().fit(X, y)
joblib.dump({"model": model, "classes": CLASSES}, "model.joblib")
print("\nSaved model.joblib (trained on all clips)")