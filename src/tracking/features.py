"""Shared settings and feature code for the eMotion gesture project.

Used by record_clips.py, train_model.py and live_recognize.py so that
training and live prediction always process hands in exactly the same way.
(No mediapipe import here on purpose - this file only needs numpy.)
"""
import numpy as np

# One entry per trained class. In record_clips.py the number keys 0-9 jump to the first ten.
# A name like "Happy_thumb" is displayed as "Happy" (text before the first underscore),
# so an emotion with two very different motions can be split into two classes later.
CLASSES = [
    "Starting", #hand wave
    "Neutral", #so-so motion
    "Neutral_shrug",
    "Neutral_67", 
    "Happy", #thumbs up 
    "Happy_fists",
    "Sad", #thumbs down
    "Sad_weepy",
    "Frustrated", #vibrating fists
    "Frustrated_claws",
    "Worried", #shaking hands
    "Worried_face",
    "Ending", #hand wave away 
    "Confirm", #okay hand
    #"Confirm_thumb", 
    #"Confirm_left",
    "Deny", #X motion
    #"Deny_thumb",
    #"Deny_right",
    "None"  # idle hands / random movement, so the model can say "no gesture"
]

CLIP_SECONDS = 1.5   # length of one gesture window
CLIP_FRAMES = 30     # every clip is resampled to this many frames

# Per frame we store two hand "slots" plus the offset between the wrists.
# Each slot: [present, wrist_x, wrist_y, palm_size, 21 landmarks x (x, y)]
SLOT = 46
FRAME_DIM = 2 * SLOT + 2


def display_name(cls):
    return cls.split("_")[0]


def _hand_block(hand_landmarks, aspect):
    """Hand shape (relative to wrist, scaled by palm size) plus where the hand is."""
    pts = np.array([[lm.x * aspect, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = pts[0]
    scale = max(float(np.linalg.norm(pts[9] - wrist)), 1e-6)  # wrist -> middle finger base
    shape = ((pts - wrist) / scale).flatten()
    return np.concatenate([[1.0, wrist[0], wrist[1], scale], shape])


def frame_to_vector(results, aspect):
    """Turn one MediaPipe result into a fixed-length vector (zeros if no hands).

    aspect = frame width / frame height.
    Two hands: slot 0 is the Left-labelled hand, slot 1 the Right-labelled one.
    One hand: it always goes in slot 0, whichever hand it is, so the one-hand
    gestures work with either hand.
    """
    vec = np.zeros(FRAME_DIM, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return vec

    hands = []
    for i, lm in enumerate(results.multi_hand_landmarks[:2]):
        label = results.multi_handedness[i].classification[0].label
        hands.append((label, lm))

    if len(hands) == 2:
        if hands[0][0] == hands[1][0]:  # both labelled the same: order by position instead
            hands.sort(key=lambda h: h[1].landmark[0].x)
        else:
            hands.sort(key=lambda h: h[0] != "Left")  # Left first

    blocks = [_hand_block(lm, aspect) for _, lm in hands]
    for s, block in enumerate(blocks):
        vec[s * SLOT:(s + 1) * SLOT] = block
    if len(blocks) == 2:
        vec[2 * SLOT] = blocks[1][1] - blocks[0][1]      # wrist dx
        vec[2 * SLOT + 1] = blocks[1][2] - blocks[0][2]  # wrist dy
    return vec


def resample(times, vecs, n=CLIP_FRAMES):
    """Pick n frames evenly spaced in time, so clips don't depend on camera fps."""
    times = np.asarray(times, dtype=np.float64)
    vecs = np.asarray(vecs, dtype=np.float32)
    targets = np.linspace(times[0], times[-1], n)
    idx = np.abs(times[:, None] - targets[None, :]).argmin(axis=0)
    return vecs[idx]


def clip_features(seq):
    """Summarize a clip (frames x FRAME_DIM) as one feature vector for the classifier.

    Uses average / spread / min / max of every value over time, the average
    frame-to-frame change (this is what captures jittering, shaking, flapping),
    and first-to-last change (captures direction, e.g. hands moving apart).
    """
    seq = np.asarray(seq, dtype=np.float32).copy()

    # Remove where in the frame the gesture happened: wrist positions become
    # relative to the clip's average wrist position.
    present = [seq[:, s * SLOT] > 0.5 for s in range(2)]
    wrists = [seq[present[s], s * SLOT + 1:s * SLOT + 3] for s in range(2) if present[s].any()]
    if wrists:
        ref = np.concatenate(wrists).mean(axis=0)
        for s in range(2):
            seq[present[s], s * SLOT + 1] -= ref[0]
            seq[present[s], s * SLOT + 2] -= ref[1]

    diffs = np.abs(np.diff(seq, axis=0))
    return np.concatenate([
        seq.mean(axis=0),
        seq.std(axis=0),
        seq.min(axis=0),
        seq.max(axis=0),
        diffs.mean(axis=0),
        seq[-1] - seq[0],
    ])
