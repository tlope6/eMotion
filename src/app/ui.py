"""Drawing helpers for the eMotion live window.

Layout:  [ camera view | side panel ]
  * side panel: title, status, live emotion bars, lock-in bar, session score, keys
  * overlays on the camera: confirm screen, result screen, optional debug info

Pure OpenCV + numpy, no mediapipe, so we will be able to tweak apperance with camera specifically
live_recognize.py only needs to call draw().

"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
PANEL_W = 300
PAD = 20

BG = (30, 26, 24)
CARD = (46, 41, 38)
TRACK = (66, 60, 56)
TEXT = (245, 245, 245)
MUTED = (160, 155, 150)
ACCENT = (240, 170, 70)
GOOD = (110, 200, 90)
BAD = (90, 90, 235)
WARN = (60, 190, 245)

EMOTION_COLORS = {
    "Happy": (60, 205, 250),
    "Sad": (215, 140, 60),
    "Frustrated": (80, 80, 235),
    "Worried": (195, 120, 185),
    "Neutral": (170, 170, 170),
}


def color_for(name):
    return EMOTION_COLORS.get(name, ACCENT)


#confirmation screen
MESSAGES = {
    "Happy": "I am so glad you feel happy! Today is going to be a great day.",
    "Sad": "It's okay to feel down. You're not alone.",
    "Frustrated": "Take a deep breath. You've got this.",
    "Worried": "One step at a time. It's going to be okay.",
    "Neutral": "Calm and steady. Nice and balanced.",
}


def wrap(s, scale, max_w, thick=3, font=FONT_BOLD):
    lines, cur = [], ""
    for word in s.split():
        trial = (cur + " " + word).strip()
        if cur and text_w(trial, scale, thick, font) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines



# small building blocks for the overall overlay

def put(img, s, org, scale=0.55, color=TEXT, thick=1, font=FONT):
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def text_w(s, scale, thick=1, font=FONT):
    return cv2.getTextSize(s, font, scale, thick)[0][0]


def put_center(img, s, y, scale, color=TEXT, thick=2, font=FONT_BOLD):
    put(img, s, ((img.shape[1] - text_w(s, scale, thick, font)) // 2, y), scale, color, thick, font)


def bar(img, x, y, w, h, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), TRACK, -1)
    fill = int(w * max(0.0, min(1.0, frac)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)


def dim(img, amount=0.6):
    img[:] = cv2.convertScaleAbs(img, alpha=1.0 - amount)


def draw_check(img, cx, cy, size, color, thick=6):
    pts = np.array([[cx - size, cy],
                    [cx - size // 3, cy + size * 2 // 3],
                    [cx + size, cy - size * 2 // 3]], np.int32)
    cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)


def draw_cross(img, cx, cy, size, color, thick=6):
    cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), color, thick, cv2.LINE_AA)
    cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), color, thick, cv2.LINE_AA)


# camera overlay

def draw_confirm(frame, emotion, conf, yes_p, no_p, ready=True):
    """'Was this right?' screen with YES / NO cards that fill as you hold a gesture."""
    h, w = frame.shape[:2]
    dim(frame)
    put_center(frame, "I think you're feeling", h // 2 - 120, 0.8, MUTED, 1, FONT)
    put_center(frame, emotion.upper(), h // 2 - 60, 1.8, color_for(emotion), 3)
    put_center(frame, f"{conf * 100:.0f}% confident", h // 2 - 22, 0.6, MUTED, 1, FONT)

    cw, ch, gap, top = 200, 150, 40, h // 2 + 10
    cards = [("YES", "Confirm gesture or O", yes_p, GOOD, draw_check),
             ("NO", "Deny gesture or X", no_p, BAD, draw_cross)]
    for i, (label, hint, prog, col, icon) in enumerate(cards):
        x = w // 2 - cw - gap // 2 if i == 0 else w // 2 + gap // 2
        cv2.rectangle(frame, (x, top), (x + cw, top + ch), CARD, -1)
        cv2.rectangle(frame, (x, top), (x + cw, top + ch), col, 2)
        icon(frame, x + cw // 2, top + 45, 22, col)
        put(frame, label, (x + (cw - text_w(label, 0.8, 2, FONT_BOLD)) // 2, top + 100),
            0.8, TEXT, 2, FONT_BOLD)
        put(frame, hint, (x + (cw - text_w(hint, 0.42)) // 2, top + 122), 0.42, MUTED)
        bar(frame, x + 15, top + ch - 18, cw - 30, 6, prog, col)

    if not ready:
        put_center(frame, "Get ready to answer...", top + ch + 35, 0.6, MUTED, 1, FONT)


def draw_result(frame, emotion, correct):
    h, w = frame.shape[:2]
    if correct:
        frame[:] = tuple(int(c * 0.55) for c in color_for(emotion))  # solid fill
        draw_check(frame, w // 2, h // 2 - 130, 30, TEXT, 8)
        lines = wrap(MESSAGES.get(emotion, "Thanks for confirming!"), 1.3, w - 100)
        y = h // 2 - 40
        for line in lines:
            put_center(frame, line, y, 1.3, TEXT, 3)
            y += 55
        put_center(frame, emotion.upper(), y + 20, 0.7, TEXT, 1, FONT)
    else:
        dim(frame)
        draw_cross(frame, w // 2, h // 2 - 50, 40, BAD, 10)
        put_center(frame, "Got it - saved as wrong", h // 2 + 40, 0.9, TEXT, 2)
        put_center(frame, emotion, h // 2 + 80, 0.7, MUTED, 1, FONT)


def draw_debug(frame, info):
    """Small black box (top-left of the camera) listing whatever you pass in."""
    box_h = 14 + 20 * len(info)
    cv2.rectangle(frame, (10, 10), (250, 10 + box_h), (0, 0, 0), -1)
    for i, (k, v) in enumerate(info.items()):
        put(frame, f"{k}: {v}", (20, 30 + i * 20), 0.45, (0, 255, 0))


# side panel to view emotions and percentages

def draw_panel(frame, status, status_color, emotions, scores, candidate, progress,
               answered, n_correct):
    """Returns a new image: the camera frame with the side panel attached on the right."""
    h, w = frame.shape[:2]
    canvas = np.full((h, w + PANEL_W, 3), BG, np.uint8)
    canvas[:, :w] = frame
    x0, inner = w + PAD, PANEL_W - 2 * PAD

    y = 42
    put(canvas, "eMotion", (x0, y), 1.0, TEXT, 2, FONT_BOLD)
    y += 22
    put(canvas, "gesture emotion sensing", (x0, y), 0.45, MUTED)

    y += 22  # status pill
    cv2.rectangle(canvas, (x0, y), (x0 + inner, y + 34), CARD, -1)
    cv2.circle(canvas, (x0 + 18, y + 17), 6, status_color, -1, cv2.LINE_AA)
    put(canvas, status, (x0 + 34, y + 23), 0.55, TEXT)

    y += 34 + 32
    put(canvas, "EMOTIONS", (x0, y), 0.45, MUTED)
    y += 14
    for name in emotions:
        p = scores.get(name, 0.0)
        hl = name == candidate
        put(canvas, name, (x0, y + 14), 0.5, TEXT if hl else MUTED, 2 if hl else 1)
        pct = f"{p * 100:.0f}%"
        put(canvas, pct, (x0 + inner - text_w(pct, 0.5), y + 14), 0.5, TEXT if hl else MUTED)
        bar(canvas, x0, y + 20, inner, 6, p, color_for(name))
        y += 36

    y += 14
    put(canvas, "LOCK-IN", (x0, y), 0.45, MUTED)
    y += 10
    label = f"{candidate}  {progress * 100:.0f}%" if candidate else "waiting for an emotion"
    put(canvas, label, (x0, y + 18), 0.55, TEXT if candidate else MUTED)
    bar(canvas, x0, y + 28, inner, 12, progress, color_for(candidate) if candidate else ACCENT)

    y += 70
    if answered:
        score = f"Session: {n_correct}/{answered} correct ({100 * n_correct // answered}%)"
    else:
        score = "Session: no answers yet"
    put(canvas, score, (x0, y), 0.45, MUTED)

    put(canvas, "O yes   X no   D debug   Q quit", (x0, h - 16), 0.42, MUTED)
    return canvas


# function that will be called from the live_recognize.py 

def draw(frame, machine, feedback, emotions, scores, hands_now, debug=None):
    """Draws everything for the current step and returns the full window image."""
    if machine.state == "CONFIRM":
        draw_confirm(frame, machine.locked, machine.locked_conf, machine.yes, machine.no)
    elif machine.state == "RESULT":
        draw_result(frame, machine.locked, machine.last_correct)
    if debug:
        draw_debug(frame, debug)

    if machine.state == "SENSING":
        if not hands_now:
            status, col = "Show your hands", MUTED
        elif machine.candidate:
            status, col = f"Sensing {machine.candidate}...", color_for(machine.candidate)
        else:
            status, col = "Watching...", ACCENT
        candidate, progress = machine.candidate, machine.progress
    elif machine.state == "CONFIRM":
        status, col = "Was that right?", WARN
        candidate, progress = machine.locked, 1.0
    else:
        status, col = "Saved", GOOD if machine.last_correct else BAD
        candidate, progress = machine.locked, 1.0

    return draw_panel(frame, status, col, emotions, scores, candidate, progress,
                      feedback.answered, feedback.correct)