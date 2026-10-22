""" This is going to keep track of which screen the app is on and fills the progess bars.

Each frame, live_recognize.py calls update() with what the mdoel currently shows, then reads the fields below to know what to draw.

    IDLE     -> hold Starting gesture (or S)                -> SENSING
    SENSING  -> one emotion stays on top for hold_seconds   -> CONFIRM
             -> hold Ending gesture (or E)                  -> SUMMARY
    CONFIRM  -> hold Confirm / Deny (or O / X)              -> RESULT
    RESULT   -> after the timer                             -> SENSING
    SUMMARY  -> hold Starting gesture (or S) to play again  -> SENSING
"""

IDLE, SENSING, CONFIRM, RESULT, SUMMARY = "IDLE", "SENSING", "CONFIRM", "RESULT", "SUMMARY"




class StateMachine:
    def __init__(self, emotions, hold_seconds = 1.0, answer_seconds=0.6, result_seconds=1.5, correct_seconds=5.0,
                 gesture_seconds=1.0):
        self.emotions = set(emotions)
        self.hold_seconds = hold_seconds
        self.answer_seconds = answer_seconds
        self.result_seconds = result_seconds
        self.correct_seconds = correct_seconds
        self.gesture_seconds = gesture_seconds #how long to hold Starting/Ending


        self.state = IDLE #starts on the start screen
        self.candidate = None #emotion the loading bar is filling for
        self.progress = 0.0 #loading bair, 0.0 -> 1.0
        self.locked = None #emotion shown on the ocnfirm screen
        self.locked_conf = 0.0
        self.yes = 0.0 #YES card progress, 0.0 -> 1.0
        self.no = 0.0 #NO card progess, 0.0 -> 1.0
        self.start_p = 0.0 #Starting gesture progress, 0.0 -> 1.0
        self.end_p = 0.0 #Ending gesture progress, 0.0 -> 1.0
        self.answer = None #This is for True/False only on the frame an answer is given
        self.answered_with = None #"gesture" or "key"
        self.last_correct = False #answer shown on the result screen
        self.just_started = False #True only on the frame a new session begins
        self._result_timer = 0.0


    def update(self, shown, conf, dt) :
        # This is showing the name of the model currently showing like an emotion, a confirmation/deny, or None
        # conf = the confidence (0-1). dt = seconds since the last frame

        self.answer = None
        self.just_started = False

        if self.state in (IDLE, SUMMARY):
            self.start_p = self._fill(self.start_p, shown == "Starting", dt, self.gesture_seconds)
            if self.start_p >= 1.0:
                self._start_session()
        elif self.state == SENSING:
            self.end_p = self._fill(self.end_p, shown == "Ending", dt, self.gesture_seconds)
            if self.end_p >= 1.0:
                self._end_session()
            else:
                self._sense(shown, conf, dt)
        elif self.state == CONFIRM:
            self._confirm(shown, dt)
        else :
            self._result_timer -= dt
            if self._result_timer <= 0:
                self._start_sensing()


    #keyboard shortcuts just in case it isn't working with the hand gestures
    def answer_key(self, correct) :
        # Calls when O (True) or X (False). Ignored outside the confirm screen
        if self.state == CONFIRM:
            self._finish(correct, "key")

    def start_key(self) :
        # S key: starts a session from the start or summary screen
        if self.state in (IDLE, SUMMARY):
            self._start_session()

    def end_key(self) :
        # E key: ends the session while sensing
        if self.state == SENSING:
            self._end_session()


    # the steps (start, sense, confirm, result, summary)

    @staticmethod
    def _fill(value, active, dt, seconds) :
        """Fill a bar while active, drain it while not"""
        step = dt / seconds
        return min(1.0, value + step) if active else max(0.0, value - step)

    def _sense(self, shown, conf, dt) :
        step = dt / self.hold_seconds
        if shown in self.emotions:
            if shown == self.candidate:
                self.progress = min(1.0, self.progress + step)
            else :
                self.candidate, self.progress = shown, 0.0 # different emotion: restart

        else :
            self.progress = max(0.0, self.progress - step) #unsure: drain slowly
            if self.progress == 0.0:
                self.candidate = None
        if self.progress >= 1.0:
            self.state = CONFIRM
            self.locked, self.locked_conf = self.candidate, conf
            self.yes = self.no = 0.0
            print(f"[locked] {self.locked} ({conf * 100:.0f}%) - waiting for answer")


    def _confirm(self, shown, dt):
        step = dt / self.answer_seconds
        self.yes = min(1.0, self.yes + step) if shown == "Confirm" else max(0.0, self.yes - step )
        self.no = min(1.0, self.no + step) if shown == "Deny" else max(0.0, self.no - step)
        if self.yes >= 1.0 :
            self._finish(True, "gesture")
        elif self.no >= 1.0 :
            self._finish(False, "gesture")

    def _finish(self, correct, via):
        self.answer, self.answered_with, self.last_correct = correct, via, correct
        self.state = RESULT
        self._result_timer = self.correct_seconds if correct else self.result_seconds


    def _start_sensing(self):
        self.state = SENSING
        self.candidate, self.progress, self.end_p = None, 0.0, 0.0
        print("[sensing] ready for the next gesture")

    def _start_session(self):
        self.just_started = True
        self.start_p = 0.0
        print("[start] new session")
        self._start_sensing()

    def _end_session(self):
        self.state = SUMMARY
        self.end_p = self.start_p = 0.0
        self.candidate, self.progress = None, 0.0
        print("[end] session over")