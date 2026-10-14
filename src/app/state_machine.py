""" This is going to keep track of which screen the app is on and fills the progess bars. 

Each frame, live_recognize.py calls update() with what the mdoel currently shows, then reads the fields below to know what to draw. 


"""

SENSING, CONFIRM, RESULT = "SENSING", "CONFIRM", "RESULT"




class StateMachine: 
    def __init__(self, emotions, hold_seconds = 1.0, answer_seconds=0.6, result_seconds=1.5):
        self.emotions = set(emotions)
        self.hold_seconds = hold_seconds
        self.answer_seconds = answer_seconds
        self.result_seconds = result_seconds


        self.state = SENSING
        self.candidate = None #emotion the loading bar is filling for 
        self.progress = 0.0 #loading bair, 0.0 -> 1.0
        self.locked = None #emotion shown on the ocnfirm screen
        self.locked_conf = 0.0
        self.yes = 0.0 #YES card progress, 0.0 -> 1.0
        self.no = 0.0 #NO card progess, 0.0 -> 1.0
        self.answer = None #This is for True/False only on the frame an answer is given
        self.answered_with = None #"gesture" or "key"
        self.last_correct = False #answer shown on the result screen
        self._result_timer = 0.0


    def update(self, shown, conf, dt) :
        # This is showing the name of the model currently showing like an emotion, a confirmation/deny, or None
        # conf = the confidence (0-1). dt = seconds since the last frame

        self.answer = None
        if self.state == SENSING: 
            self._sense(shown, conf, dt)
        elif self.state == CONFIRM: 
            self._confirm(shown, dt)
        else : 
            self._result_timer -= dt
            if self._result_timer <= 0: 
                self._start_sensing()

    def answer_key(self, correct) :
        # Calls when O (True) or X (False). Ignored outside the confirm screen
        if self.state == CONFIRM: 
            self._finish(correct, "key")


    # the three steps (sense, confirm results)
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
        self._result_timer = self.result_seconds

    def _start_sensing(self):
        self.state = SENSING
        self.candidate, self.progress = None, 0.0
        print("[sensing] ready for the next gesture")