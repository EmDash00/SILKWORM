import threading as th


class PauseableLoop(th.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._paused = False
        self._stopped = True
        self._pause_condition = th.Condition(th.Lock())

    def run(self):
        try:
            while not self._stopped:
                with self._pause_condition:
                    while self._paused:
                        self._pause_condition.wait()

                    self._target(*self._args, **self._kwargs)
        finally:
            del self._target, self._args, self._kwargs

    def start(self, pause=False):
        self._stopped = False

        self._paused = pause
        if (pause):
            self._pause_condition.acquire()

        super().start()

    def stop(self):
        self._stopped = True

    @property
    def paused(self):
        return self._paused

    def pause(self):
        if (not self._stopped):
            self._paused = True
            self._pause_condition.acquire()

    def resume(self):
        if (not self._stopped):
            self._paused = False
            self._pause_condition.notify()
            self._pause_condition.release()
