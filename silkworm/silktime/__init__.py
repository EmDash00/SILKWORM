from timeit import default_timer, process_time
from math import floor
from enum import Enum
"""
.. module::silktime
   :platform: Cross platform
   :synopsis: Contains the timing libraries.

.. moduleauthor:: Drason "Emmy" Chow <drasonchow@gmail.com>
"""


class Units(Enum):
    """Enum that contains various units the Timer class can be set to"""

    SEC = 1
    MS = 1e3
    US = 1e6
    NS = 1e9


class Timer:
    """
    High precision timer utility class that makes it easy to time events
    elegantly
    """

    def __init__(self, t0=None, units=Units.SEC, start=False,
                 count_sleep=True):
        """
        Creates a timer object.

        :param t0: starting time of your timer in units of your preferred
        timing function. Defaults to have timer start at 0

        :type t0: float, None

        :param units: enum valued time unit. Can be seconds, milliseconds,
        microseconds, or nanoseconds

        :type units: Units

        :param start: Whether or not your timer is started after instantiation.
        False by default.

        :type start: bool

        :param count_sleep: The timer counts time the process is asleep as
        having elapsed. True by default.

        :type count_sleep: bool

        :returns: newly instantitated Timer object
        :rtype: Timer
        """

        if (count_sleep):
            self._clock = default_timer
        else:
            self._clock = process_time

        if (t0 is None):
            self._t0 = self._clock()
        else:
            self._t0 = t0

        self._units = units
        self._paused = not (start)

        self._mem = 0

    @staticmethod
    def wait(duration, units=Units.SEC, count_sleep=False):
        """
        Causes the current thread to pause for the indicated duration.

        :param duration: the duration to wait in your chosen units
        :param units: time units from the Units enum.
        """
        clock = default_timer if not count_sleep else process_time
        t0 = clock()

        while ((clock() - t0) * units.value < duration):
            pass

    def set_units(self, units):
        """ Set the units for the timer to use """

        self._units = units

    def elapsed(self):
        """
        Compute the amount of time elapsed since the last elapsed time reset

        :return: time elapsed in the timer's units (timer - t0)
        :rtype: float
        """
        if (self._paused):
            return (self._mem)
        else:
            return ((self._clock() - self._t0) * self._units.value + self._mem)

    def elapsed_raw(self):
        if (self._paused):
            return self._mem
        else:
            return (self._clock() - self._t0 + self._mem)

    def reset(self, pause=True):
        """
        Resets the time elapsed. Currently there is no way to count cycles
        independently of time elapsed. This is a feature planned for a
        future update.

        :param pause: pause the timer after resetting. default value is True.
        :rtype pause: bool
        """
        self._paused = pause

        self._mem = 0
        self._t0 = self._clock()

    def start(self):
        """
        Start the timer if it's paused.
        Does nothing if the timer is already started.
        """

        if (self._paused):
            self._paused = False
            self._t0 = self._clock()

    def stop(self):
        """
        Stops the timer if it's not paused.
        Does nothing if the timer is already paused.
        """

        if (not self._paused):
            self._mem = self.elapsed()
            self._paused = True


class TimeMismatch(Exception):
    pass


class IntervalTimer(Timer):
    """
    Inherits from Timer.
    An elegant way of measuring a number of elapsed fixed time intervals.
    """

    def __init__(self, interval, strict=False, grace_period=1, **kwargs):
        """
        Instantiates a CycleTimer object

        :param interval: interval between timer cycles.
        interval = 0 will raise a ValueError

        :type interval: float

        :param strict: if true, the timer will raise an exception if an
        unacceptable number of clock cycles is skipped. False by default.

        :type strict: bool

        :param grace_period: maximum acceptable time between calls to
        elapsed() or tick(). Good if lag in a system is unacceptable.

        :type grace_period: float

        :raises: ValueError
        :returns: newly instantiated CycleTimer object
        :rtype: CycleTimer

        """

        super().__init__(**kwargs)

        self._interval = interval
        self._strict = strict
        self._grace_period = grace_period

        self._cycles = 0

        # Call to elapsed sets _cycles
        self.elapsed()

    def set_strictness(self, strict=False):
        """
        Sets the strictness of the timer.

        :param strict: The strictness of the timer. False by default.
        :type strict: bool
        """

        self._strict = strict

    def set_interval(self, interval, align=False, redefine_past=False):
        """
        Set the interval between timer cycles

        :param interval: interval between timer cycles in the timer's units
        :type interval: float

        :param align: whether or not to wait for the current timer cycle to
        complete. False sets the interval immediately. Defaults to false
        :type align: bool

        :param redefine_past: redefines all past cycles to use the new
        interval, reinterpretting the current value of elapsed time.
        Defaults to false.

        :type redefine_past: bool
        """

        if (align):
            while (not self.tick()):
                pass

        if (redefine_past):
            self._mem *= (self._interval / interval)
        else:
            self._mem = self.elapsed()
            self._t0 = default_timer()

        self._interval = interval

    def set_grace_period(self, grace_period):
        """
        Sets the grace period of the timer.

        :param grace_period: the grace period of the time
        :type grace_period: float
        """

        self._grace_period = grace_period

    def elapsed(self):
        """
        Compute total fractional number of timer cycles since the last set

        :raises TimerMismatch: Exception raised in strict timers that exceed
        the grace period between calls to elapsed() or tick()

        :returns: Fractional number of timer cycles elapsed
        :rtype: float
        """

        try:
            if (self._paused):

                return self._mem

            else:

                now = ((default_timer() - self._t0) *
                       self._units.value) / self._interval + self._mem

                # expected difference is 1
                if (self._strict):
                    if ((now - self._cycles) > (self._grace_period + 1)):
                        raise TimeMismatch(
                            "Timer mistmatch. One or more clock cycle(s) "
                            "was skipped.")
                self._cycles = now
                return now

        except ValueError:
            print(
                "Cycles cannot be counted with interval 0 (default interval)")
            raise ValueError

    def elapsed_discrete(self):
        """
        Compute whole number of timer cycles since the last set

        :returns: Whole number of timer cycles elapsed
        :rtype: int
        """
        return floor(self.elapsed())

    def stop(self):
        """Pauses the timer noting the current number of cycles"""
        super().stop()
        self._cycles = self._mem

    def tick(self):
        """
        Tells whether one or more clock cycles have been completed since the
        last call tick() or cycles()

        :returns: True or False depending on whether or not one or more whole
        clock cycles have been completed since the last call to tick()
        or elapsed()

        :rtype: bool
        """
        return (floor(self._cycles) < self.elapsed_discrete())
