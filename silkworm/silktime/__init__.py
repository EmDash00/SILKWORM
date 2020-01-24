__all__ = ["CycleTimer", "TimedTask", "Timer", "Units"]
from timeit import default_timer
from .timer import Timer
from .cycletimer import CycleTimer
from .tasking import TimedTask
from .units import Units
"""
.. module::timer
   :platform: Cross platform
   :synopsis: Contains the Timer class.

.. moduleauthor:: Drason "Emmy" Chow <drasonchow@gmail.com>

