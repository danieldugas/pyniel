from __future__ import print_function

from matplotlib import pyplot as plt
import time
from math import floor
import numpy as np


def progress_bar(batchmaker, ax=plt.figure("progress_bar").add_subplot(111)):
    ax.scatter(time.time(), batchmaker.n_batches_consumed())
    ax.set_ylim([0, batchmaker.n_batches_consumed() + batchmaker.n_batches_remaining()])


class TextProgressBar:
    def __init__(self, width=50, silent_init=False):
        self.width = width
        self.last_consumed = 0
        if not silent_init:
            print("|" + " " * self.width + "|")
        self.max_ = None
        self.remaining_points = None

    def update(self, batchmaker):
        # max progress
        consumed = batchmaker.n_batches_consumed()
        max_ = consumed + batchmaker.n_batches_remaining()
        if self.remaining_points is not None and self.max_ != max_:
            raise Warning(
                "Maximum limit for progress has changed.\
                           Use a separate progress bar for each batchmaker."
            )
        self.max_ = max_

        # start of progress bar
        if self.remaining_points is None:
            print("|", end="")
            # a trigger value for each remaining point
            self.remaining_points = list(
                np.round(1.0 * (np.arange(self.width) + 1) * self.max_ / self.width)
            )

        # if progress has reverted, start a new bar
        if consumed < self.last_consumed:
            self.abort()

        # Find how many bar points to print
        points = len([p for p in self.remaining_points if p <= consumed])
        self.remaining_points = [p for p in self.remaining_points if p > consumed]
        print("." * points, end="")

        self.last_consumed = consumed

        # Check if bar is complete
        if len(self.remaining_points) == 0:
            self.close_and_reset()

    def abort(self):
        missing = int(floor((1.0 - self.last_consumed) * self.width)) - 1
        if missing < 0:
            raise ValueError(
                "Negative missing points should not occur,\
            possible error within batchmaker. "
                + str(self.last_consumed)
            )
        print("x" + " " * missing, end="")
        self.close_and_reset()

    def close_and_reset(self):
        print("|", end="")
        self.__init__(width=self.width, silent_init=True)
