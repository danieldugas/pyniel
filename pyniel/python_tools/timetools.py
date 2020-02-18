import time

class WalltimeRate(object):
    """ Convenience class for reproducing rostime Rate with walltime """
    def __init__(self, frequency):
        if frequency <= 0:
            raise ValueError("Frequency can not be <= 0")
        self.last_time = time.time()
        self.sleep_duration = 1./frequency

    def _time_to_next(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        # moved back in time
        if elapsed < 0:
            self.last_time = current_time
            return 0.
        # missed a few cycles
        if elapsed > self.sleep_duration:
            n_bins_to_skip = int(elapsed / self.sleep_duration)
            self.last_time = self.last_time + n_bins_to_skip * self.sleep_duration
            return 0.
        return self.sleep_duration - elapsed

    def remaining(self):
        return self._time_to_next()

    def sleep(self):
        time_to_next = max(self._time_to_next(), 0)
        self.last_time += time_to_next
        time.sleep(time_to_next)
