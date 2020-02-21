import time

class WalltimeRate(object):
    """ Convenience class for reproducing rostime Rate with walltime """
    def __init__(self, frequency, keep_phase=False):
        if frequency <= 0:
            raise ValueError("Frequency can not be <= 0")
        self.last_time = time.time()
        self.sleep_duration = 1./frequency
        self.keep_phase = keep_phase

    def _time_to_next_and_sleep(self, sleep=False):
        current_time = time.time()
        elapsed = current_time - self.last_time
        # moved back in time
        if elapsed < 0:
            self.last_time = current_time
            time_to_next = 0
            if sleep:
                # sleep(0)
                self.last_time = current_time
            return time_to_next
        # missed a few cycles
        elif elapsed > self.sleep_duration:
            time_to_next = 0
            if sleep:
                # sleep(0)
                if self.keep_phase:
                # update last_time to the start of the current bin, violating frequency to
                # keep the original phase.
                # |     |     |     |     |     |  x  x <- resulting next scheduled timeout
                # ^start                           ^now
                #             ^last_time        ^'new' last_time
                    n_bins_to_skip = int(elapsed / self.sleep_duration)
                    self.last_time = self.last_time + n_bins_to_skip * self.sleep_duration
                else:
                    self.last_time = current_time
            return time_to_next
        # normal situation
        else:
            time_to_next = max(self.sleep_duration - elapsed, 0)
            if sleep:
                time.sleep(time_to_next)
                self.last_time += time_to_next
            return time_to_next

    def remaining(self):
        return self._time_to_next_and_sleep(sleep=False)

    def sleep(self):
        return (self._time_to_next_and_sleep(sleep=True) == 0)
