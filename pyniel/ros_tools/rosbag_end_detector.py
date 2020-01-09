import rospy
from timeit import default_timer as timer
import threading

def example_rosbag_end_callback():
    print("Rosbag end/pause detected!")
    return

class RosbagEndDetector(object):
    def __init__(self, rosbag_end_callback=None):
        self.latest_rostime = None
        self.rosbag_started = False
        if rosbag_end_callback is None:
            rosbag_end_callback = example_rosbag_end_callback
        self.rosbag_end_callback = rosbag_end_callback
        # start checking in separate thread
        self.check_every_s()

    def check_every_s(self):
        if self.latest_rostime is None:
            self.latest_rostime = rospy.Time.now()
        else:
            rostime = rospy.Time.now()
            if rostime-self.latest_rostime <= rospy.Duration(0.05) and self.rosbag_started:
                self.rosbag_end_callback()
                return
            elif not self.rosbag_started:
                rospy.logwarn("Rosbag started, enabling end detector.")
                self.rosbag_started = True
            self.latest_rostime = rostime
        threading.Timer(1, self.check_every_s).start()

