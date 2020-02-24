import rospy
import traceback

def shutdown_if_fail(function):
    def wrapper_for_routine(*args, **kwargs):
        try:
            function(*args, **kwargs)
        except: # noqa
            traceback.print_exc()
            rospy.signal_shutdown("Exception caught in routine thread.")
            raise
    return wrapper_for_routine

