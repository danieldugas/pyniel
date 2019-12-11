import numpy as np
import rospy
from sensor_msgs.msg import OccupancyGrid

def numpy_to_occupancy_grid_msg(arr, ref_map2d, frame_id, time=None):
    if not len(arr.shape) == 2:
            raise TypeError('Array must be 2D')
    arr = arr.T * 100.
    if not arr.dtype == np.int8:
        arr = arr.astype(np.int8)
    if time is None:
        time = rospy.Time.now()
    grid = OccupancyGrid()
    grid.header.frame_id = frame_id
    grid.header.stamp.secs = time.secs
    grid.header.stamp.nsecs = time.nsecs
    grid.data = arr.ravel()
    grid.info.resolution = ref_map2d.resolution()
    grid.info.height = arr.shape[0]
    grid.info.width = arr.shape[1]
    grid.info.origin.position.x = ref_map2d.origin[0]
    grid.info.origin.position.y = ref_map2d.origin[1]
    return grid 
