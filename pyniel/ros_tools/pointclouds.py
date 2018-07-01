import warnings
import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
""" ros pointcloud tools
Daniel Dugas
"""


def pointcloud2_to_numpy(pointcloud, field_names=None):
    """ Convert from ros PointCloud2 Message to numpy array.
    preserves structure (height, width)

    Parameters
    ----------
    pointcloud : PointCloud2
        point cloud message to convert
    field_names : list or None
        default None. Keeps all fields.
        if list of strings, fetches only fields where name is in field_names.
        ordering of fields in output array is same as order in PointCloud2 fields.

    Returns
    -------
    result : ndarray
        converted array.
        shape is (width, n_fields) if pointcloud.height is 1
        shape is (height, width, n_fields) otherwise

    Example
    -------
    """
    assert isinstance(pointcloud, PointCloud2)
    if field_names is None:
        field_names = [f.name for f in pointcloud.fields]
    result = np.array([list(point) for point in pc2.read_points(pointcloud, field_names=field_names)])
    if pointcloud.height > 1:
        result = result.reshape((pointcloud.height, pointcloud.width, -1))
    return result



if __name__ == "__main__":
    import doctest
    doctest.testmod()
