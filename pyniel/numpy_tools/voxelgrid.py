# github.com/danieldugas/pyniel
import numpy as np

def pointcloud_to_voxelgrid(pointcloud, gridsize=24, n_spatial_dims=None, return_indices=False, pad=False):
    """ Transforms a pointcloud into a voxelgrid

    Parameters
    ----------
    pointcloud : array_like
        The point cloud, of shape (n_points, n_spatial_dims+n_extra_channels).
        The first n_spatial_dims values of the second dimension should
        correspond to the spatial channels x, y, z, ...
    gridsize : int, or tuple
        The shape of the grid, in i j k ... dimensions
        (corresponding to x y z ... respectively).
        if a single int, cubic grid with size_i = size_j = size_k = ...
        if a tuple, len(tuple) should be n_spatial_dims
    n_spatial_dims : int or None, optional
        if specified, will discard any dimensions past n_spatial_dims as extra_channels
    return_indices : bool, optional
        if True, returns an array of shape (n_points, n_spatial_dims), containing
        the indices in voxelgrid of each point in the original pointcloud
    pad : bool, optional
        default False,
        if True, the final result will be padded by one at the start and end of each spatial dimension
        resulting in a final grid of shape gridsize + 2

    Returns
    -------
    voxelgrid : array_like
        voxelgrid of shape (gridsize,n_spatial_dims+1)
        a channel for the occupancy of each voxel is added after after the spatial channels.
        extra_channels are discarded
    ijk : array_like
        array of shape (n_points, n_spatial_dims), containing the indices in voxelgrid of
        each point in the original pointcloud


    Example
    -------
    >>> a = np.array([[ 594.91059495,  -12.68847179,  380.85336289], [ -29.15473572,  617.57815194,  192.9789925 ], [ 617.35293371,  331.56653079,   11.62966491], [-182.31756149,  323.28363397,  -62.67164303], [ -45.40210734,  374.61999772,  236.48058249], [-238.75399034, -192.61805858,  -31.56503694], [  57.93188967,  645.13968563,    9.51061704], [ 342.32439168, -194.1729977 ,  258.29641547], [ 609.48471435,  663.82602109, -120.32760802], [ 109.49060671, -236.72820364,  269.298423  ]])
    >>> pointcloud_to_voxelgrid(a, gridsize=3)[...,-1]
    array([[[1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 1.]],
    <BLANKLINE>
           [[0., 0., 1.],
            [0., 0., 0.],
            [1., 0., 0.]],
    <BLANKLINE>
           [[0., 0., 2.],
            [1., 0., 0.],
            [1., 0., 0.]]])
    """
    # Sanitize params
    if n_spatial_dims is None:
        n_spatial_dims = pointcloud.shape[1]
    if not isinstance(n_spatial_dims, int):
        raise ValueError("Expected n_spatial_dims to be an int, got {}".format(type(n_spatial_dims)))
    if isinstance(gridsize, int):
        gridsize = np.array([gridsize] * n_spatial_dims)
    else:
        gridsize = np.array(gridsize)
        if gridsize.shape != (n_spatial_dims,):
            raise ValueError(
            "Parameter gridsize must either be an int, or a tuple of length n_spatial_dims."
            "n_spatial_dims={}, gridsize.shape={}".format(n_spatial_dims, gridsize.shape))
    n_extra_channels = pointcloud.shape[1] - n_spatial_dims
    if n_extra_channels < 0:
        raise ValueError(
                "Expected pointcloud to have shape (n_points, n_spatial_dims + n_extra_channels),"
                " got {}".format(pointcloud.shape))
    # Find the xyz limits and step size
    xyzmin = pointcloud[:,:n_spatial_dims].min(axis=0)
    xyzmax = pointcloud[:,:n_spatial_dims].max(axis=0)
    dxyz = (xyzmax - xyzmin) / gridsize
    # Transform x y z ... to integer indices, normalizing axes
    ijk = ((pointcloud[:,:n_spatial_dims] - xyzmin) / dxyz).astype(int)
    # points at the border of the last voxels should be put in the last voxels
    out_of_range = (ijk == gridsize)
    ijk[out_of_range] -= 1
    # Initialize voxel grid
    # Create xyz values to fill voxel grid
    list_of_axis_ranges = [np.arange(dimsize) * dxi + xi0
            for dimsize, dxi, xi0 in zip(gridsize, dxyz, xyzmin)]
    if pad:
        list_of_axis_ranges = [np.pad(dim, (1,1), mode='reflect', reflect_type='odd')
                               for dim in list_of_axis_ranges]
    xyzgrid = np.stack(np.meshgrid(*list_of_axis_ranges, indexing="ij"), axis=-1)
    # Calculate occupancies
    ijk_unique, occupancies = np.unique(ijk, return_counts=True, axis=0)
    occupancygrid = np.zeros(tuple(gridsize))
    occupancygrid[tuple(ijk_unique.T)] = occupancies
    padsize = 0
    if pad:
        occupancygrid = np.pad(occupancygrid, 1, mode='constant', constant_values=0)
        padsize = 2
    # Concatenate xyz (n1,n2,n3,...,n_spatial_dims) and
    # occupancies (n1,n2,n3,...,1) to make full size voxelgrid
    voxelgrid = np.zeros(tuple(gridsize + padsize)+(n_spatial_dims+1,))
    voxelgrid[...,:n_spatial_dims] = xyzgrid
    voxelgrid[...,n_spatial_dims] = occupancygrid
    if return_indices:
        return voxelgrid, ijk
    else:
        return voxelgrid

def pointcloud_to_xyzgrid(pointcloud, gridsize=24, n_spatial_dims=None, pad=False):
    """ Transforms a pointcloud into an xyz grid

    Parameters
    ----------
    pointcloud : array_like
        The point cloud, of shape (n_points, n_spatial_dims+n_extra_channels).
        The first n_spatial_dims values of the second dimension should
        correspond to the spatial channels x, y, z, ...
    gridsize : int, or tuple
        The shape of the grid, in i j k ... dimensions
        (corresponding to x y z ... respectively).
        if a single int, cubic grid with size_i = size_j = size_k = ...
        if a tuple, len(tuple) should be n_spatial_dims
    n_spatial_dims : int or None, optional
        if specified, will discard any dimensions past n_spatial_dims as extra_channels
    pad : bool, optional
        default False,
        if True, the final result will be padded by one at the start and end of each spatial dimension
        resulting in a final grid of shape gridsize + 2

    Returns
    -------
    xyzgrid : array_like
        voxelgrid of shape (gridsize,n_spatial_dims)
        extra_channels are discarded


    """
    # Sanitize params
    if n_spatial_dims is None:
        n_spatial_dims = pointcloud.shape[1]
    if not isinstance(n_spatial_dims, int):
        raise ValueError("Expected n_spatial_dims to be an int, got {}".format(type(n_spatial_dims)))
    if isinstance(gridsize, int):
        gridsize = np.array([gridsize] * n_spatial_dims)
    else:
        gridsize = np.array(gridsize)
        if gridsize.shape != (n_spatial_dims,):
            raise ValueError(
            "Parameter gridsize must either be an int, or a tuple of length n_spatial_dims."
            "n_spatial_dims={}, gridsize.shape={}".format(n_spatial_dims, gridsize.shape))
    n_extra_channels = pointcloud.shape[1] - n_spatial_dims
    if n_extra_channels < 0:
        raise ValueError(
                "Expected pointcloud to have shape (n_points, n_spatial_dims + n_extra_channels),"
                " got {}".format(pointcloud.shape))
    # Transform x y z ... to integer indices, normalizing axes
    xyzmin = pointcloud[:,:n_spatial_dims].min(axis=0)
    xyzmax = pointcloud[:,:n_spatial_dims].max(axis=0)
    dxyz = (xyzmax - xyzmin) / gridsize
    # Initialize voxel grid
    # Create xyz values to fill voxel grid
    list_of_axis_ranges = [np.arange(dimsize) * dxi + xi0
            for dimsize, dxi, xi0 in zip(gridsize, dxyz, xyzmin)]
    if pad:
        list_of_axis_ranges = [np.pad(dim, (1,1), mode='reflect', reflect_type='odd')
                               for dim in list_of_axis_ranges]
    xyzgrid = np.stack(np.meshgrid(*list_of_axis_ranges, indexing="ij"), axis=-1)
    return xyzgrid


if __name__ == "__main__":
    import doctest

    doctest.testmod()
