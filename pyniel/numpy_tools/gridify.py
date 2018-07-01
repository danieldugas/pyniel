from .divisors import divisors
import numpy as np


def gridify(
    x, y, val=None, grid=None, resolution=(100, 100), value_approximation="min"
):
    # Converts a set of points in x, y to an occupancy grid
    # Optionally a value can be attached to each point, in which case cell values are approximated
    # according to the value_approximation method
    # - average: cell value is average of values for points in cell
    # - min: cell value is min of values for points in cell
    # - max: cell value is max of values for points in cell
    # - center: cell value is value for point closest to center of cell
    # - occupancy: force reverting to occupancy values
    # In case of excessive memory use (large grid size, many points, etc)
    # the function attempts to subdivide and solve the problem recursively.
    if grid is None:
        dx = (max(x) - min(x)) / resolution[0]
        hdx = dx / 2.
        dy = (max(y) - min(y)) / resolution[1]
        hdy = dy / 2.
        grid_x = np.linspace(min(x) + hdx, max(x) - hdx, resolution[0])
        grid_y = np.linspace(min(y) + hdy, max(y) - hdy, resolution[1])
        xx, yy = np.meshgrid(grid_x, grid_y)
        xminbound = xx - hdx
        xmaxbound = xx + hdx
        yminbound = yy - hdy
        ymaxbound = yy + hdy
    else:  # TODO: in this case the grid is decentered. Option?
        grid_x = grid[0]
        grid_y = grid[1]
        resolution = (
            len(grid_x),
            len(grid_y),
        )  # NOTE: in this case resolution is coerced
        hdx = np.pad(
            np.diff(grid_x), (0, 1), "symmetric"
        )  # diff with the last element duplicated
        hdy = np.pad(np.diff(grid_y), (0, 1), "symmetric")
        xx, yy = np.meshgrid(grid_x, grid_y)
        xminbound = xx
        xmaxbound = xx + hdx[None, :]
        yminbound = yy
        ymaxbound = yy + hdy[:, None]
        # in case grid is smaller than data, remove points outside.
        crop = np.logical_or.reduce(
            (
                x < np.min(xminbound),
                x > np.max(xmaxbound),
                y < np.min(yminbound),
                y > np.max(ymaxbound),
            )
        )
        x = np.delete(x, crop)
        y = np.delete(y, crop)
        if val is not None:
            val = np.delete(val, crop)
    # C maps in which cell belongs each point - C.shape = (n, resolution)
    try:
        C = np.logical_and.reduce(
            (
                x[:, None, None] > xminbound,
                y[:, None, None] > yminbound,
                x[:, None, None] < xmaxbound,
                y[:, None, None] < ymaxbound,
            )
        )
        masked = np.ma.masked_array(val[:, None, None] * np.ones(C.shape), mask=~C)
        if val is None or value_approximation == "occupancy":
            values = np.sum(C.astype(int), axis=0)
        elif value_approximation == "average":
            values = np.average(masked, axis=0)
        elif value_approximation == "min":
            values = np.min(masked, axis=0)
        elif value_approximation == "max":
            values = np.max(masked, axis=0)
        elif value_approximation == "center":
            # TODO: would return the value of point closest to grid center
            D = np.sqrt(
                np.square(xx[None, :, :] - x[:, None, None])
                + np.square(yy[None, :, :] - y[:, None, None])
            )
            masked_D = np.ma.masked_array(D, mask=~C, fill_value=None)
            idx0 = np.argmin(masked_D, axis=0).flatten()
            idx12 = np.indices(xx.shape)
            idx1 = idx12[0].flatten()
            idx2 = idx12[1].flatten()
            values = masked[idx0, idx1, idx2].reshape(xx.shape)
    except MemoryError:
        # TODO: what happens when resolution has large primes!
        subdivisions = tuple(map(min, divisors(resolution)))
        sub_resolutions = tuple(np.array(resolution) / np.array(subdivisions))
        print(
            "Insufficient memory, splitting into %s x %s subproblems of resolution %s",
            subdivisions[0],
            subdivisions[1],
            sub_resolutions,
        )
        for sub_grid_x in np.split(xminbound[0, :], subdivisions[0]):
            for sub_grid_y in np.split(yminbound[:, 0], subdivisions[1]):
                _, sub_values = gridify(
                    x,
                    y,
                    val=val,
                    grid=(sub_grid_x, sub_grid_y),
                    value_approximation=value_approximation,
                )
                try:
                    v_values = np.vstack((v_values, sub_values))
                except:
                    v_values = sub_values
            try:
                values = np.hstack((values, v_values))
            except:
                values = np.copy(v_values)
                del v_values
        assert values.shape == xx.shape

    return (xx, yy), values
