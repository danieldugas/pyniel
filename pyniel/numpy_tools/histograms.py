import numpy as np

_range = range


def histogram_along_axis(a, axis=None, bins=10, range=None, weights=None):
    """ Get a histogram of values along specific axis, all sharing the same bins
    last axis in output should have size bins, and contains histogram values
    - Daniel Dugas

    See numpy.histogram

    Example
    -------
    >>> a = np.ones((2,2))
    >>> a
    array([[1., 1.],
           [1., 1.]])
    >>> histogram_along_axis(a, bins=4, axis=0)
    (array([[0, 0, 2, 0],
           [0, 0, 2, 0]]), array([0.5 , 0.75, 1.  , 1.25, 1.5 ]))
    """
    _, bin_edges = np.histogram(
        np.array([np.min(a), np.max(a)]), bins=bins, range=range, weights=weights
    )

    axis_ = axis
    # protect the new histogram dimension from being summed
    if axis is None:
        axis_ = tuple(_range(len(a.shape)))
    elif isinstance(axis, tuple) or isinstance(axis, list):
        axis_ = tuple([i - 1 if i < 0 else i for i in axis if i <= len(a.shape)])
    elif np.ndim(axis) == 0:
        if axis >= len(a.shape):
            np.sum(a, axis=axis)
            raise ValueError
        if axis == -1:
            axis_ = -2
    else:
        raise ValueError("Invalid axis value")
    return (
        np.sum(
            np.logical_or(
                np.logical_and(
                    a[..., None] >= bin_edges[:-1], a[..., None] < bin_edges[1:]
                ),
                (a[...] == bin_edges[-1])[..., None],
            ).astype(int),
            axis=axis_,
        ),
        bin_edges,
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
