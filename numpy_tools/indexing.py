import numpy as np

def as_idx(a, axis=None):
    """ Returns an array of shape a containing indices for a

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    axis : int, None, list of ints, or 'all'
        Axis along which to return indices. If None, the flat index.

    Returns
    -------
    result : ndarray
        Array of shape (a.shape, len(axis))
        if axis is None or a single int, (a.shape,)

    Notes
    -----

    Example
    -------
    >>> a = np.array([[0, 0, 0], [0, 0, 9], [3, 6, 0]])
    >>> a
    array([[0, 0, 0],
           [0, 0, 9],
           [3, 6, 0]])
    >>> as_idx(a)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> as_idx(a, axis='all')
    array([[[0, 0],
            [0, 1],
            [0, 2]],
    <BLANKLINE>
           [[1, 0],
            [1, 1],
            [1, 2]],
    <BLANKLINE>
           [[2, 0],
            [2, 1],
            [2, 2]]])


    >>> as_idx(a, axis=0)
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2]])
    """
    if axis is None:
        return np.arange(len(a.flatten())).reshape(a.shape)
    idxs = np.array(np.where(np.ones(a.shape))).T.reshape(a.shape + (-1,))
    if axis == 'all':
        return idxs
    return idxs[...,axis]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
