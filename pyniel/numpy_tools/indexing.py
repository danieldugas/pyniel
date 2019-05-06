import warnings
import numpy as np

""" Index tools
Daniel Dugas
"""


def as_idx_array(a, axis=None):
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
    >>> as_idx_array(a)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> as_idx_array(a, axis='all')
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
    >>> as_idx_array(a, axis=0)
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2]])
    """
    if axis is None:
        return np.arange(len(a.flatten())).reshape(a.shape)
    idxs = np.array(np.where(np.ones(a.shape))).T.reshape(a.shape + (-1,))
    if axis == "all":
        return idxs
    return idxs[..., axis]


def as_idx(a):
    """ returns a tuple with indices for all values in a

    Parameters
    ----------
    a : array_like
        Array to be reshaped.

    Returns
    -------
    result : tuple of ndarrays
        of size len(a.shape)
        each ndarray has shape (len(a.flatten()),)

    Example
    -------
    >>> a = np.array([[0, 0, 0], [0, 0, 9], [3, 6, 0]])
    >>> a
    array([[0, 0, 0],
           [0, 0, 9],
           [3, 6, 0]])
    >>> as_idx(a)
    (array([0, 0, 0, 1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
    >>> np.all(a[as_idx(a)] == a.flatten())
    True
    """
    return np.where(np.ones(a.shape))


def filter_invalid_idx(idx, array, ignore_extra_dimensions=False):
    """ removes indices from idx which do not point to a location in array

    Parameters
    ----------
    idx : tuple of indices
        indices to verify
    array : ndarray
        array in which indices are tested
    ignore_extra_dimensions : bool
        if True, idx values for axes not in array are ignored, and do not
        prevent indices from being valid if the other axes are within array.
        if False, all indices in idx with more dimensions than a are strictly
        marked as invalid.

    Returns
    -------
    result : tuple of indices
        subset of idx with only valid values

    Example
    -------
    >>> a = np.ones((1,2,3))
    >>> b = np.ones((3,2,1))
    >>> idx = np.where(b==1)
    >>> idx
    (array([0, 0, 1, 1, 2, 2]), array([0, 1, 0, 1, 0, 1]), array([0, 0, 0, 0, 0, 0]))
    >>> filter_invalid_idx(idx, a)
    (array([0, 0]), array([0, 1]), array([0, 0]))
    """
    if not ignore_extra_dimensions:
        if len(array.shape) < len(idx):
            warnings.warn(
                "Indices point to more dimensions than exist in "
                "array. Default behavior is to flag them all as "
                "invalid. Set ignore_extra_dimensions to True to "
                "reduce strictness. Otherwise ensure that the "
                "dimensionalities match."
            )
            return tuple()
    maxdim = min(len(array.shape), len(idx))  # larger dimensions are irrelevant
    max_idx = np.array(array.shape[:maxdim]) - 1  # max index for each axis
    test = np.array(idx[:maxdim]).T
    is_valid = np.all(np.logical_and(test <= max_idx, test >= -1 - max_idx), axis=-1)
    return tuple(test[is_valid].T)


def sliding_window(a, size=3, fill_value=np.nan):
    """ Applies a sliding window of shape size^a_ndim to array a,

    TODO: generalize to a shape > 2x2
    returns an array of shape (size^a_ndim, a.shape)
    in which the first dimension corresponds to value of a for each flat index
    in the window."""
    # We generate pad_widths for each
    paddings = np.array(np.where(np.ones((size,) * a.ndim))).T[:, :, None] * [1, -1] + [
        0,
        size - 1,
    ]
    paddeds = np.array(
        [np.pad(a, pad, "constant", constant_values=fill_value) for pad in paddings]
    )
    return paddeds[:, size // 2 : -(size // 2), size // 2 : -(size // 2)]

def filter_if_out_of_bounds(indices_list, a):
    valid_mask = np.all(
            np.logical_and(indices_list >= 0, indices_list < a.shape),
            axis=-1,
            )
    return indices_list[valid_mask]

def batchsplit(x, batchsize, axis=0):
    """ similar to np.split, but takes a batch size instead of a 
    number of splits as input
    """
    if axis != 0:
        raise NotImplementedError
    n_full_sections =  int(np.floor(len(x) / batchsize))
    if n_full_sections == 0:
        return [x]
    indices = [n * batchsize for n in range(1, n_full_sections+1)]
    sections = np.split(x, indices, axis=axis)
    if len(sections[-1]) == 0:
        sections = sections[:-1]
    return sections

if __name__ == "__main__":
    import doctest

    doctest.testmod()
