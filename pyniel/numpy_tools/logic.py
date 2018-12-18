import warnings
import numpy as np

""" Logic tools
Daniel Dugas
"""


def equals(tuple_of_arrays):
    """ Apply == operator to multiple arrays.
    Cells in output are true where array1 == array2 == array3 == ...

    Parameters
    ----------
    tuple_of_arrays : tuple
        contains ndarrays on which the function performs equality checks.

    Returns
    -------
    result : ndarray
        Array of type bool, with the same shape as the largest array in tuple,
        if broadcasting is successful.

    Example
    -------
    >>> A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> B = np.array([[9, 1, 9], [9, 9, 1], [0, 9, 0]])
    >>> C = np.array([[6, 1, 6], [6, 9, 1], [6, 0, 0]])
    >>> equals((A, B, C))
    array([[False,  True, False],
           [False, False,  True],
           [False, False,  True]], dtype=bool)
    >>> D = np.array([0, 0 ,0])
    >>> equals((A, B, C, D))
    array([[False, False, False],
           [False, False, False],
           [False, False,  True]], dtype=bool)
    """
    truth_value = np.ones(tuple_of_arrays[0].shape)
    if len(tuple_of_arrays) <= 1:
        warnings.warn("Expected more than 1 array for logical comparison.")
        return truth_value
    for array1, array2 in zip(tuple_of_arrays[:-1], tuple_of_arrays[1:]):
        truth_value = np.logical_and(truth_value, array1 == array2)
    return truth_value


def generic_is(a, axis, generic, generic_checker=None):
    """ generic function for ismin, ismax

    Produces same result as generic function, but as truth array instead of
    indices along axis. Compatible with functions which identify a single value
    along a given axis (max, min, ...).

    e.g. ismin <- argmin(a, axis), returns an array of same shape as a
    where only min values along axis are True

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    axis : int or None
        Axis along which to apply generic. If None, the array is flattened
        before sorting.
    generic : ufunc
        function to apply to a, which outputs an array of shape
        a.shape.pop(axis) containing indices along the axis in a where the
        generic condition is satisfied.
    generic_checker : ufunc, optional
        function to apply to a for checking consistency of results.

    Returns
    -------
    result : ndarray
        Array of type bool,

    Notes
    -----
    interestingly enough,
    a[isxxx(a,axis)] is not equal to np.xxx(a,axis).flatten().
    the values will be the same in both, but their order differs.
    (see relevant example)
    I'm not sure wether this could cause buggy behavior, so be wary.

    Example
    -------
    >>> a = np.array([[0, 0, 0], [0, 0, 9], [3, 6, 0]])
    >>> a
    array([[0, 0, 0],
           [0, 0, 9],
           [3, 6, 0]])
    >>> generic_is(a, 0, np.argmax, np.max)
    array([[False, False, False],
           [False, False,  True],
           [ True,  True, False]], dtype=bool)

    as mentioned in the notes:
    >>> a[ismax(a,axis=0)]
    array([9, 3, 6])
    >>> np.max(a,axis=0).flatten()
    array([3, 6, 9])

    In a, 9 comes before 3 and 6 in the flattened index, but in np.max(a,axis),
    the matrix is squashed vertically, and 9 ends up after 3 and 6.
    """
    result = np.zeros(a.shape)
    if axis is None:
        result[np.unravel_index(generic(a, axis=None), a.shape)] = 1
    else:
        targets = generic(a, axis=axis)
        # Transform the indices along axis into indices along all a-axes
        # Why is this so ugly, is there no function for this?
        index = list(np.unravel_index(np.arange(len(targets.flatten())), targets.shape))
        index.insert(axis, targets.flatten())
        index = tuple(index)
        if generic_checker is not None:
            assert np.all(
                np.sort(a[index]) == np.sort(generic_checker(a, axis=axis).flatten())
            )
        result[index] = 1
    if generic_checker is not None:
        assert np.all(np.sum(result, axis=axis) == 1)
    return result.astype(bool)


def ismin(a, axis=None):
    """ see generic_is()
    """
    return generic_is(a, axis=axis, generic=np.argmin)


def ismax(a, axis=None):
    """ see generic_is()
    """
    return generic_is(a, axis=axis, generic=np.argmax)


def absmaximum(a, b):
    """ Elementwise maximum values from
    two equally shaped arrays a and b

    The value picked for each element is that with
    the largest absolute value.

    If the absolute values for an element are equal but
    of different signs, the sign for a is picked.

    Example
    -------
    >>> a = np.array([[-1, 0, 1], [ 2, -2, 0], [0,  3, -3]])
    >>> b = np.array([[ 3, 0,-3], [-1,  1, 0], [0, -2,  2]])
    >>> absmaximum(a, b)
    array([[ 3,  0, -3],
           [ 2, -2,  0],
           [ 0,  3, -3]])
    """
    return a * (np.abs(a) >= np.abs(b)) + b * (np.abs(b) > np.abs(a))

def absminimum(a, b):
    """ Elementwise minimum values from
    two equally shaped arrays a and b

    The value picked for each element is that with
    the smallest absolute value.

    If the absolute values for an element are equal but
    of different signs, the sign for a is picked.

    Example
    -------
    >>> a = np.array([[-1, 0, 1], [ 2, -2, 0], [0,  3, -3]])
    >>> b = np.array([[ 3, 0,-3], [-1,  1, 0], [0, -2,  2]])
    >>> absminimum(a, b)
    array([[-1,  0,  1],
           [-1,  1,  0],
           [ 0, -2,  2]])
    """
    return a * (np.abs(a) <= np.abs(b)) + b * (np.abs(b) < np.abs(a))

if __name__ == "__main__":
    import doctest

    np.random.seed(42)
    doctest.testmod()
