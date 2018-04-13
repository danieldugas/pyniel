import numpy as np
# Daniel Dugas, do-what-you-want-license

def equals(tuple_of_arrays):
    """ Apply == operator to multiple arrays
    """
    truth_value = np.ones(tuple_of_arrays[0].shape)
    if len(tuple_of_arrays) <= 1:
        raise Warning("Expected more than 1 array for logical comparison.")
        return truth_value
    for array1, array2 in zip(tuple_of_arrays[:-1],
                              tuple_of_arrays[1:]):
        truth_value = np.logical_and(truth_value, array1 == array2)
    return truth_value

def generic_is(a, axis, generic, generic_checker):
    """ generic function for ismin, ismax

    Produces same result as generic function, but as truth array instead of indices along axis
    Compatible with functions which identify a single value along a given axis (max, min, ...).

    e.g. ismin <- argmin(a, axis), returns an array of same shape as a
    where only min values along axis are True

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    axis : int or None
        Axis along which to apply generic. If None, the array is flattened before
        sorting.
    generic : ufunc
        function to apply to a, which outputs an array of shape a.shape.pop(axis)
        containing indices along the axis in a where the generic condition is satisfied.
    generic_check : ufunc
        function to apply to a for checking consistency of results

    Returns
    -------
    result : ndarray
        Array of type bool,

    Notes
    -----
    interestingly enough,
    a[isxxx(a,axis)] != np.min(a,axis).flatten()
    the values will be the same in both, but their order differs.
    for example with a = [[0, 0, 0],
                          [0, 0, 9],
                          [3, 6, 0]]
    a[isxx(a,axis=0)] = [9, 3, 6] and np.min(a,axis=0).flatten() = [3, 6, 9]
    (in a, 9 comes before 3 and 6 in the absolute index,
    but in np.min(a,axis), the matrix is squashed vertically, and 9 ends up after 3 and 6)
    I'm not sure wether this could cause buggy behavior, so be wary.

    Example
    -------
    >>> a = np.array([[0, 0, 0], [0, 0, 9], [3, 6, 0]])
    >>> a
    array([[0, 0, 0],
           [0, 0, 9],
           [3, 6, 0]])
    >>> generic_is(a, 0, np.argmin, np.min)
    array([[False, False, False],
           [False, False,  True],
           [True,   True, False]])
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
        #assert np.all(np.sort(a[index]) == np.sort(generic_checker(a, axis=axis).flatten()))
        result[index] = 1
    #assert np.all(np.sum(result, axis=axis) == 1)
    return result.astype(bool)

def ismin(a, axis=None):
    """ see generic_is()
    """
    return generic_is(a, axis=axis, generic=np.argmin, generic_checker=np.min)
def ismax(a, axis=None):
    """ see generic_is()
    """
    return generic_is(a, axis=axis, generic=np.argmax, generic_checker=np.max)
