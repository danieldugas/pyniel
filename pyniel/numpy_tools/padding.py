import numpy as np
from indexing import as_idx, filter_invalid_idx


def resize_with_zeros(a, shape):
    """ special case of resize_with_fill where fill_value is 0
    """
    return resize_with_fill(a, shape, fill_value=0)


def resize_with_fill(a, shape, fill_value=np.nan):
    """ returns a, resized to shape, filling blanks with zeros

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    shape : tuple of ints
        new shape
    fill_value : value, optional
        value to fill in newly created cells

    Returns
    -------
    result : ndarray
        Array of shape (shape)

    Notes
    -----
    the new shape must have the same amount of dimensions as the old,
    as the function can not know which form of reshaping makes sense otherwise.

    Example
    -------
    >>> a = np.ones((2,3))
    >>> a
    array([[1., 1., 1.],
           [1., 1., 1.]])
    >>> resize_with_fill(a, (3,4))
    array([[ 1.,  1.,  1., nan],
           [ 1.,  1.,  1., nan],
           [nan, nan, nan, nan]])
    """
    if len(shape) != len(a.shape):
        raise ValueError(
            "Expected new shape to have as many dimensions as "
            "old shape. Reshape to desired number of dimensions "
            "before calling resize_with_fill"
        )
    resized = np.full(shape, fill_value=fill_value)
    idx = filter_invalid_idx(as_idx(a), resized)
    resized[idx] = a[idx]
    return resized


def from_list_of_lists(l, fill_value=np.nan):
    """ fill a list of lists with fill_value until all sublists are same size,
    and turn the result into an np.array

    Parameters
    ----------
    l : list
        list of lists
    fill_value : value, optional
        value to fill in newly created cells

    Returns
    -------
    result : ndarray
        Array of shape (len(l), longest)

    Example
    -------
    >>> l = [[0,1,2],[0,1],[0]]
    >>> from_list_of_lists(l)
    array([[ 0.,  1.,  2.],
           [ 0.,  1., nan],
           [ 0., nan, nan]])
    >>> l = [[[0],[1],[2]],[[0],[1]],[[0]]]
    >>> from_list_of_lists(l, fill_value=[np.nan])
    array([[[ 0.],
            [ 1.],
            [ 2.]],
    <BLANKLINE>
           [[ 0.],
            [ 1.],
            [nan]],
    <BLANKLINE>
           [[ 0.],
            [nan],
            [nan]]])
    """
    if isinstance(l[0][0], list) and not isinstance(fill_value, list):
        errorstring = (
            "fill_value must be the same type as items in the "
            "sublist. Here items in the sublist are lists of size "
            "{}, for example: {}. fill_value must also be a list "
            "of size {}."
        ).format(str(len(l[0][0])), str(l[0][0]), str(len(l[0][0])))
        raise ValueError(errorstring)
    length = len(sorted(l, key=len, reverse=True)[0])
    return np.array([list(li) + [fill_value] * (length - len(li)) for li in l])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
