import numpy as np

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
