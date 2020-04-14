import numpy as np

def chunks_split(array, chunksize, axis=0):
    """
    split array into equally sized chunks of size chunksize

    outputs:
    -----
    chunks: list of equally shaped chunks, which have similar shape to input array except for axis, which has size chunksize
    remainder: similar shape to input array except for axis, which has size in range [0,chunksize]
    """
    n_chunks = array.shape[axis] // chunksize
    splittable, remainder  = np.split(array, [n_chunks*chunksize], axis=axis)
    chunks = np.split(splittable, n_chunks, axis=axis)
    return chunks, remainder

