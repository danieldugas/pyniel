import numpy as np

def resize_with_zeros(a, shape):
    flat = np.zeros(shape).flatten()
    flat_a = a.flatten()
    flat[np.arange(len(flat_a))] = flat_a
    return flat.reshape(shape)
