import numpy as np

def one_hot(labels, depth, on_value=1.0, off_value=0.0):
    """ Takes an numpy array and replaces the last axis with a one_hot encoding

    Args:
            labels (np.ndarray): Input array
            depth (int): amount of classes in one_hot encoding

    Returns:
            (np.ndarray) one_hot encoding with shape labels.shape + (depth,)

    The inverse of one_hot is np.argmax(one_hot_array, axis=-1)
    """
    y = np.array(labels)
    one_hot = np.ones(y.shape + (depth,)) * off_value
    one_hot[..., np.arange(y.shape[-1]), y.astype(int)] = on_value
    return one_hot
