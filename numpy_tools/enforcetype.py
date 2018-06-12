import numpy as np


def asarray(x):
    """
  converts scalars or python lists to np.array
  None outputs to None
  """
    if x is None:
        return None
    y = np.array(x)
    if len(y.shape) == 0:
        y = np.array([x])
    return y
