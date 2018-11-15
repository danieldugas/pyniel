import warnings
import numpy as np
from matplotlib import pyplot as plt

""" Pyplot tools
Daniel Dugas
"""
def gridshow(*args, **kwargs):
    """ Wrapper for pyplot.imshow, for use with non-image 2D arrays.

    As opposed to imshow, in gridshow:
    first dim of the array -> x axis of the plot
    second dim of the array -> y axis of the plot
    origin='lower' (doesn't flip y axis)

    All other kwargs and args are passed on as-is
    """
    if not 'origin' in kwargs:
        kwargs['origin'] = 'lower'
    return plt.imshow(*(arg.T if i == 0 else arg for i, arg in enumerate(args)), **kwargs)

