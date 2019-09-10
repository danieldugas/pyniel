import matplotlib.pyplot as plt
import numpy as np

# copied from map2d.py in danieldugas/pymap2d
def gridshow(*args, **kwargs):
    """ utility function for showing 2d grids in matplotlib,
    wrapper around pyplot.imshow

    use 'extent' = [-1, 1, -1, 1] to change the axis values """
    from matplotlib import pyplot as plt
    if not 'origin' in kwargs:
        kwargs['origin'] = 'lower'
    if not 'cmap' in kwargs:
        kwargs['cmap'] = plt.cm.Greys
    return plt.imshow(*(arg.T if i == 0 else arg for i, arg in enumerate(args)), **kwargs)

def plotstep(*args, **kwargs):
    if 'where' not in kwargs:
        kwargs['where'] = 'mid'
    # add x dim if missing
    x = None
    y = None
    newargs = []
    for i, arg in enumerate(args):
        if type(arg) in [np.ndarray, list]:
            if i == 0:
                x = arg
                continue
            if i == 1:
                y = arg
                continue
        newargs.append(arg)
    if y is None and x is not None:
        y = x
        x = np.arange(len(y))
        args = [x, y] + newargs
    # plot
    plt.step(*args, **kwargs)

