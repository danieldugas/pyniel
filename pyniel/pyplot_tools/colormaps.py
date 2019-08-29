import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import pyplot as plt

# same as Greys, but shrunk and cycled 10 times over the range
Greys_c10 = ListedColormap(plt.cm.Greys(np.mod(np.linspace(0., 10., 256), 1.)))

# Red to green through yellow
trafficlight = LinearSegmentedColormap.from_list("", ["tomato","yellow","springgreen"])

