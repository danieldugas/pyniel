from matplotlib import pyplot as plt

def make_legend_pickable(legend, lines):
    """ Allows clicking the legend to toggle line visibility
    arguments:
        legend: the legend object (output of plt.legend())
        lines: list of line objects corresponding to legend items.
               should be of same length as legend.get_lines()
               Note: line objects can be anything which has a set_visible(bool is_visible) method
    """
    lineobjects = {}
    legenditems = legend.get_lines()
    for item, line in zip(legenditems, lines):
        item.set_picker(True)
        item.set_pickradius(10)
        lineobjects[item] = line
    def on_click_legenditem(event):
        legenditem = event.artist
        is_visible = legenditem.get_visible()
        lineobjects[legenditem].set_visible(not is_visible)
        legenditem.set_visible(not is_visible)
        plt.gcf().canvas.draw()
    plt.connect('pick_event', on_click_legenditem)
