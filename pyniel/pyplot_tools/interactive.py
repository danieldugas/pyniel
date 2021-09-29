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

class InteractivePlot(object):
    """
    A simple example of a clickable plot which changes based on where it is clicked
    """
    def __init__(self):
        # draggables
        self.vcx = 0.0
        self.vcy = -0.15
        self.mu = 5.
        self.rho = 1.
        self.current_neighbor = 2

        self.update_plot()
        minvx = -2.
        maxvx = 2.
        minvy = -2.
        maxvy = 2.
        plt.axis([minvx, maxvx, minvy, maxvy])
        def onclick(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                self.vcx = event.xdata
                self.vcy = event.ydata
            elif event.button == 3:
                delta = np.sqrt((event.xdata - self.vcx)**2 + (event.ydata - self.vcy)**2)
                self.rho = 1. / delta / self.mu
                print("rho={}".format(self.rho))
        plt.figure("vc")
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        while True:
            self.update_plot()

    def update_plot(self):
        plot_flow_edge_model(self.vcx, self.vcy, self.mu, self.rho, self.current_neighbor)
        plt.pause(0.1)

