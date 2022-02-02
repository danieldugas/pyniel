from matplotlib import pyplot as plt

def set_visible(lines, visible):
    if isinstance(lines, list):
        for line in lines:
            line.set_visible(visible)
    else:
        lines.set_visible(visible)

def get_visible(lines):
    if isinstance(lines, list):
        for line in lines:
            return line.get_visible()
        return False
    else:
        return lines.get_visible()

def make_legend_pickable(legend, lines):
    """ Allows clicking the legend to toggle line visibility
    arguments:
        legend: the legend object (output of plt.legend())
        lines: list of list of line objects corresponding to legend items.
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
        is_visible = get_visible(legenditem)
        set_visible(lineobjects[legenditem], not is_visible)
        set_visible(legenditem, not is_visible)
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
        def plot_flow_edge_model(vcx, vcy, mu, rho, current_neighbor):
            nudge_case = False
            # apply model
            neighbor_offsets = np.array([[0, 1], [1, 0], [ 0,-1], [-1, 0], [1, 1], [1,-1], [-1, 1], [-1,-1]])
            n = current_neighbor
            oi = neighbor_offsets[n, 0]
            oj = neighbor_offsets[n, 1]
            edge_length = np.sqrt(oi**2 + oj**2)
            edge_x = - oi / edge_length
            edge_y = - oj / edge_length
            vc_norm2 = vcx*vcx + vcy*vcy
            vc_norm = np.sqrt(vc_norm2)
            delta_max = 1. / (rho * mu)
            # velocity along edge which minimizes friction
            l_star = edge_x*vcx + edge_y*vcy # dot product
            delta_min2 = vc_norm2 - l_star*l_star
            l_max = -1.
            nudge_vel = 0.1
            lbda_0 = (nudge_vel + vc_norm) * mu * rho
            l_nudge = min(nudge_vel / lbda_0, 0.1)
            if delta_min2 <= delta_max*delta_max: # no intersection between friction-constraint circle and line
                l_max = l_star + np.sqrt(delta_max*delta_max - delta_min2)
            if l_max <= 0: # not allowed to move according to hard constraint
                cos_theta = l_star / vc_norm
                if cos_theta >= np.cos(np.pi/4.): # check if the solution is missed because of discretization
                    l_max = l_star
            # nudge case # TODO: sqrt -> csqrt, predeclare missing 
            l_opt = max(l_max, l_nudge)
            print("optimal velocity: {}".format(l_opt))

            from matplotlib import pyplot as plt
            plt.figure("vc")
            plt.cla()
            for oi, oj in neighbor_offsets:
                plt.plot([0, oi], [0, oj], color='lightgrey')
            plt.plot([0, vcx], [0, vcy], color='yellow')
            plt.plot([0, edge_x], [0, edge_y], color='black')
            plt.gca().add_artist(plt.Circle((vcx, vcy), delta_max, fill=False, color='yellow'))
            plt.gca().add_artist(plt.Circle((edge_x * l_star, edge_y * l_star), 0.02, fill=False, color='black'))
            plt.gca().add_artist(plt.Circle((edge_x * l_opt, edge_y * l_opt), 0.04, fill=False, color='red' if nudge_case else 'black'))
            plt.gca().add_artist(plt.Circle((edge_x * l_max, edge_y * l_max), 0.02, fill=True, color='black'))
            plt.gca().add_artist(plt.Circle((edge_x * l_nudge, edge_y * l_nudge), 0.02, fill=True, color='red'))
            plt.axis('equal')
        plot_flow_edge_model(self.vcx, self.vcy, self.mu, self.rho, self.current_neighbor)
        plt.pause(0.1)

