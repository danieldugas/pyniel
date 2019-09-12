from __future__ import print_function
import numpy as np
import copy
import matplotlib.pyplot as plt

from pyniel.data_structures.graph import Graph, Node


class GraphCreatorGui(object):
    def __init__(self, figure=None, click_radius=1, debug=False):
        self.debug = debug
        self.click_radius = click_radius
        self.fig = figure
        # state variables
        self.graph = Graph()
        self.selected = None


    def run(self):
        if self.fig is None:
            self.fig = plt.figure()
        self.cp, = plt.plot([], [], 'o--', color='red')
        self.sp, = plt.plot([], [], 's--')
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.update_figure()
        plt.show()

    def onclick(self, event):
        is_leftclick = event.button == 1
        is_middleclick = event.button == 2
        is_rightclick = event.button == 3
        ix, iy = event.xdata, event.ydata
        if ix is None or iy is None:
            if self.debug:
                print("invalid click")
                print(event)
            return
        # Find if an existing node is under the click
        clickednode = None
        for nodeid in self.graph.nodes:
            if np.linalg.norm(self.graph.nodes[nodeid].coords - np.array([ix, iy])) < self.click_radius:
                clickednode = nodeid
                break

        newid = None

        if is_leftclick:
            if self.selected is None:
                if clickednode is not None:
                    # select clicked node
                    if self.debug:
                        print("Selecting node")
                    self.selected = clickednode
                else:
                    if self.debug:
                        print("Adding new node and selecting it")
                    # add new node
                    newid = self.graph.get_new_id()
                    self.graph.nodes[newid] = Node(edges=set(), coords=np.array([ix, iy]))
                    self.selected = newid
            else:
                if clickednode is not None:
                    if clickednode == self.selected:
                        if self.debug:
                            print("Unselecting node")
                        # unselect clicked node
                        self.selected = None
                    else:
                        if self.debug:
                            print("Toggling edge")
                        if clickednode in self.graph.nodes[self.selected].edges:
                            # remove link between self.selected and clickednode
                            self.graph.nodes[self.selected].edges.remove(clickednode)
                            self.graph.nodes[clickednode].edges.remove(self.selected)
                        else:
                            # add link between self.selected and clickednode
                            self.graph.nodes[self.selected].edges.add(clickednode)
                            self.graph.nodes[clickednode].edges.add(self.selected)
                else:
                    if self.debug:
                        print("Adding new node connected to selected")
                    # add new node connected to self.selected
                    newid = self.graph.get_new_id()
                    self.graph.nodes[newid] = Node(edges={self.selected}, coords=np.array([ix, iy]))
                    # add link between self.selected and new node
                    self.graph.nodes[self.selected].edges.add(newid)
                    # update selection
                    self.selected = newid
        elif is_rightclick:
                if clickednode is not None:
                    if self.debug:
                        print("Removing node")
                    # remove clicked node
                    for edge in self.graph.nodes[clickednode].edges:
                        self.graph.nodes[edge].edges.remove(clickednode)
                    self.graph.nodes.pop(clickednode)
                    self.selected = None
                else:
                    if self.debug:
                        print("Unselecting node")
                    self.selected = None
        elif is_middleclick:
                if clickednode is not None:
                    if self.debug:
                        print("Selecting node")
                    self.selected = clickednode

        if self.debug:
            print(event)
            print("newid:", newid)
            print("selected:", self.selected)
            print("clicked:", clickednode)
            print("nodes:", self.graph.nodes)
            print("------------------------------")

        self.update_figure()

        if False:
            self.fig.canvas.mpl_disconnect(cid)

    def update_figure(self):
        coords = self.graph.as_2d_plot()
        # visualize nodes and edges
        if coords:
            X = np.array(coords)[:,0]
            Y = np.array(coords)[:,1]
        else: 
            X = []
            Y = []
        self.cp.set_data(X,Y)
        # visualize selection
        if self.selected is not None:
            X = [self.graph.nodes[self.selected].coords[0]]
            Y = [self.graph.nodes[self.selected].coords[1]]
        else:
            X = []
            Y = []
        self.sp.set_data(X,Y)
        # draw
        self.fig.canvas.draw()

def test_graph_creator_gui():
    """ Feed random mouse inputs to test the graph building algorithm """
    fig = plt.figure()
    plt.plot([0, 0, 1, 1, 0], [0,1, 1, 0, 0], 'k-')
    gcg = GraphCreatorGui(figure=fig, click_radius=0.05)
    gcg.cp, = plt.plot([], [], 'o--', color='red')
    gcg.sp, = plt.plot([], [], 's--')
    cl, = plt.plot([], [], 'x', color='black')
    class RandomEvent(object):
        def __init__(self):
            self.button = np.random.randint(4)
            self.xdata = np.random.rand()
            self.ydata = np.random.rand()
    for i in range(1000):
        event = RandomEvent()
        cl.set_data([event.xdata], [event.ydata])
        gcg.onclick(event)
        plt.pause(0.001)
    return 1

if __name__=="__main__":
    test_graph_creator_gui()

