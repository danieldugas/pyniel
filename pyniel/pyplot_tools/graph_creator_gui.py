from __future__ import print_function
import numpy as np
import copy
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, edges=set(), coords=np.array([None, None])):
        self.edges = edges
        self.coords = coords

    def __str__(self):
        return "("+str(self.edges)+", "+str(self.coords)+")"
    def __repr__(self):
        return "("+str(self.edges)+", "+str(self.coords)+")"

class GraphCreatorGui(object):
    def __init__(self, figure=None, click_radius=1, debug=False):
        self.debug = debug
        self.click_radius = click_radius
        self.fig = figure
        # state variables
        self.nodes = {} # {id: (edges, coords) }
        self.freeids = {0}
        self.selected = None

    def run(self):
        if self.fig is None:
            self.fig = plt.figure()
        self.cp, = plt.plot([], [], 'o--', color='red')
        self.sp, = plt.plot([], [], 's--')
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
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
        for nodeid in self.nodes:
            if np.linalg.norm(self.nodes[nodeid].coords - np.array([ix, iy])) < self.click_radius:
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
                    newid = self.freeids.pop() if self.freeids else sorted(list(self.nodes))[-1]+1
                    self.nodes[newid] = Node(edges=set(), coords=np.array([ix, iy]))
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
                        if clickednode in self.nodes[self.selected].edges:
                            # remove link between self.selected and clickednode
                            self.nodes[self.selected].edges.remove(clickednode)
                            self.nodes[clickednode].edges.remove(self.selected)
                        else:
                            # add link between self.selected and clickednode
                            self.nodes[self.selected].edges.add(clickednode)
                            self.nodes[clickednode].edges.add(self.selected)
                else:
                    if self.debug:
                        print("Adding new node connected to selected")
                    # add new node connected to self.selected
                    newid = self.freeids.pop() if self.freeids else sorted(list(self.nodes))[-1]+1
                    self.nodes[newid] = Node(edges={self.selected}, coords=np.array([ix, iy]))
                    # add link between self.selected and new node
                    self.nodes[self.selected].edges.add(newid)
                    # update selection
                    self.selected = newid
        elif is_rightclick:
                if clickednode is not None:
                    if self.debug:
                        print("Removing node")
                    # remove clicked node
                    for edge in self.nodes[clickednode].edges:
                        self.nodes[edge].edges.remove(clickednode)
                    self.nodes.pop(clickednode)
                    self.freeids.add(clickednode)
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
            print("freeids:", self.freeids)
            print("newid:", newid)
            print("selected:", self.selected)
            print("clicked:", clickednode)
            print("nodes:", self.nodes)
            print("------------------------------")

        # list of coordinates from graph edges by walking through each edge
        tempnodes = copy.deepcopy(self.nodes)
        coords = []
        while tempnodes:
            coords.append([np.nan, np.nan])
            start = list(tempnodes)[0]
            coords.append(tempnodes[start].coords)
            if tempnodes[start].edges:
                end = tempnodes[start].edges.pop()
                tempnodes[end].edges.remove(start)
                coords.append(tempnodes[end].coords)
            else:
                tempnodes.pop(start)
        # visualize
        if coords:
            X = np.array(coords)[:,0]
            Y = np.array(coords)[:,1]
        else: 
            X = []
            Y = []
        self.cp.set_data(X,Y)

        if self.selected is not None:
            X = [self.nodes[self.selected].coords[0]]
            Y = [self.nodes[self.selected].coords[1]]
        else:
            X = []
            Y = []
        self.sp.set_data(X,Y)
        self.fig.canvas.draw()

        if False:
            self.fig.canvas.mpl_disconnect(cid)

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

