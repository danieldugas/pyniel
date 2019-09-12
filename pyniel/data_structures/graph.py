import copy
import numpy as np
import pickle

class Node(object):
    def __init__(self, edges=set(), coords=np.array([None, None])):
        self.edges = edges
        self.coords = coords

    def __str__(self):
        return "("+str(self.edges)+", "+str(self.coords)+")"
    def __repr__(self):
        return "("+str(self.edges)+", "+str(self.coords)+")"

class Graph(object):
    """ Directed graph """
    def __init__(self):
        self.nodes = {} # { nodeid: node, ... }

    def __str__(self):
        result = "Graph:\n"
        for key in self.nodes:
            result += "  ({:>5.1f}, {:>5.1f}) {} ->".format(
                self.nodes[key].coords[0], self.nodes[key].coords[1], key)
            for e in self.nodes[key].edges:
                result += " {}".format(e)
            result += "\n"
        return result
    def __repr__(self):
        return self.nodes.__repr__()

    def get_new_id(self):
        if self.nodes:
            return sorted(list(self.nodes))[-1]+1
        else:
            return 0

    def as_2d_plot(self):
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
        return coords

    def reassign_ids(self):
        # map oldnodeid to new nodeid
        map_old_to_new = {}
        for i, key in enumerate(self.nodes):
            map_old_to_new[key] = i
        # apply map to graph
        newnodes = {}
        for key in self.nodes:
            newkey = map_old_to_new[key]
            newnodes[newkey] = self.nodes[key]
            oldedges = self.nodes[key].edges
            # clear edges and replace with newkeys
            newnodes[newkey].edges = set()
            for e in oldedges:
                newnodes[newkey].edges.add(map_old_to_new[e])
        self.nodes = newnodes

    def save_to_file(self, filepath):
        with open(filepath,"wb") as f:
            pickle.dump(self.nodes, f)
        print("Graph saved to {}".format(filepath))

    def restore_from_file(self, filepath):
        with open(filepath,"rb") as f:
            self.nodes = pickle.load(f)
        print("Graph loaded from {}".format(filepath))

