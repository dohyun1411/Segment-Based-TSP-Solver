import copy
import numpy as np


class Node:
    group = {}

    def __init__(self, id, x, y):
        self.id = int(id) - 1
        self.x = float(x)
        self.y = float(y)
        self.xy = np.array([self.x, self.y])
        Node.group[self.id] = self
    
    def __repr__(self):
        return f"{self.id}"
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __gt__(self, other):
        return self
    
    def dist(self, node):
        return np.linalg.norm(self.xy - node.xy)
    

class Graph:
    def __init__(self, graph=None):
        if graph:
            self.dim = graph.dim
            self.nodes = copy.deepcopy(graph.nodes)
        else:
            self.dim = 0
            self.nodes = []
    
    @property
    def cycle(self):
        return self.nodes + [self.nodes[0]]

    def load(self, filenmae):
        f = open(filenmae, 'r')
        d = 0
        for line in f:
            if 0 < d <= self.dim:
                id, x, y = line.split()
                self.nodes.append(Node(id, x, y))
                d += 1
            elif "DIMENSION" in line:
                self.dim = int(line.split(":")[-1])
            elif "NODE_COORD_SECTION" == line.strip():
                d = 1


def check(route):
    tmp = [x.id for x in route]
    assert len(tmp) == len(set(tmp))

def dist(route):
    check(route)
    res = 0
    for i in range(len(route) - 1):
        cur = route[i]
        next = route[i + 1]
        res += cur.dist(next)
    res += route[-1].dist(route[0])
    return res