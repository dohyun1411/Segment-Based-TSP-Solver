import copy, random
from tqdm import tqdm
from graph import *


class NN:
    def __init__(self, graph, route=None):
        self.graph = copy.deepcopy(graph)
        self.dim = self.graph.dim
        self.route = []
        self.dist = 0.0
        if route:
            self.route = copy.deepcopy(route)
            for i in range(self.dim - 1):
                self.dist += route[i].dist(route[i + 1])
            self.dist += self.route[-1].dist(self.route[0])
    
    def run(self, i=None, disable=False):
        if i is None:
            i = random.randrange(0, self.dim)
        # while True:
        for _ in tqdm(range(self.dim), disable=disable):
            start = self.graph.nodes.pop(i)
            self.route.append(start)
            if not self.graph.nodes:
                break
            i, dist = min(enumerate(start.dist(node) for node in self.graph.nodes), key=lambda x: x[1])
            self.dist += dist
        self.dist += self.route[-1].dist(self.route[0])


class PNN(NN):
    def __init__(self, graph, route=None):
        super(PNN, self).__init__(graph, route)
    
    def run(self, i=None, disable=False):
        if i is None:
            i = random.randrange(0, self.dim)
        for _ in tqdm(range(self.dim), disable=disable):
            start = self.graph.nodes.pop(i)
            self.route.append(start)
            if not self.graph.nodes:
                break
            dist = [start.dist(node) for node in self.graph.nodes]
            weight = [1 / (dist[i] + 1e-5) for i in range(len(self.graph.nodes))]
            i = random.choices(range(len(self.graph.nodes)), weight)[0]
            self.dist += dist[i]
        self.dist += self.route[-1].dist(self.route[0])


if __name__ == '__main__':
    # filename = "./a280.tsp" # 2579
    # filename = "./berlin52.tsp" # 7542
    filename = "./tsplib/rl11849.tsp" # [920847,923473]
    # filename = "./vm1084.tsp" # 239297
    graph = Graph()
    graph.load(filename)

    nn = NN(graph)
    i = 279
    nn.run()
    route = nn.route

    import csv
    with open("solution.csv", 'w', newline='') as f:
        wr = csv.writer(f)
        for node in route:
            wr.writerow([node.id + 1])
    # nn2 = NN(graph)
    # nn2.run(i, True)
    # print(nn2.route, nn2.dist)



    # C = PNN
    # nn = C(graph)
    # nn.run(disable=True)
    # print(list(reversed(nn.route)), nn.dist)
    # i = nn.route[-1].id - 1
    # nn2 = C(graph)
    # nn2.run(i=i, disable=True)
    # print(nn2.route, nn2.dist)

    # nn2.route.reverse()
    # new_route = []
    # i = graph.dim - 1
    # while len(new_route) < graph.dim:
    #     if random.random() < 0.5: # 낮아야할 듯
    #         while nn.route[i] in new_route:
    #             i -= 1
    #             if i < 0:
    #                 i = graph.dim - 1
    #         new_route.append(nn.route[i])
    #     else:
    #         while nn2.route[i] in new_route:
    #             i -= 1
    #             if i < 0:
    #                 i = graph.dim - 1
    #         new_route.append(nn2.route[i])
    # check(new_route)
    # nn3 = C(graph, new_route)
    # print(nn3.route, nn3.dist)
    
    # nn3 = C(graph, route=nn2.route)
    # r = random.randrange(graph.dim)
    # tmp = nn3.route[0]
    # nn3.route[0] = nn3.route[r]
    # nn3.route[r] = tmp
    # check(nn3.route)
    # nn4 = C(graph, route=nn3.route)
    # print(nn4.route, nn4.dist)


    # for _ in range(50):
    #     nn = NN(graph)
    #     nn.run(True)
    #     check(nn.route)
    #     print(nn.dist)
    # print("===================")
    # for _ in range(5):
    #     nn = PNN(graph)
    #     nn.run(True)
    #     check(nn.route)
    #     print(nn.dist)

