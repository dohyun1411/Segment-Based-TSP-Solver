import copy, random
from tqdm import tqdm
import matplotlib.pyplot as plt
from graph import *


PMUT1 = 0.1
PMUT2 = 0.1
PMUT3 = 0.1
X = 0.25
Y = 0.25
Z = 0.25
INF = 1e9

def create_segment(nodes, start, ends):
    segment = []
    i = start
    while True:
        start_node = Node.group[i]
        nodes -= {start_node.id}
        segment.append(start_node)
        if not nodes:
            break
        i = min(nodes, key=lambda x: start_node.dist(Node.group[x]))
        if i in ends:
            break
    return segment

def create_pivot(dim, num_seg=None):
    if num_seg is None:
        num_seg = random.randint(2, dim)
    return random.sample(range(dim), k=num_seg)

def create_route(dim, pivot=None, num_seg=None):
    if pivot is None:
        pivot = create_pivot(dim, num_seg)
    route = []
    nodes = {i for i in range(dim)}
    tmp = copy.deepcopy(pivot)
    while nodes and tmp:
        start = tmp.pop()
        seg = create_segment(nodes, start, tmp)
        route.extend(seg)
    if len(route) != dim:
        return None
    check(route)
    return route

def crossover(p1, p2):
    l = min(len(p1), len(p2))
    r = random.randrange(l)
    child = p1[:r]
    for pp in p2[r:]:
        if pp in child: continue
        child.append(pp)
    return child

def mutate1(p, dim):
    remain = {i for i in range(dim)} - set(p)
    q = copy.deepcopy(p)
    for r in remain:
        if random.random() < PMUT1:
            i = random.randrange(len(q))
            q = q[:i] + [r] + q[i:]
    return q

def mutate2(p):
    q = copy.deepcopy(p)
    for pp in p:
        if len(q) <= 2:
            break
        if random.random() < PMUT2:
            q.remove(pp)
    return q

def mutate3(p, dim):
    remain = {i for i in range(dim)} - set(p)
    q = copy.deepcopy(p)
    for i in range(len(p)):
        if len(q) <= 2:
            break
        if random.random() < PMUT3:
            q[i] = random.choice(list(remain))
    return q

def mutate4(p):
    i, j = random.sample(range(len(p)), 2)
    s = min(i, j)
    e = max(i, j)
    sub = p[s:e]
    return p[:s] + sorted(sub) + p[e:]

def run(graph, pop_size, max_fit, verbose=0, plot=0):
    NPOP = pop_size # 50
    NBEST = int(NPOP * 0.3) # 15
    NCHILD = NPOP // 5 # 10
    NMUT = NPOP // 2 # 25
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlabel("Generatoin")
        plt.ylabel("Distance")
        plt.show(block=False)
        x = []
        y = []
        nny = []

        from nearest_neighbor import NN
        nn = NN(graph)
        nn.run(disable=not verbose)
        nnd = nn.dist
        if verbose:
            print(f"NN dist: {nnd:.2f}")

    dim = graph.dim
    pop = []
    for _ in range(NPOP):
        pivot = create_pivot(dim)
        pop.append((INF, pivot, None))

    gen = 1
    fit = 0
    while True:
        routes = []
        next_pop = []
        for _, pivot, _ in tqdm(pop, disable=not verbose):
            route = create_route(dim, pivot)
            if route is None: continue
            if route in routes: continue

            d = dist(route)
            fit += 1
            next_pop.append((d, pivot, route))
            routes.append(route)

        best = sorted(next_pop)[:min(NBEST, len(next_pop))]
        if verbose:  
            print(f"Generation {gen}, {fit} fitness calc (pop size: {len(next_pop)}) - dist: {best[0][0]:.2f}")
        if plot:
            x.append(gen)
            y.append(best[0][0])
            nny.append(nnd)
            l1, = ax.plot(x, y)
            l2, = ax.plot(x, nny)
            ax.legend([l1, l2], ["SEG-GA", "NN"])
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        gen += 1
        if fit > max_fit:
            break
        pop = []
        pop.extend(best)
        for _ in range(NCHILD):
            a, b = random.sample(best, k=2)
            child = crossover(a[1], b[1])
            pop.append((INF, child, None))
        mutated = []
        for _ in range(NMUT + NPOP - len(next_pop)):
            p = random.choice(pop)[1]
            r = random.random()
            if r < X:
                m = mutate1(p, dim)
            elif r < X + Y:
                m = mutate2(p)
            elif r < X + Y + Z:
                m = mutate3(p, dim)
            else:
                m = mutate4(p)
            mutated.append((INF, m, None))
        pop.extend(mutated)
    
    return best[0][2]