import os, csv, click
from seg_ga import *


def write(route):
    with open("solution.csv", 'w', newline='') as f:
        wr = csv.writer(f)
        for node in route:
            wr.writerow([node.id + 1])

@click.command()
@click.argument('filename', default='ch150.tsp')
@click.option('--pop', '-p', default=50)
@click.option('--max_fit', '-f', default=1000)
def main(filename, pop, max_fit):
    filename = os.path.join("tsplib", filename)
    graph = Graph()
    graph.load(filename)
    route = run(graph, pop_size=pop, max_fit=max_fit, verbose=1, plot=0)
    write(route)


if __name__ == "__main__":
    main()