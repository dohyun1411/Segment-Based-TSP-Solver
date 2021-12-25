# Segment Based TSP Solver
Segment based TSP solver, which is the combination of nearest neighbor algorithm and genetic algorithm.

For more about this solver, please refer [report](./report.pdf).

## Requirements
There are some requirements for this code, please install them by
```
pip install -r requirements.txt
```
or
```
pip install numpy
pip install matplotlib
pip install tqdm
pip install click
```

## How to Run
To solve a specific tsp file,
```
python tsp_solver.py [tsp file]
python tsp_solver.py a280.tsp
```

Then, it will create a ```solution.csv``` file. To see the content of it,
```
cat solution.csv
```

I provided two parameters:
* population (```-p```): you can control the population size
* fitness evaluations (```-f```): you can control the limit of the total number of fitness evaluations

You can use it like below.
```
python tsp_solver.py a280.tsp -t 50 -f 1000
```
