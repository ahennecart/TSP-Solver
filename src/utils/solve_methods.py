import numpy as np
from concorde.tsp import TSPSolver
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_lin_kernighan

from src.utils.search_methods import local_search, greedy_search


ACCEPTED_METHODS = [
    'exact',
    'TSP_Solver',
    'local_search',
    'lin_kernighan',
    'Concorde'
]

def solve_small_tsp(method, reduce_distance_matrix=None, position_node=None):
    """Method used to solve a reduced TSP"""
    if method not in ACCEPTED_METHODS:
        raise NotImplementedError(f"The {method} is not implemented, try an other one in {ACCEPTED_METHODS}")
    if method == "exact" and reduce_distance_matrix is not None:
        tour_nodes, _ = solve_tsp_dynamic_programming(reduce_distance_matrix)  # exact TSP solver (take more time)
    elif method == "TSP_Solver" and reduce_distance_matrix is not None:
        tour_nodes, _ = solve_tsp_simulated_annealing(reduce_distance_matrix)  # Approximation
    elif method == "local_search" and reduce_distance_matrix is not None:
        tour_nodes = greedy_search(reduce_distance_matrix)
        tour_nodes, _ = local_search(reduce_distance_matrix, tour_nodes, 0, full=True)
    elif method == "lin_kernighan" and reduce_distance_matrix is not None:
        tour_nodes, _ = solve_tsp_lin_kernighan(reduce_distance_matrix)
    elif method == "Concorde" and position_node is not None:
        xs, ys = position_node.T[0], position_node.T[1]
        solver = TSPSolver.from_data(xs, ys, norm="GEOM")
        solution = solver.solve()
        tour_nodes = solution.tour
    else:
        raise ValueError(f"The value for reduce_distance_matrix position_node is None and can not be used with the selected method")
    return tour_nodes