"""File with all the search method used by the models"""

import numpy as np
from typing import DefaultDict
from copy import copy, deepcopy
import math

from python_tsp.distances import euclidean_distance_matrix

# Greedy search with real coef:
def greedy_search(distance_matrix):
    tour = [-1] * len(distance_matrix)
    visited_node = DefaultDict(int)
    
    actual = 0
    tour[0] = actual
    visited_node[0] = 1
    best = None
    
    for i in range(1, len(tour)):
        minimum = math.inf
        for j in range(len(distance_matrix)):
            if distance_matrix[actual][j] < minimum and visited_node[j] == 0:
                best = j
                minimum = distance_matrix[actual][j]
        tour[i] = best
        visited_node[best] = 1
        actual = best
    return tour

# Greedy search :
def probalistic_greedy_search(distance_matrix, real_distance_matrix):
    tour = [-1] * len(distance_matrix)
    visited_node = DefaultDict(int)
    
    actual = 0
    tour[0] = actual
    visited_node[0] = 1
    best = None
    
    for i in range(1, len(tour)):
        minimum = math.inf
        for j in range(len(distance_matrix)):
            if distance_matrix[actual][j] >= 0.99 and real_distance_matrix[actual][j]+1 < minimum and visited_node[j] == 0:
                best = j
                minimum = real_distance_matrix[actual][j]+1
            elif distance_matrix[actual][j] < 1 and distance_matrix[actual][j] < minimum and visited_node[j] == 0:
                best = j
                minimum = distance_matrix[actual][j]
        tour[i] = best
        visited_node[best] = 1
        actual = best
    return tour

# Greedy search with the not inverted coef mat :
def probalistic_greedy_search_2(coef_matrix, real_distance_matrix):
    tour = [-1] * len(coef_matrix)
    visited_node = DefaultDict(int)
    
    actual = 0
    tour[0] = actual
    visited_node[0] = 1
    best = None
    
    for i in range(1, len(tour)):
        maximal = -1
        for j in range(len(coef_matrix)):
            if coef_matrix[actual][j] == 0 and 0-real_distance_matrix[actual][j] > maximal and visited_node[j] == 0:
                best = j
                maximal = 0-real_distance_matrix[actual][j]
            elif coef_matrix[actual][j] > 0 and coef_matrix[actual][j] > maximal and visited_node[j] == 0:
                best = j
                maximal = coef_matrix[actual][j]
        tour[i] = best
        visited_node[best] = 1
        actual = best
    return tour

# Greedy search in a not fully connected graph :
def probalistic_greedy_search_not_connected(coef_matrix):
    tour = [None] * len(coef_matrix)  # [node, previous_visited_node, node_leading_to_dead_end]
    for i in range(len(coef_matrix)):
        tour[i] = [-1, None, None]
    tour[0] = [0, DefaultDict(int), DefaultDict(int)]
    tour[0][1][0] = 1
    i = 0
    actual = 0
    while i < len(coef_matrix)-1:
        best = -1
        maximal = 0
        for j in range(len(coef_matrix)):
            if coef_matrix[actual][j] > maximal and tour[i][1][j] == 0 and tour[i][2][j] == 0:
                best = j
                maximal = coef_matrix[actual][j]
        # If we have found a next node
        if best != -1:
            tour[i+1] = [best, deepcopy(tour[i][1]), DefaultDict(int)]
            tour[i+1][1][best] = 1
            actual = best
            i += 1
        # If we have found an dead-end
        else:
            if i == 0:
                raise RuntimeError("The graph is not in one part (i==0)")
            i -= 1
            tour[i][2][tour[i+1][0]] = 1
            tour[i+1] = [-1, None, None]
            actual = tour[i][0]
    if tour[-1][0] == -1:
        raise RuntimeError("The graph is not in one part (tour not complete)")
    return [x[0] for x in tour]

# Greedy search :
def probalistic_greedy_search_mult(distance_matrix, real_distance_matrix):
    tour = [-1] * len(distance_matrix)
    visited_node = DefaultDict(int)
    
    actual = 0
    tour[0] = actual
    visited_node[0] = 1
    
    for i in range(1, len(tour)):
        minimum = math.inf
        for j in range(len(distance_matrix)):
            if (distance_matrix[actual][j]*real_distance_matrix[actual][j]) < minimum and visited_node[j] == 0:
                next = j
                minimum = distance_matrix[actual][j]*real_distance_matrix[actual][j]
        tour[i] = next
        visited_node[next] = 1
        actual = next
    return tour

# Beam search :
def probalistic_beam_search(distance_matrix, real_distance_matrix, widht):
    tour = [0] * widht
    # We put the n second node here to ensure that they are differents
    firstDict = DefaultDict(int)
    firstDict[0] = 1
    for i in range(widht):
        tour[i] = [math.inf, [0], None]  # (cost, [selected tour], DefaultDict with all selected node in tour)
    already_taken_first_choice = DefaultDict(int)
    already_taken_first_choice[0] = 1
    for i in range(widht):
        actual_best = None
        minimum = math.inf
        for j in range(1, len(distance_matrix)):
            if distance_matrix[0][j] >= 1 and real_distance_matrix[0][j]+1 < minimum and already_taken_first_choice[j] == 0:
                minimum = real_distance_matrix[0][j]+1
                actual_best = j
            elif distance_matrix[0][j] < 1 and distance_matrix[0][j] < minimum and already_taken_first_choice[j] == 0:
                minimum = distance_matrix[0][j]
                actual_best = j
        tour[-1] = [distance_matrix[0][actual_best], [0, actual_best], DefaultDict(int)]
        tour[-1][2][0] = 1
        tour[-1][2][actual_best] = 1
        tour.sort(key=lambda x: x[0])
        already_taken_first_choice[actual_best] = 1

    for i in range(2, len(distance_matrix)):
        list_candidate = [-1] * (widht**2)
        # for all kept node, we take the widht best next node
        for k in range(widht):
            actual = tour[k][1][-1]
            small_list_candidate = [[tour[k][0], math.inf, [0], -1]] * widht  # [last_cost, cost for this edge, [selected tour], DefaultDict with all selected node in tour]
            for j in range(len(distance_matrix)):
                if tour[k][2][j] == 0 and distance_matrix[actual][j] == 1:
                    if real_distance_matrix[actual][j]+1 < small_list_candidate[-1][1] and tour[k][2][j] == 0:
                        small_list_candidate[-1] = [tour[k][0], real_distance_matrix[actual][j]+1, tour[k][1].copy(), deepcopy(tour[k][2])]
                        small_list_candidate[-1][2].append(j)
                        small_list_candidate[-1][3][j] = 1
                        small_list_candidate.sort(key=lambda x: x[1])
                elif distance_matrix[actual][j] < 1 and distance_matrix[actual][j] < small_list_candidate[-1][1] and tour[k][2][j] == 0:
                    small_list_candidate[-1] = [tour[k][0], distance_matrix[actual][j], tour[k][1].copy(), deepcopy(tour[k][2])]
                    small_list_candidate[-1][2].append(j)
                    small_list_candidate[-1][3][j] = 1
                    small_list_candidate.sort(key=lambda x: x[1])
            for l in range(widht):
                list_candidate[k*widht+l] = small_list_candidate[l]
        list_candidate.sort(key=lambda x: x[0]+x[1])
        for k in range(widht):
            tour[k] = [list_candidate[k][0]+list_candidate[k][1], list_candidate[k][2], list_candidate[k][3]]
    return tour[0][1]

# Adding a 2-opt local search after the first found search:
def local_search(distance_matrix, initial_path, initial_cost=0, full=False):
    path = initial_path
    cost = initial_cost
    improved = True
    iterr = 0
    maxiterr = 1000 if full else 100
    while improved and iterr < maxiterr:
        iterr += 1
        improved = False
        
        for i in range(-1, len(path)-2):
            for j in range(i+1, len(path)-1):
                gain = distance_matrix[path[i]][path[i+1]] + distance_matrix[path[j]][path[j+1]] - distance_matrix[path[i]][path[j]] - distance_matrix[path[i+1]][path[j+1]]
                if gain > 0.000001:
                    path = swap_tour(path, i, j)
                    cost -= gain
                    improved = True
    return path, cost

def swap_tour(path, i, j):
    new_path = copy(path)
    for k in range(i, j):
        new_path[k+1] = path[i+j-k]
    return new_path

def get_cost(distance_matrix, tour):
    """Function used to claculate the lenght of a tour based on a distance matrix"""
    cost = 0
    for i in range(len(tour)-1):
        cost += distance_matrix[tour[i]][tour[i+1]]
    cost += distance_matrix[tour[-1]][tour[1]]
    return cost

# Gready search with depth :
def probalistic_greedy_search_with_depth(distance_matrix, real_distance_matrix, depth):
    """A greedy search method that can see more that the next node"""
    
    tour = [-1] * len(distance_matrix)
    visited_node = DefaultDict(int)
    
    actual = 0
    tour[0] = actual
    visited_node[0] = 1
    
    for i in range(1, len(tour)):
        minimum_depht = math.inf
        for a in range(len(distance_matrix)):
            if visited_node[a] == 0:
                sum_minimum = distance_matrix[actual][a] if distance_matrix[actual][a] != 1 else real_distance_matrix[actual][a]+1
                small_actual = a
                small_visited_node = DefaultDict(int)
                small_visited_node[a] = 1
                for k in range(depth if a < len(distance_matrix)-depth else len(distance_matrix)-a+1):
                    small_minimum = 1
                    for j in range(len(distance_matrix)):
                        if (distance_matrix[small_actual][j]) < small_minimum and visited_node[j] == 0 and small_visited_node[j] == 0:
                            small_next = j
                            small_minimum = distance_matrix[small_actual][j]
                    if small_minimum == 1:
                        small_minimum = math.inf
                        for j in range(len(real_distance_matrix)):
                            if (real_distance_matrix[small_actual][j]+1) < small_minimum and visited_node[j] == 0:
                                small_next = j
                                small_minimum = real_distance_matrix[small_actual][j]+1
                    sum_minimum += small_minimum
                    small_visited_node[small_next] = 1
                    small_actual = small_next
                if sum_minimum < minimum_depht:
                    big_next = a
                    minimum_depht = sum_minimum
        tour[i] = big_next
        visited_node[big_next] = 1
        actual = big_next
            
    return tour


if __name__ == "__main__":
    node_pos = np.array([[0.5, 0],
                         [0, 0],
                         [0, 0.5],
                         [0.5, 0.5],
                         [0.75, 0.25]])
    coef_matrix = np.array([[0, 1, 0, 2, 1],
                            [1, 0, 1, 2, 0],
                            [0, 1, 0, 1, 0],
                            [2, 2, 1, 0, 1],
                            [1, 0, 0, 1, 0]])
    true_coef_mat = 1 - coef_matrix/2
    print(probalistic_greedy_search_with_depth(true_coef_mat, euclidean_distance_matrix(node_pos), 4))