"""File with the Mahalanobis distance solver for TSP, it give a matrix of probability"""

import numpy as np
import matplotlib.pyplot as plt
from random import randint, random

from scipy.spatial.distance import mahalanobis
from python_tsp.distances import euclidean_distance_matrix

from src.utils.solve_methods import solve_small_tsp
from src.utils.search_methods import local_search, probalistic_greedy_search, probalistic_greedy_search_2

class TSP_Problem_Mahalanobis(object):
    """Class used to get all information about the tsp problem like the distance matrix, the nearest node, etc."""
    
    def __init__(self, position_node, distance_matrix=None):
        self.position_node = position_node
        self.distance_matrix = distance_matrix if distance_matrix is not None else euclidean_distance_matrix(position_node)
        self.coef_matrix = np.zeros((len(self.distance_matrix), len(self.distance_matrix)))
        # self.coef_matrix2 = np.zeros((len(self.distance_matrix), len(self.distance_matrix)))
        # self.inverted_coef_mat = None
        # self.coef_max = 0

    def get_nearest(self, index, n_neighbor):
        """Method used to get a distance matrix with all the nearest neighbor for node at given index"""
        # We create the covariance matrix :
        cov_mat = np.array([[random(), 0],
                            [0, random()]])
        # Selection of the n_best node
        n_best_node = np.ones((n_neighbor, 2)).tolist()  # [[index, Mahalanobis_distance], ...]
        for i in range(len(self.position_node)):
            mahal_dist = mahalanobis(self.position_node[index], self.position_node[i], cov_mat)
            if i != index and mahal_dist < n_best_node[-1][1]:
                n_best_node[-1] = [i, mahal_dist]
                n_best_node.sort(key=lambda x: x[1])
        n_best_node.append([index, 0])

        # Creation of the matrix with these selected node
        new_distance_matrix = np.zeros((n_neighbor+1, n_neighbor+1))
        for i in range(n_neighbor):
            index_first = n_best_node[i][0]
            for j in range(i, n_neighbor):
                index_second = n_best_node[j][0]
                new_distance_matrix[i][j] = self.distance_matrix[index_first][index_second]
                new_distance_matrix[j][i] = self.distance_matrix[index_first][index_second]
        corresponding_node = [x[0] for x in n_best_node]
        new_position_node = [self.position_node[x] for x in corresponding_node]
        return np.array(new_distance_matrix), np.array(new_position_node), corresponding_node

    def get_real_path(self, path, corresponding_node):
        real_path = [0] * len(path)
        for i in range(len(path)):
            real_path[i] = corresponding_node[path[i]]
        return real_path

    def update_coef_matrix(self, path):
        for i in range(len(path)-1):
            self.coef_matrix[path[i]][path[i+1]] += 1
            self.coef_matrix[path[i+1]][path[i]] += 1
        self.coef_matrix[path[0]][path[-1]] += 1
        self.coef_matrix[path[-1]][path[0]] += 1
        # if self.coef_matrix[path[i]][path[i+1]] > self.coef_max:
        #     self.coef_max = self.coef_matrix[path[i]][path[i+1]]
        # if self.coef_matrix[path[0]][path[-1]] > self.coef_max:
        #     self.coef_max = self.coef_matrix[path[0]][path[-1]]

        """
        for i in range(len(path)-1):
            self.coef_matrix2[path[i]][path[i+1]] = 1
            self.coef_matrix2[path[i+1]][path[i]] = 1
        self.coef_matrix2[path[0]][path[-1]] = 1
        self.coef_matrix2[path[-1]][path[0]] = 1
        """

    def solve(self, number_of_repetition=1, n_neighbor=10, method="TSP_Solver"):
        """Methos used to get the probalistic matrix"""

        self.coef_matrix = np.zeros((len(self.distance_matrix), len(self.distance_matrix)))

        for _ in range(number_of_repetition):
            index = randint(0,len(self.position_node)-1)
            reduce_distance_matrix, reduce_position_node, corresponding_node = self.get_nearest(index, n_neighbor)
            tour_nodes = solve_small_tsp(method, reduce_distance_matrix, reduce_position_node)
            tour_nodes = self.get_real_path(tour_nodes, corresponding_node)
            self.update_coef_matrix(tour_nodes)
        # self.inverted_coef_mat = 1 - self.coef_matrix / self.coef_max
        # tour = probalistic_greedy_search(self.inverted_coef_mat, self.distance_matrix)
        tour = probalistic_greedy_search_2(self.coef_matrix, self.distance_matrix)
        tour, _ = local_search(self.distance_matrix, tour, 0)
        return tour
