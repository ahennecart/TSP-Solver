"""File with the square and line solver for TSP, it give a matrix of probability"""

# TODO : faire comme pour KNN et Maha : ne pas appeler le solver "Concorde" avec toutes les positions node, mais n'avoir qu'une petite liste

import numpy as np
from python_tsp.distances import euclidean_distance_matrix

from src.utils.solve_methods import solve_small_tsp
from src.utils.search_methods import local_search, probalistic_greedy_search, probalistic_greedy_search_2


class TSP_Problem_Zone(object):
    """Class used to get all information about the tsp problem like the distance matrix, the nearest node, etc."""
    
    def __init__(self, position_node, number_of_division, zone_lenght, distance_matrix=None):
        self.zone_lenght = zone_lenght
        self.position_node = position_node
        self.number_of_division = number_of_division
        self.distance_matrix = distance_matrix if distance_matrix is not None else euclidean_distance_matrix(position_node)
        self.set_zone_matrix(number_of_division, position_node)
        self.coef_matrix = np.zeros((len(position_node), len(position_node)))
        # self.coef_matrix2 = np.zeros((len(position_node), len(position_node)))
        # self.inverted_coef_mat = None
        # self.coef_max = 0
        
    def set_zone_matrix(self, number_of_division, position_node):
        self.zone_matrix = [0] * number_of_division
        for i in range(number_of_division):
            self.zone_matrix[i] = [0] * number_of_division
            
        for a in range(len(position_node)):
            i = int(position_node[a][0] * number_of_division)
            j = int(position_node[a][1] * number_of_division)
            if type(self.zone_matrix[i][j]) == int:
                self.zone_matrix[i][j] = [(a, position_node[a])]
            else:
                self.zone_matrix[i][j].append((a, position_node[a]))
    
    def get_nodes_square(self, index_i, index_j):
        """Method used to get a distance matrix with all the selected node in a square in the different zones"""
        new_position_node = []
        new_corresponding_node_list = []
        for i in range(self.zone_lenght):
            for j in range(self.zone_lenght):
                zone = self.zone_matrix[index_i+i][index_j+j]
                if zone != 0:
                    for k in range(len(zone)):
                        new_corresponding_node_list.append(zone[k][0])
                        new_position_node.append(zone[k][1])
        if len(new_position_node) <= 2:
            return None, None, None
        return np.array(euclidean_distance_matrix(new_position_node)), np.array(new_position_node), np.array(new_corresponding_node_list)
    
    def get_nodes_line(self, index, width):
        """
        Method used to get a distance matrix with all the selected node in a line in the different zones
        index : 0 to self.number_of_division-width = vertical ligne
                self.number_of_division-width+1 to 2*self.number_of_division-2*width = horizontal line
        """
        new_position_node = []
        new_corresponding_node_list = []
        for i in range(self.number_of_division):
            for j in range(width):
                if index <= self.number_of_division-width:
                    zone = self.zone_matrix[index+j-width+1][i]
                else:
                    zone = self.zone_matrix[i][index+j-self.number_of_division-width+1]
                if zone != 0:
                    for k in range(len(zone)):
                        new_corresponding_node_list.append(zone[k][0])
                        new_position_node.append(zone[k][1])
        if len(new_position_node) <= 2:
            return None, None, None
        new_corresponding_node_list
        return np.array(euclidean_distance_matrix(new_position_node)), np.array(new_position_node), np.array(new_corresponding_node_list)
    
    def get_real_path(self, path, corresponding_node):
        """Method used to retrieve the real path taken by the TSP on the small TSP"""
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

    def solve_square(self, zone_lenght, line_width=1, method="TSP_Solver"):
        """Method used to solve the TSP with the square method"""
        for index_i in range(self.number_of_division - zone_lenght + 1):
            for index_j in range(self.number_of_division - zone_lenght + 1):
                reduce_distance_matrix, reduce_position_node, corresponding_node = self.get_nodes_square(index_i, index_j)
                if reduce_distance_matrix is not None:
                    tour_nodes = solve_small_tsp(method, reduce_distance_matrix, reduce_position_node)
                    tour_nodes = self.get_real_path(tour_nodes, corresponding_node)
                    self.update_coef_matrix(tour_nodes)
        for index in [0, self.number_of_division-1, self.number_of_division, 2*self.number_of_division-1]:
            reduce_distance_matrix, reduce_position_node, corresponding_node = self.get_nodes_line(index, line_width)
            if reduce_distance_matrix is not None:
                tour_nodes = solve_small_tsp(method, reduce_distance_matrix, reduce_position_node)
                tour_nodes = self.get_real_path(tour_nodes, corresponding_node)
                self.update_coef_matrix(tour_nodes)
        # self.inverted_coef_mat = 1 - self.coef_matrix / self.coef_max
        # tour = probalistic_greedy_search(self.inverted_coef_mat, self.distance_matrix)
        tour = probalistic_greedy_search_2(self.coef_matrix, self.distance_matrix)
        tour, _ = local_search(self.distance_matrix, tour, 0)
        return tour
        
    def solve_line(self, line_width, method="TSP_Solver"):
        """Method used to solve the TSP with the line method"""

        for index in range(2*self.number_of_division):
            reduce_distance_matrix, reduce_position_node, corresponding_node = self.get_nodes_line(index, line_width)
            
            if reduce_distance_matrix is not None:
                tour_nodes = solve_small_tsp(method, reduce_distance_matrix, reduce_position_node)
                tour_nodes = self.get_real_path(tour_nodes, corresponding_node)
                self.update_coef_matrix(tour_nodes)
        # self.inverted_coef_mat = 1 - self.coef_matrix / self.coef_max
        # tour = probalistic_greedy_search(self.inverted_coef_mat, self.distance_matrix)
        tour = probalistic_greedy_search_2(self.coef_matrix, self.distance_matrix)
        tour, _ = local_search(self.distance_matrix, tour, 0)
        return tour