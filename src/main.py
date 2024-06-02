"""Main file with exemple of how to use the solvers"""
import torch
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import time

from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import euclidean_distance_matrix

from models.KNN import TSP_Problem_KNearest
from models.Square_division import TSP_Problem_Zone
from models.Mahalanobis import TSP_Problem_Mahalanobis
from utils.search_methods import *
from utils.dataloader import *
from utils.plots import *

if __name__ == "__main__":
    dataset_path = "data/tsp10-200_concorde.txt"
    batch_size = 16
    num_samples = 25600 # 1280 samples per TSP size
    neighbors = 0.20
    knn_strat = 'percentage'
    # We load the dataset
    dataset = TSP.make_dataset(
        filename=dataset_path, batch_size=batch_size, num_samples=num_samples, 
        neighbors=neighbors, knn_strat=knn_strat, supervised=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # We solve the matrix of probability
    i = 0
    with torch.no_grad():
        for bat_idx, bat in enumerate(dataloader):
            if bat['nodes'].shape[1] == 200:
                start_time = time.time()
                x = bat['nodes']
                for plot_idx in range(1, len(bat['nodes'])):
                    # problem_knn = TSP_Problem_Mahalanobis(x[plot_idx], n_neighbor=5)
                    # problem_knn.solve(number_of_repetition=1000)
                    problem_knn = TSP_Problem_Mahalanobis(x[plot_idx])
                    problem_knn.solve(number_of_repetition=500, n_neighbor=5, method="local_search")
                    i+=1
                    break
            if i >= 1:
                break
    
    print(f"temps total : {time.time()-start_time:.3f}")
    f = plt.figure(0, figsize=(20, 5))
    p1 = f.add_subplot(131)
    sns.heatmap(
                    problem_knn.coef_matrix,
                    cmap="YlGnBu",
                    ax=p1,
                    square=True
                ).set_title("Matrice de proba")
    p2 = f.add_subplot(132)
    sns.heatmap(
                    problem_knn.coef_matrix2,
                    cmap="YlGnBu",
                    ax=p2,
                    square=True
                ).set_title("Matrice de proba 2")
    p3 = f.add_subplot(133)
    sns.heatmap(
                    problem_knn.inverted_coef_mat,
                    cmap="YlGnBu",
                    ax=p3,
                    square=True
                ).set_title("Matrice de proba inversee")
    plt.show()

    # We perform a gready search with the matrix
    tour = probalistic_greedy_search(problem_knn.inverted_coef_mat, problem_knn.distance_matrix)
    print(tour)
    gt_cost = get_cost(problem_knn.distance_matrix, tour)
    f = plt.figure(0, figsize=(20, 5))
    p1 = f.add_subplot(141)
    plot_tsp(
        p1, 
        problem_knn.position_node.cpu().numpy(), 
        np.array(tour), 
        title=f"TSP-Solver: {gt_cost:.3f}"
    )
    p2 = f.add_subplot(142)
    plot_tsp_heatmap(
        p2, 
        x[plot_idx].cpu().numpy(), 
        problem_knn.coef_matrix, 
        threshold=0.1,
        title="Prediction Heatmap"
    )
    plt.show()
    # Followed by a little local search to upgrade the solution
    tour_local_search, _ = local_search(problem_knn.distance_matrix, tour, gt_cost)
    print(tour_local_search)
    f = plt.figure(0, figsize=(20, 5))
    p1 = f.add_subplot(141)
    plot_tsp(
        p1, 
        problem_knn.position_node.cpu().numpy(), 
        np.array(tour_local_search), 
        title=f"TSP-Solver: {get_cost(problem_knn.distance_matrix, tour_local_search):.3f}"
    )
    plt.show()