"""File used to convert an TSP instance to an TSP instance readeable by the concorde"""

import sys
from torch.utils.data import DataLoader

from dataloader import *

def to_concorde_instance(position_node, file_name):
    """Function used to convert an TSP instance to an TSP instance readeable by the concorde"""
    with open(f"{file_name}.tsp", 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.

        print(f"NAME: {file_name}")
        print(f"TYPE: TSP")
        print(f"DIMENSION: {len(position_node)}")
        print(f"EDGE_WEIGHT_TYPE: EUC_2D")
        print(f"NODE_COORD_SECTION")
        for i in range(len(position_node)):
            print(f"{i+1} {position_node[i][0]*1000} {position_node[i][1]*1000}")


if __name__ == "__main__":
    print("start")
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
    with torch.no_grad():
        for bat_idx, bat in enumerate(dataloader):
            if bat['nodes'].shape[1] == 200:
                to_concorde_instance(bat['nodes'][0], "premier_test")
                break
    print("end")