import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import glob

def density_based_clustering(data, dc, rhoc):
    """
    Perform density-based clustering on the input data.

    Parameters:
        data (pd.DataFrame): Input data with coordinates and weights.
            It must have columns ['channel0', 'channel1', 'value'].
        dc (float): Distance cutoff for density calculation.
        rhoc (float): Minimum density threshold for seeds.

    Returns:
        pd.DataFrame: Data with cluster IDs and seed information.
    """
    n_points = len(data)
    densities = np.zeros(n_points)
    cluster_ids = -1 * np.ones(n_points, dtype=int)
    is_seed = np.zeros(n_points, dtype=int)

    coords = data[['channel0', 'channel1']].to_numpy()
    distances = cdist(coords, coords)

    for i in range(n_points):
        densities[i] = np.sum(distances[i] <= dc)

    is_seed[densities >= rhoc] = 1

    cluster_id = 0
    for i in range(n_points):
        if is_seed[i] and cluster_ids[i] == -1:
            cluster_ids[i] = cluster_id
            expand_cluster(i, cluster_id, distances, densities, cluster_ids, dc, rhoc)
            cluster_id += 1

    data['cluster_id'] = cluster_ids
    data['is_seed'] = is_seed
    return data

def expand_cluster(index, cluster_id, distances, densities, cluster_ids, dc, rhoc):
    neighbors = np.where(distances[index] <= dc)[0]
    for neighbor in neighbors:
        if cluster_ids[neighbor] == -1:
            cluster_ids[neighbor] = cluster_id
            if densities[neighbor] >= rhoc:
                expand_cluster(neighbor, cluster_id, distances, densities, cluster_ids, dc, rhoc)

if __name__ == "__main__":
    local_dir = "/Users/meenandm/Documents/DBScan-for-Clustering/ttbar_mu100/"
    dc = 2.0
    rhoc = 4

    # List all CSV files in the local directory
    csv_files = glob.glob(os.path.join(local_dir, "*cells.csv"))


    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        data = pd.read_csv(csv_file)

        required_columns = {"geometry_id", "hit_id", "channel0", "channel1", "value"}
        if not required_columns.issubset(data.columns):
            print(f"Skipping {csv_file}. Missing required columns: {required_columns}")
            continue

        clustered_data = density_based_clustering(data, dc, rhoc)
        output_file = csv_file.replace(".csv", "_clustered.csv")
        clustered_data.to_csv(output_file, index=False)
        print(f"Clustering results saved to {output_file}")
