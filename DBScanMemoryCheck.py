import os
import pandas as pd
import numpy as np
import glob
from numba import njit, prange
from collections import deque
import time
import psutil

@njit(parallel=True)
def compute_pairwise_distances(coords):
    n = coords.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in prange(n):
        for j in range(n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]
            dist_matrix[i, j] = (dx**2 + dy**2 + dz**2) ** 0.5
    return dist_matrix

@njit()
def calculate_densities(distances, dc):
    return np.sum(distances <= dc, axis=1)

def expand_cluster(index, cluster_id, distances, densities, cluster_ids, dc, rhoc):
    queue = deque([index])
    while queue:
        current = queue.popleft()
        neighbors = np.where(distances[current] <= dc)[0]
        for neighbor in neighbors:
            if cluster_ids[neighbor] == -1:
                cluster_ids[neighbor] = cluster_id
                if densities[neighbor] >= rhoc:
                    queue.append(neighbor)

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
    coords = data[['x_global', 'y_global', 'z_global']].to_numpy(dtype=np.float32)
    cluster_ids = -1 * np.ones(n_points, dtype=np.int32)

    print("Computing distances...")
    distances = compute_pairwise_distances(coords)

    print("Calculating densities...")
    densities = calculate_densities(distances, dc)
    is_seed = (densities >= rhoc).astype(np.int32)

    print("Expanding clusters...")
    cluster_id = 0
    for i in range(n_points):
        if is_seed[i] and cluster_ids[i] == -1:
            cluster_ids[i] = cluster_id
            expand_cluster(i, cluster_id, distances, densities, cluster_ids, dc, rhoc)
            cluster_id += 1

    data['cluster_id'] = cluster_ids
    data['is_seed'] = is_seed
    return data

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {mem_mb:.2f} MB")

if __name__ == "__main__":
    local_dir = "/Users/meenandm/Documents/DBScan-for-Clustering/ttbar_mu100_strips/"
    dc = 45  # Adjust for 3D distances; e.g., 50 mm 
    rhoc = 13  # Adjust for density; e.g., 22 hits/mm^3 from Histogram Analysis 70th percentile

    csv_files = glob.glob(os.path.join(local_dir, "*-cells-global.csv"))

    for csv_file in csv_files:
        print_memory_usage()
        print(f"Processing file: {csv_file}")
        data = pd.read_csv(csv_file)

        required_columns = {"geometry_id", "hit_id", "x_global", "y_global", "z_global"}
        if not required_columns.issubset(data.columns):
            print(f"Skipping {csv_file}. Missing required columns: {required_columns}")
            continue

        start_time = time.time()
        clustered_data = density_based_clustering(data, dc, rhoc)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        output_file = csv_file.replace(".csv", "_clustered.csv")
        clustered_data.to_csv(output_file, index=False)
        print(f"Clustering results saved to {output_file}")
