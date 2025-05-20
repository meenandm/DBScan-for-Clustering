import os
import glob
import pandas as pd

def print_cluster_summary(file):
    df = pd.read_csv(file)

    # Count occurrences of each hit_id
    hit_counts = df['hit_id'].value_counts()

    # Assign cluster_id:
    # noise (-1) if hit_id appears once, else a cluster_id number
    cluster_ids = {}
    current_cluster_id = 0
    for hit_id, count in hit_counts.items():
        if count == 1:
            continue  # noise, no cluster ID
        cluster_ids[hit_id] = current_cluster_id
        current_cluster_id += 1

    df['cluster_id'] = df['hit_id'].apply(lambda x: cluster_ids.get(x, -1))

    # Calculate summary
    total_points = len(df)
    noise_points = (df['cluster_id'] == -1).sum()
    num_clusters = len(set(cluster_ids.values()))
    
    print(f"File: {os.path.basename(file)}")
    print(f"  Total points: {total_points}")
    print(f"  Clusters: {num_clusters}")
    print(f"  Noise points: {noise_points}")
    print(f"  Noise percentage: {100 * noise_points / total_points:.2f}%")
    print()

if __name__ == "__main__":
    data_dir = "/Users/meenandm/Documents/DBScan-for-Clustering/ttbar_mu100_strips"
    global_files = sorted(glob.glob(os.path.join(data_dir, "*cells-global.csv")))

    if not global_files:
        print("No *cells-global.csv files found.")
    else:
        for file in global_files:
            print_cluster_summary(file)
