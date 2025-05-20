import os
import glob
import pandas as pd

def summarize_clusters(clustered_files):
    total_points = 0
    total_noise_points = 0
    all_cluster_ids = []
    total_clusters = 0

    print("File-wise cluster summary:")
    for file in clustered_files:
        df = pd.read_csv(file)
        total_points += len(df)
        noise_points = (df['cluster_id'] == -1).sum()
        total_noise_points += noise_points
        
        # Exclude noise points from unique cluster IDs
        unique_ids = df['cluster_id'].unique()
        unique_ids = [cid for cid in df['cluster_id'].unique() if cid != -1] # exclude noise
        all_cluster_ids.extend(unique_ids)

        print(f"- {os.path.basename(file)}: "
              f"{len(unique_ids)} clusters, {noise_points} noise points, {len(df)} total points")
        
        total_clusters = total_clusters + len(set(all_cluster_ids))

    print("\n=== Overall Summary ===")
    print(f"Total data points: {total_points}")
    print(f"Total clusters (excluding noise): {total_clusters}")
    print(f"Total noise points: {total_noise_points}")
    print(f"Noise percentage: {100 * total_noise_points / total_points:.2f}%")

if __name__ == "__main__":
    clustered_dir = "/Users/meenandm/Documents/DBScan-for-Clustering/ttbar_mu100_strips"
    clustered_files = sorted(glob.glob(os.path.join(clustered_dir, "*cells-global_clustered.csv")))

    if not clustered_files:
        print("No clustered CSV files found.")
    else:
        print(f"[INFO] Found {len(clustered_files)} clustered files.")
        summarize_clusters(clustered_files)
