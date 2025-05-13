import pandas as pd
import matplotlib.pyplot as plt
import os

# Set path to clustered CSV
clustered_file = "/Users/meenandm/Documents/DBScan-for-Clustering/ttbar_mu100_strips/event000000005-cells_clustered.csv"

# Load the clustered data
data = pd.read_csv(clustered_file)

# Check necessary columns
if not {'channel0', 'channel1', 'cluster_id'}.issubset(data.columns):
    raise ValueError("Missing required columns: channel0, channel1, or cluster_id")

# Total number of noise points
noise_points = (data['cluster_id'] == -1).sum()

# Total number of unique clusters (excluding noise)
unique_clusters = data['cluster_id'].nunique() - (1 if -1 in data['cluster_id'].values else 0)

print(f"Total number of unique clusters (excluding noise): {unique_clusters}")
print(f"Total number of noise points: {noise_points}")
print(f"Total number of data points: {len(data)}")

# Plot clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    data['channel0'],
    data['channel1'],
    c=data['cluster_id'],
    cmap='tab20',  # Good for categorical coloring
    s=10,
    alpha=0.7
)

plt.colorbar(scatter, label="Cluster ID")
plt.xlabel("channel0")
plt.ylabel("channel1")
plt.title("Clustered Cells for event000000005")
plt.grid(True)
plt.tight_layout()
plt.show()
