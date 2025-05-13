import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import psutil

def get_available_memory_gb():
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)

def load_limited_coords(files, max_points=10000):
    all_coords = []
    total_loaded = 0
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["channel0", "channel1"])
        except Exception as e:
            print(f"[SKIP] {f}: {e}")
            continue

        coords = df[["channel0", "channel1"]].dropna().to_numpy()
        if coords.size == 0:
            continue

        if total_loaded + len(coords) > max_points:
            needed = max_points - total_loaded
            all_coords.append(coords[:needed])
            break
        else:
            all_coords.append(coords)
            total_loaded += len(coords)

    if not all_coords:
        raise RuntimeError("No valid data loaded.")
    return np.vstack(all_coords)

def calculate_k_distances(coords, k=3):
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    return np.sort(distances[:, -1])

def estimate_densities_sampled(coords, dc, sample_size=5000):
    sample_coords = coords[np.random.choice(len(coords), min(sample_size, len(coords)), replace=False)]
    tree = NearestNeighbors(radius=dc).fit(coords)
    neighbors = tree.radius_neighbors(sample_coords, return_distance=False)
    return np.array([len(n) for n in neighbors])

def main():
    directory = "/Users/meenandm/Documents/DBScan-for-Clustering/ttbar_mu100_strips"
    files = glob.glob(os.path.join(directory, "*cells.csv"))

    print(f"[INFO] Memory available: {get_available_memory_gb():.2f} GB")
    print("[INFO] Loading up to 10,000 points from files...")

    try:
        coords = load_limited_coords(files, max_points=10000)
    except RuntimeError as e:
        print(str(e))
        return

    print(f"[INFO] Loaded {len(coords)} coordinates for estimation.")

    # --- Step 1: Estimate dc
    print("[INFO] Generating k-distance graph to estimate dc...")
    k_distances = calculate_k_distances(coords, k=4)

    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.xlabel("Points sorted by 3-NN distance")
    plt.ylabel("3-NN distance")
    plt.title("K-distance Graph")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    try:
        dc = float(input("Enter estimated dc from elbow (e.g., 10): "))
    except ValueError:
        print("[ERROR] Invalid dc input.")
        return

    # --- Step 2: Estimate rhoc
    print("[INFO] Estimating densities from sampled points...")
    densities = estimate_densities_sampled(coords, dc)

    plt.figure(figsize=(8, 5))
    plt.hist(densities, bins=40, color='skyblue', edgecolor='black')
    plt.xlabel("Density (neighbors within dc)")
    plt.ylabel("Frequency")
    plt.title(f"Density Histogram (dc = {dc})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for p in [70, 80, 90, 95]:
        print(f"[INFO] Suggested rhoc at {p}th percentile: {np.percentile(densities, p):.2f}")

if __name__ == "__main__":
    main()
