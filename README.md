# DBScan-for-Clustering

# Basic Density-Based Clustering

This project provides a simple implementation of a density-based clustering algorithm in Python. The algorithm is designed to cluster points based on their local density and uses parameters like the distance cutoff (`dc`) and density threshold (`rhoc`) to identify clusters.

## Features

1. **Density-Based Clustering**:
   - Clusters points based on their local density.
   - Uses a pairwise distance matrix to calculate densities.
   - Identifies high-density points as "seeds" and expands clusters from these seeds.

2. **Recursive Cluster Expansion**:
   - Expands clusters by recursively adding neighbors of seed points that meet the density criteria.

3. **Flexible Input**:
   - Accepts input data in `pandas.DataFrame` format with required columns for coordinates (`x0`, `x1`, etc.) and weights (`weight`).

4. **Customizable Parameters**:
   - `dc`: Distance cutoff for neighborhood calculation.
   - `rhoc`: Minimum density threshold to identify seeds.

5. **CSV Integration**:
   - Input data can be loaded from a CSV file.
   - Results can be exported to a CSV file for further analysis.

## Requirements

- Python 3.7+
- Libraries:
  - `numpy`
  - `pandas`
  - `scipy`

Install the required libraries using:
```bash
pip install numpy pandas scipy