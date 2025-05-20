# DBScan-for-Clustering

# Basic Density-Based Clustering

This project provides a simple implementation of a density-based clustering algorithm in Python. The algorithm is designed to cluster points based on their local density and uses parameters like the distance cutoff (`dc`) and density threshold (`rhoc`) to identify clusters.

## Features

1. **Local To Global Transformation**:
   - Using Trackml-detector.csv and cell files from ttbar_mu100 for transformation
   - Parameterising Strip Cells
   - Each Unique hit_id represents a data point within same cluster or indivual points

2. **Density-Based Clustering**:
   - Clusters points based on their local density.
   - Uses a pairwise distance matrix to calculate densities.
   - Identifies high-density points as "cluster" and noise points which lie for density threshold.

3. **Recursive Cluster Expansion**:
   - Expands clusters by recursively adding neighbors of data points that meet the density criteria.
   - Enter estimated dc from elbow (range 45-60): 47.5
   - Estimating densities from sampled points...__
     Suggested `rhoc` at 70th percentile: 22.00__
     Suggested `rhoc` at 80th percentile: 25.00__
     Suggested `rhoc` at 90th percentile: 29.00__
     Suggested `rhoc` at 95th percentile: 32.00__

3. **Flexible Input**:
   - Accepts input data in `pandas.DataFrame` format with required columns for coordinates (`x0`, `x1`, `x2` for hit and `cx`, `cy`, `cz` for detector modules.) and weights (`values`).

4. **Customizable Parameters**:
   - `dc`: Distance cutoff for neighborhood calculation.
   - `rhoc`: Minimum density threshold to hits/mm^2.

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
