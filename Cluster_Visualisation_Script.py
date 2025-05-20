import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

def run_clustering_summary_and_visualization(data_directory, output_plots_directory="cluster_visualizations"):
    """
    Combines the cluster summary and visualization logic into one function.

    Args:
        data_directory (str): Path to the directory containing _cells_clustered.csv files.
        output_plots_directory (str): Directory to save the generated plots.
    """
    os.makedirs(output_plots_directory, exist_ok=True) # Ensure output directory exists

    all_clustered_files = sorted(glob.glob(os.path.join(data_directory, "*cells-global_clustered.csv")))
    
    if not all_clustered_files:
        print(f"[ERROR] No 'cells-global_clustered.csv' files found in '{data_directory}'. Please check the path.")
        return

    # --- Part 1: Textual Cluster Summary ---
    print("--- Running Cluster Summary ---")
    print(f"[INFO] Found {len(all_clustered_files)} clustered files for summary.")

    total_points_overall = 0
    total_noise_overall = 0
    total_clusters_overall = 0 # This will accumulate unique clusters across all files

    print("File-wise cluster summary:")
    for f_path in all_clustered_files:
        try:
            df_summary = pd.read_csv(f_path, usecols=["cluster_id"])
        except KeyError:
            print(f"[WARNING] Skipping {os.path.basename(f_path)}: 'cluster_id' column not found.")
            continue
        except Exception as e:
            print(f"[ERROR] Could not read {os.path.basename(f_path)}: {e}")
            continue

        current_total_points = len(df_summary)
        current_noise_points = (df_summary["cluster_id"] == -1).sum()

        # Count unique clusters, excluding -1 (noise)
        current_unique_clusters = df_summary["cluster_id"].nunique(dropna=True)
        if -1 in df_summary["cluster_id"].values:
            current_unique_clusters -= 1

        total_points_overall += current_total_points
        total_noise_overall += current_noise_points
        total_clusters_overall += current_unique_clusters

        print(f"- {os.path.basename(f_path)}: {current_unique_clusters} clusters, {current_noise_points} noise points, {current_total_points} total points")

    print("\n=== Overall Summary ===")
    print(f"Total data points: {total_points_overall}")
    print(f"Total clusters (excluding noise): {total_clusters_overall}")
    print(f"Total noise points: {total_noise_overall}")
    if total_points_overall > 0:
        print(f"Noise percentage: {100 * total_noise_overall / total_points_overall:.2f}%")
    else:
        print("Noise percentage: N/A (No data points to summarize)")

    # --- Part 2: Cluster Visualization ---
    print("\n--- Generating Cluster Visualizations ---")
    print(f"[INFO] Found {len(all_clustered_files)} clustered files for visualization.")

    for i, f_path in enumerate(all_clustered_files):
        print(f"Processing {os.path.basename(f_path)} for visualization...")
        
        try:
            df_plot = pd.read_csv(f_path)
        except Exception as e:
            print(f"[ERROR] Could not read {os.path.basename(f_path)} for plotting: {e}")
            continue

        # Ensure necessary columns exist for plotting
        required_cols_plot = ['x_global', 'y_global', 'z_global', 'cluster_id']
        if not all(col in df_plot.columns for col in required_cols_plot):
            print(f"[WARNING] Skipping visualization for {os.path.basename(f_path)}: Missing one or more required columns ({required_cols_plot}).")
            continue

        # Separate noise points from clustered points
        noise_points = df_plot[df_plot['cluster_id'] == -1]
        clustered_points = df_plot[df_plot['cluster_id'] != -1]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot clustered points
        if not clustered_points.empty:
            # Use a color mapping based on cluster_id for visual distinction on the plot
            # but we will NOT add a separate label for each cluster_id in the legend.
            
            # Create a color map for clustered points
            unique_cluster_ids = clustered_points['cluster_id'].unique()
            if len(unique_cluster_ids) > 20: # Use HSV if too many clusters for distinct tab20
                colors = sns.color_palette("hsv", n_colors=len(unique_cluster_ids))
            else:
                colors = sns.color_palette("tab20", n_colors=len(unique_cluster_ids))
            
            # Map cluster_ids to colors
            cluster_id_to_color = {cid: colors[i] for i, cid in enumerate(unique_cluster_ids)}
            
            # Plot all clustered points. We'll add a single label for all of them later.
            for cluster_id, color in cluster_id_to_color.items():
                subset = clustered_points[clustered_points['cluster_id'] == cluster_id]
                ax.scatter(subset['x_global'], subset['y_global'], subset['z_global'],
                           color=color, s=5, alpha=0.6) # No label here yet

            # Add a single legend entry for all clustered points
            # We plot a dummy point just for the legend entry.
            ax.scatter([], [], [], color='blue', s=5, alpha=0.6, label='Clustered Points') # Use a representative color for the legend
        else:
            print(f"[INFO] No clusters found (excluding noise) for plotting in {os.path.basename(f_path)}.")

        # Plot noise points
        if not noise_points.empty:
            ax.scatter(noise_points['x_global'], noise_points['y_global'], noise_points['z_global'],
                       color='grey', marker='x', s=10, alpha=0.8, label='Noise Points')
        else:
            print(f"[INFO] No noise points found for plotting in {os.path.basename(f_path)}.")

        ax.set_title(f'Data Point Distribution for {os.path.basename(f_path)}\n(Total Points: {len(df_plot)})')
        ax.set_xlabel('X_global')
        ax.set_ylabel('Y_global')
        ax.set_zlabel('Z_global')
        
        # Place legend for only 'Clustered Points' and 'Noise Points'
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), markerscale=2)

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for legend and title
        
        plot_filename = os.path.join(output_plots_directory, f"distribution_{os.path.basename(f_path).replace('.csv', '.png')}")
        plt.savefig(plot_filename)
        plt.close() # Close the plot to free memory

    print(f"\n[INFO] All individual distribution plots saved to '{output_plots_directory}/'.")

    # --- Part 3: Overall Distribution Summary Plot (Pie Chart) ---
    if total_points_overall > 0:
        total_clustered_points_overall = total_points_overall - total_noise_overall
        
        labels = ['Clustered Points', 'Noise Points']
        sizes = [total_clustered_points_overall, total_noise_overall]
        colors = ['#66b3ff', '#999999'] # Blue for clustered, grey for noise
        explode = (0.05, 0) # Explode the 'Clustered Points' slice slightly for emphasis

        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90, pctdistance=0.85)
        ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        ax_pie.set_title('Overall Data Point Distribution (Clustered vs. Noise) Across All Files')

        plot_filename_summary_pie = os.path.join(output_plots_directory, "overall_distribution_summary_pie.png")
        plt.savefig(plot_filename_summary_pie)
        plt.close()
        print(f"[INFO] Overall distribution summary pie chart saved to '{plot_filename_summary_pie}'.")
    else:
        print("[INFO] No data points to create overall distribution summary pie chart.")

    print("\nProcess completed.")

# --- Main execution block ---
if __name__ == "__main__":
    # Define your data directory
    clustered_data_directory = "/Users/meenandm/Documents/DBScan-for-Clustering/ttbar_mu100_strips"
    
    # Define where you want the plots to be saved
    output_plots_directory = "/Users/meenandm/Documents/DBScan-for-Clustering/cluster_visualizations"

    run_clustering_summary_and_visualization(clustered_data_directory, output_plots_directory)