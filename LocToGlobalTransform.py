import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

detector = pd.read_csv('tml_detector/trackml-detector.csv')


cx , cy , cz = detector['cx'], detector['cy'], detector['cz']

# Dictionary:

# {volume_id:7,8,9 module_t=0.15, pitch_u=0.05,pitch_v=0.05635}
# {volume_id:12,13,14 module_t=0.25, pitch_u=0.08,pitch_v=1.2}
# {volume_id:16,18 module_t=0.35, pitch_u=0.12, pitch_v=10.4}
# {volume_id:17 module_t=0.35, pitch_u=0.12, pitch_v=10.8}
# No information for volume_id:5,10


# Create new columns with default values
detector['module_t'] = 0.0
detector['pitch_u'] = 0.0
detector['pitch_v'] = 0.0

# Define the mapping of volume_id to module parameters
volume_params = {
    (5, 10): {'module_t': 0.0, 'pitch_u': 0.0, 'pitch_v': 0.0},
    (7, 8, 9): {'module_t': 0.15, 'pitch_u': 0.05, 'pitch_v': 0.05635},
    (12, 13, 14): {'module_t': 0.25, 'pitch_u': 0.08, 'pitch_v': 1.2},
    (16, 18): {'module_t': 0.35, 'pitch_u': 0.12, 'pitch_v': 10.4},
    (17,): {'module_t': 0.35, 'pitch_u': 0.12, 'pitch_v': 10.8},
}

# Apply mapping
for volume_ids, params in volume_params.items():
    mask = detector['volume_id'].isin(volume_ids)
    detector.loc[mask, ['module_t', 'pitch_u', 'pitch_v']]

# Coordinate transformation function
def compute_global_coordinates(row):
    x_local_3D = np.array([
        row['channel0'] * row['pitch_u'],
        row['channel1'] * row['pitch_v'],
        0
    ])
    R = np.array([
        [row['rot_xu'], row['rot_xv'], row['rot_xw']],
        [row['rot_yu'], row['rot_yv'], row['rot_yw']],
        [row['rot_zu'], row['rot_zv'], row['rot_zw']]
    ])
    t = np.array([row['cx'], row['cy'], row['cz']])
    x_global = np.dot(R, x_local_3D) + t
    return pd.Series({'x_global': x_global[0], 'y_global': x_global[1], 'z_global': x_global[2]})

# Directory with cell files
cell_folder = 'ttbar_mu100_strips'

# Process each CSV cell file
for filename in os.listdir(cell_folder):
    if filename.endswith('-cells.csv'):
        cell_path = os.path.join(cell_folder, filename)
        print(f"Processing {filename}")

        # Load cell data
        cell_file = pd.read_csv(cell_path)

        # Merge with detector info
        merged_df = pd.merge(cell_file, detector, on='geometry_id', how='inner')

        # Apply transformation
        global_coords = merged_df.apply(compute_global_coordinates, axis=1)
        merged_df = pd.concat([merged_df, global_coords], axis=1)

        # Save new file with "_global" suffix
        out_path = os.path.join(cell_folder, filename.replace('-cells.csv', '-cells-global.csv'))
        merged_df.to_csv(out_path, index=False)

        print(f"Saved transformed file to {out_path}")


# Example DataFrame structure
# df = pd.DataFrame({
#     'channel0': [...],  # Local x cell indices
#     'channel1': [...],  # Local y cell indices
#     'pitch_u': ...,     # Pitch in x direction
#     'pitch_v': ...,     # Pitch in y direction
#     'cx': ..., 'cy': ..., 'cz': ...,  # Translation components
#     'rot_xu': ..., 'rot_xv': ..., 'rot_xw': ...,  # Rotation components
#     'rot_yu': ..., 'rot_yv': ..., 'rot_yw': ...,
#     'rot_zu': ..., 'rot_zv': ..., 'rot_zw': ...
# })




# Load the global coordinates
global_coord_df = pd.read_csv('ttbar_mu100_strips/event000000003-cells-global.csv')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot x, y, z
ax.scatter(global_coord_df['x_global'], global_coord_df['y_global'],global_coord_df['z_global'], c='black', label='x, y, z', marker='o', s=3)

# Plot cx, cy, cz
ax.scatter(detector['cx'], detector['cy'], detector['cz'], c='red', label='cx, cy, cz', marker='x', s=12)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of x, y, z and cx, cy, cz')
ax.legend()
plt.show()