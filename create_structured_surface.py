# create_structured_surface.py

import os
import glob
import pandas as pd
import numpy as np
import configparser
import sys
import json
from tqdm import tqdm
from scipy.spatial import KDTree

from src.processing import transformation


# --- Utility Functions ---

def find_latest_session_dir(data_dir):
    """Finds the most recently modified session directory."""
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return max(subdirs, key=os.path.getmtime) if subdirs else None


def get_latest_file_in_pass(pass_dir, base_name):
    """Gets the path of the latest file matching a basename in a pass directory."""
    search_path = os.path.join(pass_dir, f"{base_name}_*.csv")
    files = glob.glob(search_path)
    return max(files, key=os.path.getctime) if files else None


# --- Core Data Processing Functions ---

def load_points_for_layer(pass_dirs, paths_cfg, llt_calib_cfg):
    """
    Loads all points for a given list of pass directories (belonging to one layer).
    """
    if not pass_dirs:
        return None

    x_coords_file = get_latest_file_in_pass(pass_dirs[0], paths_cfg.get('x_coords_filename'))
    if not x_coords_file:
        print(f"  - Error: Could not find x_coordinates file in {pass_dirs[0]}.")
        return None
    x_coords = pd.read_csv(x_coords_file, dtype=np.float64).values.flatten()
    T_tcp_llt = transformation.create_llt_hand_eye_matrix(llt_calib_cfg)

    all_world_points = []
    for pass_dir in pass_dirs:
        sync_file = get_latest_file_in_pass(pass_dir, paths_cfg.get('sync_filename'))
        if not sync_file: continue

        merged_df = pd.read_csv(sync_file)
        profile_cols = [col for col in merged_df.columns if col.startswith('Z_Point_')]

        for _, row in merged_df.iterrows():
            T_base_tcp = transformation.create_robot_pose_matrix(row['robot_X'], row['robot_Y'], row['robot_Z'],
                                                                 row['robot_A'], row['robot_B'], row['robot_C'])
            z_values = pd.to_numeric(row[profile_cols].values, errors='coerce')
            valid_indices = ~np.isnan(z_values)
            if not np.any(valid_indices): continue

            world_points = transformation.map_llt_points_to_world_vectorized(
                x_coords[valid_indices], z_values[valid_indices], T_base_tcp, T_tcp_llt
            )
            all_world_points.append(world_points)

    if not all_world_points:
        return None

    unified_points = np.vstack(all_world_points)
    if len(unified_points) > 0:
        z_median = np.median(unified_points[:, 2])
        unified_points = unified_points[np.abs(unified_points[:, 2] - z_median) < 20]

    return unified_points


def interpolate_to_grid(points, grid_resolution=0.15):
    """
    Interpolates a cloud of 3D points onto a regular 2D grid using IDW.
    """
    if points is None or len(points) == 0:
        return None, None, None

    kdtree = KDTree(points[:, :2])
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    grid_x_coords = np.arange(x_min, x_max, grid_resolution)
    grid_y_coords = np.arange(y_min, y_max, grid_resolution)

    if len(grid_x_coords) == 0 or len(grid_y_coords) == 0:
        print("  - Warning: Not enough point spread to create a grid. Skipping layer.")
        return None, None, None

    grid_X, grid_Y = np.meshgrid(grid_x_coords, grid_y_coords)
    grid_points_flat = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T

    search_radius = grid_resolution * 2
    neighbor_indices_list = kdtree.query_ball_point(grid_points_flat, r=search_radius)
    interpolated_z_flat = np.full(grid_points_flat.shape[0], np.nan)

    for i in tqdm(range(len(grid_points_flat)), desc="Interpolating Grid Nodes", leave=False):
        neighbor_indices = neighbor_indices_list[i]
        if not neighbor_indices: continue

        neighbor_points = points[neighbor_indices]
        query_point_xy = grid_points_flat[i]
        distances = np.linalg.norm(neighbor_points[:, :2] - query_point_xy, axis=1)
        weights = 1.0 / (distances + 1e-9)
        interpolated_z_flat[i] = np.sum(neighbor_points[:, 2] * weights) / np.sum(weights)

    interpolated_Z = interpolated_z_flat.reshape(grid_X.shape)
    return grid_X, grid_Y, interpolated_Z


def save_structured_data(output_dir, layer_id, grid_X, grid_Y, grid_Z):
    """Saves the structured grid data for a single layer as a .npy file."""
    if grid_Z is None: return

    structured_surface = np.stack([grid_X, grid_Y, grid_Z], axis=-1)
    npy_filename = os.path.join(output_dir, f"structured_surface_layer_{layer_id:02d}.npy")
    np.save(npy_filename, structured_surface)
    print(f"  - Saved structured data to: {os.path.basename(npy_filename)}")

# --- Main Execution Block ---

def main():
    """
    Main execution function to process all layers in a session and save
    the structured output files.
    """
    print("--- Structured Surface Processor ---")
    try:
        with open('config/layup_plan.json', 'r') as f:
            layup_plan = json.load(f)
        config_ini_path = layup_plan.get('config_file', 'config/config.ini')
        config = configparser.ConfigParser()
        config.read(config_ini_path)
        paths_cfg = config['FilePaths']
        llt_calib_cfg = config['LLT_HandEye']
    except Exception as e:
        print(f"FATAL ERROR: Could not load configuration or layup plan. Details: {e}")
        return

    session_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_session_dir(
        paths_cfg.get('output_directory', 'data/'))
    if not session_path:
        print("Error: Could not find any session directory to process.")
        return
    print(f"\nProcessing Layup Session: {os.path.basename(session_path)}")

    # Process each layer defined in the layup plan
    for layer_info in layup_plan.get('layers', []):
        layer_id = layer_info.get('layer_id')
        print(f"\n--- Processing Layer {layer_id} ---")

        layer_dir = os.path.join(session_path, f"Layer_{layer_id:02d}")
        if not os.path.isdir(layer_dir):
            print(f"  - Warning: Directory not found for Layer {layer_id}. Skipping.")
            continue

        pass_dirs = sorted(glob.glob(os.path.join(layer_dir, "Pass_*")))

        # 1. Load raw points for this layer
        points = load_points_for_layer(pass_dirs, paths_cfg, llt_calib_cfg)
        if points is None:
            print(f"  - No points loaded for Layer {layer_id}. Skipping.")
            continue

        # 2. Interpolate points to a grid
        grid_X, grid_Y, grid_Z = interpolate_to_grid(points, grid_resolution=0.15)

        # 3. Save the structured data for this layer
        if grid_Z is not None:
            save_structured_data(layer_dir, layer_id, grid_X, grid_Y, grid_Z)
        else:
            print(f"  - Interpolation failed for Layer {layer_id}. No output file saved.")

    print("\n\nProcessing complete. All layers have been converted to structured .npy files.")
    print("You can now run 'inspect_structured_surface.py' to visualize the results.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected top-level error occurred: {e}")
        import traceback

        traceback.print_exc()