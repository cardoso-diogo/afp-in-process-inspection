# transform_features.py
import os
import glob
import pandas as pd
import numpy as np
import configparser
import sys
import json
from tqdm import tqdm

# This assumes your transformation.py has the simplified, corrected
# create_llt_hand_eye_matrix function that does NOT take layup_angle_deg.
from src.processing import transformation


def find_latest_session_dir(data_dir):
    """Finds the most recently modified session directory."""
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs: return None
    return max(subdirs, key=os.path.getmtime)


def get_latest_file_in_pass(pass_dir, base_name):
    """Gets the path of the latest file matching a basename in a pass directory."""
    search_path = os.path.join(pass_dir, f"{base_name}_*.csv")
    files = glob.glob(search_path)
    if not files: return None
    return max(files, key=os.path.getctime)


def process_pass_for_transformation(pass_dir, paths_cfg, llt_calib_cfg):
    """
    Loads features and poses for a pass, and transforms features to world coordinates.
    """
    print(
        f"\n--- Transforming Features for: {os.path.basename(os.path.dirname(pass_dir))}/{os.path.basename(pass_dir)} ---")

    features_file = get_latest_file_in_pass(pass_dir, paths_cfg.get('laser_features_filename'))
    sync_file = get_latest_file_in_pass(pass_dir, paths_cfg.get('sync_filename'))
    if not features_file or not sync_file:
        print("  - Warning: Missing features or synchronized scan file. Skipping pass.");
        return

    features_df = pd.read_csv(features_file)
    sync_df = pd.read_csv(sync_file)
    print(f"  - Loaded {len(features_df)} feature sets to transform.")

    # Merge features with robot poses using the timestamp
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], utc=True)
    sync_df['timestamp'] = pd.to_datetime(sync_df['timestamp'], utc=True)
    pose_cols = ['timestamp'] + [f'robot_{axis}' for axis in ['X', 'Y', 'Z', 'A', 'B', 'C']]
    merged_df = pd.merge_asof(
        left=features_df.sort_values('timestamp'),
        right=sync_df[pose_cols].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    ).dropna(subset=pose_cols)

    if merged_df.empty:
        print("  - Warning: Could not merge features with robot poses. Skipping.");
        return

    T_tcp_llt = transformation.create_llt_hand_eye_matrix(llt_calib_cfg)

    transformed_results = []

    feature_points_map = {
        'gap_edge': ('gap_edge_x', 'gap_edge_z'),
        'current_tow_left_edge': ('current_tow_left_edge_x', 'current_tow_left_edge_z'),
        'current_tow_right_edge': ('current_tow_right_edge_x', 'current_tow_right_edge_z'),
    }

    for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="  - Transforming Features"):
        T_base_tcp = transformation.create_robot_pose_matrix(
            row['robot_X'], row['robot_Y'], row['robot_Z'],
            row['robot_A'], row['robot_B'], row['robot_C']
        )
        result = {'timestamp': row['timestamp']}
        for feature_name, (x_col, z_col) in feature_points_map.items():
            if x_col in row and pd.notna(row[x_col]) and pd.notna(row[z_col]):
                world_point = transformation.map_llt_point_to_world(
                    row[x_col], row[z_col], T_base_tcp, T_tcp_llt
                )
                result[f'{feature_name}_world_X'] = world_point[0]
                result[f'{feature_name}_world_Y'] = world_point[1]
                result[f'{feature_name}_world_Z'] = world_point[2]
        transformed_results.append(result)

    # Save the transformed features
    transformed_df = pd.DataFrame(transformed_results)
    print(f"  - Successfully transformed {len(transformed_df)} feature sets.")

    transformed_base_name = paths_cfg.get('transformed_laser_filename')
    pass_timestamp = os.path.basename(features_file).replace(f"{paths_cfg.get('laser_features_filename')}_",
                                                             "").replace(".csv", "")
    output_filename = os.path.join(pass_dir, f"{transformed_base_name}_{pass_timestamp}.csv")

    print(f"  - Saving transformed features to: {os.path.basename(output_filename)}")
    transformed_df.to_csv(output_filename, index=False, float_format='%.4f')


def main():
    print("--- Step 3: Transforming Extracted Features to World Coordinates ---")

    try:
        config_ini_path = os.path.join('config', 'config.ini')
        config = configparser.ConfigParser()
        if not config.read(config_ini_path):
            raise FileNotFoundError(f"Config file not found or is empty at: {config_ini_path}")

        paths_cfg = config['FilePaths']
        llt_calib_cfg = config['LLT_HandEye']
    except Exception as e:
        print(f"ERROR: Could not load configuration. Details: {e}")
        return

    output_dir = paths_cfg.get('output_directory', 'data/')
    session_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_session_dir(output_dir)
    if not session_path or not os.path.isdir(session_path):
        print("Error: Session directory not found.")
        return
    print(f"\nProcessing Layup Session: {os.path.basename(session_path)}")

    pass_dirs = sorted(glob.glob(os.path.join(session_path, "Layer_*", "Pass_*")))
    if not pass_dirs:
        print("  - No 'Pass_XX' subdirectories found in this session.")
        return

    for pass_dir in pass_dirs:
        try:
            process_pass_for_transformation(pass_dir, paths_cfg, llt_calib_cfg)
        except Exception as e:
            print(f"Error processing {pass_dir}: {e}")
            continue

    print("\nFeature transformation for all passes is complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()