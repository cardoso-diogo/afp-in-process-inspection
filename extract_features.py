# extract_features.py

import os
import glob
import pandas as pd
import numpy as np
import configparser
import sys
from tqdm import tqdm
from scipy.signal import savgol_filter


def find_latest_session_dir(data_dir):
    """Finds the most recently modified session directory."""
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs: return None
    return max(subdirs, key=os.path.getmtime)


def analyze_profile_for_edges(x_coords, z_values, pass_number):
    """
    Finds tow/gap edges using a state-aware algorithm with a final
    validation step based on minimum tow width.
    """
    features = {
        'gap_edge_x': np.nan, 'gap_edge_z': np.nan,
        'current_tow_left_edge_x': np.nan, 'current_tow_left_edge_z': np.nan,
        'current_tow_right_edge_x': np.nan, 'current_tow_right_edge_z': np.nan,
    }

    sensor_x_resolution_mm = np.abs(np.mean(np.diff(x_coords)))
    MIN_TOW_WIDTH_MM = 6.0
    MIN_WIDTH_IN_INDICES = int(MIN_TOW_WIDTH_MM / sensor_x_resolution_mm)

    try:
        # 1. Pre-processing & Smoothing
        z_processed = np.copy(z_values);
        z_processed[z_processed < 1.0] = np.nan
        z_inpainted = pd.Series(z_processed).interpolate(method='linear', limit_direction='both', limit=50).to_numpy()
        smoothed_z = savgol_filter(z_inpainted, 41, 3)

        # 2. Create a Binary Profile
        valid_z = smoothed_z[~np.isnan(smoothed_z)]
        if len(valid_z) < 100: return features
        z_floor, z_ceiling = np.percentile(valid_z, 10), np.percentile(valid_z, 90)
        if (z_ceiling - z_floor) < 0.1: return features
        z_mid_threshold = z_floor + (z_ceiling - z_floor) * 0.5
        binary_profile = (smoothed_z > z_mid_threshold).astype(int)

        # 3. Find all edges
        diff = np.diff(binary_profile)
        rising_edge_indices = np.where(diff == 1)[0]
        falling_edge_indices = np.where(diff == -1)[0]
        if rising_edge_indices.size == 0 or falling_edge_indices.size == 0: return features

        # 4. State-aware logic to find the CANDIDATE edges
        current_tow_left_idx, current_tow_right_idx, gap_edge_idx = None, None, None

        if pass_number == 1:
            if rising_edge_indices.size > 0:
                left_idx = rising_edge_indices[0]
                possible_right_edges = falling_edge_indices[falling_edge_indices > left_idx]
                if possible_right_edges.size > 0:
                    current_tow_left_idx = left_idx
                    current_tow_right_idx = possible_right_edges[0]
        else:  # Pass > 1
            if rising_edge_indices.size >= 2:
                current_tow_left_idx = rising_edge_indices[0]
                gap_edge_idx = rising_edge_indices[1]
                possible_current_right = falling_edge_indices[falling_edge_indices > current_tow_left_idx]
                if possible_current_right.size > 0:
                    current_tow_right_idx = possible_current_right[0]

        # 5. VALIDATION STEP
        if current_tow_left_idx is not None and current_tow_right_idx is not None:
            width_in_indices = current_tow_right_idx - current_tow_left_idx
            if width_in_indices < MIN_WIDTH_IN_INDICES:
                return features  # Reject tow if it's too narrow

            # 6. ASSIGN FEATURES (only if validation passed)
            features['current_tow_left_edge_x'] = x_coords[current_tow_left_idx]
            features['current_tow_left_edge_z'] = smoothed_z[current_tow_left_idx]
            features['current_tow_right_edge_x'] = x_coords[current_tow_right_idx]
            features['current_tow_right_edge_z'] = smoothed_z[current_tow_right_idx]

            if gap_edge_idx is not None:
                features['gap_edge_x'] = x_coords[gap_edge_idx]
                features['gap_edge_z'] = smoothed_z[gap_edge_idx]

    except Exception:
        pass
    return features


def remove_outliers_by_iqr(df, column, factor=1.5):
    if df.empty or column not in df.columns: return df
    Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    IQR = Q3 - Q1;
    lower_bound, upper_bound = Q1 - (factor * IQR), Q3 + (factor * IQR)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def process_pass_for_features(pass_dir, paths_cfg):
    print(f"\n--- Extracting Features for: {os.path.basename(pass_dir)} ---")
    try:
        pass_number = int(os.path.basename(pass_dir).replace('Pass_', ''))
        print(f"  - Detected as Pass #{pass_number}.")
    except ValueError:
        print(f"  - Warning: Could not determine pass number from folder name. Skipping.");
        return

    sync_pattern = os.path.join(pass_dir, f"{paths_cfg.get('sync_filename')}_*.csv")
    x_coords_pattern = os.path.join(pass_dir, f"{paths_cfg.get('x_coords_filename')}_*.csv")
    sync_files, x_coords_files = glob.glob(sync_pattern), glob.glob(x_coords_pattern)
    if not sync_files or not x_coords_files:
        print("  - Warning: Missing input files. Skipping pass.");
        return

    sync_file, x_coords_file = max(sync_files, key=os.path.getctime), max(x_coords_files, key=os.path.getctime)
    merged_df = pd.read_csv(sync_file)

    raw_x_coords = pd.read_csv(x_coords_file, dtype=np.float64).values.flatten()
    raw_x_coords[raw_x_coords == 0] = np.nan
    x_coords = pd.Series(raw_x_coords).interpolate(method='linear', limit_direction='both').to_numpy()

    profile_cols = [col for col in merged_df.columns if col.startswith('Z_Point_')]
    print(f"  - Loaded {len(merged_df)} synchronized profiles.")

    all_features = []
    for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="  - Analyzing Profiles"):
        z_values = pd.to_numeric(row[profile_cols].values, errors='coerce')
        features = analyze_profile_for_edges(x_coords, z_values, pass_number)
        if pd.notna(features['current_tow_left_edge_x']):
            features['timestamp'] = row['timestamp']
            all_features.append(features)

    if not all_features:
        print("  - Warning: No valid tow features were extracted from any profile in this pass.")
        return

    features_df = pd.DataFrame(all_features)
    print(f"  - Initially extracted features from {len(features_df)} of {len(merged_df)} profiles.")

    features_df['tow_width'] = features_df['current_tow_left_edge_x'] - features_df['current_tow_right_edge_x']
    initial_count = len(features_df)
    features_df_cleaned = remove_outliers_by_iqr(features_df, 'tow_width')
    removed_count = initial_count - len(features_df_cleaned)
    if removed_count > 0:
        print(f"  - Removed {removed_count} outliers based on inconsistent tow width.")
    features_df_cleaned = features_df_cleaned.drop(columns=['tow_width'])

    features_base_name = paths_cfg.get('laser_features_filename')
    pass_timestamp = os.path.basename(sync_file).replace(f"{paths_cfg.get('sync_filename')}_", "").replace(".csv", "")
    output_filename = os.path.join(pass_dir, f"{features_base_name}_{pass_timestamp}.csv")
    print(f"  - Saving {len(features_df_cleaned)} cleaned features to: {os.path.basename(output_filename)}")
    features_df_cleaned.to_csv(output_filename, index=False, float_format='%.4f')


def main():
    print("--- Step 2: Extracting and Cleaning Tow Edge Features ---")
    config = configparser.ConfigParser();
    config.read('config/config.ini')
    paths_cfg = config['FilePaths']
    session_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_session_dir(
        paths_cfg.get('output_directory', 'data/'))
    if not session_path or not os.path.isdir(session_path):
        print(f"Error: Session directory not found: {session_path}");
        return
    print(f"\nProcessing Layup Session: {os.path.basename(session_path)}")
    pass_dirs = sorted(glob.glob(os.path.join(session_path, "Layer_*", "Pass_*")))
    if not pass_dirs:
        print("  - No 'Pass_XX' subdirectories found in this session.");
        return
    for pass_dir in pass_dirs:
        process_pass_for_features(pass_dir, paths_cfg)
    print("\nFeature extraction for all passes is complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {e}"); import traceback; traceback.print_exc()