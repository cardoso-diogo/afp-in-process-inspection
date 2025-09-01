#compute_fidelity_metrics.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
import json
import configparser


def find_latest_session_dir(data_dir):
    """Finds the most recently modified session directory."""
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def get_latest_file_by_basename(pass_dir, base_name):
    """Finds the latest file in a pass directory matching a base name from the config."""
    if not base_name:
        print("  - Error: Base name for file search is empty.")
        return None
    search_path = os.path.join(pass_dir, f"{base_name}_*.csv")
    files = glob.glob(search_path)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def get_orientation_from_plan(layup_plan, layer_id, pass_id):
    """Gets the orientation for a specific pass from the layup plan."""
    try:
        for layer in layup_plan.get('layers', []):
            if layer.get('layer_id') == layer_id:
                for p in layer.get('passes', []):
                    if p.get('pass_id') == pass_id:
                        return p.get('orientation', 0.0)
    except (KeyError, TypeError):
        return 0.0
    return 0.0


def calculate_fidelity_metrics(df, orientation):
    """
    Calculates LPD and other metrics based on the pass orientation.
    """
    left_cols = ['current_tow_left_edge_world_X', 'current_tow_left_edge_world_Y', 'current_tow_left_edge_world_Z']
    right_cols = ['current_tow_right_edge_world_X', 'current_tow_right_edge_world_Y', 'current_tow_right_edge_world_Z']

    df = df.dropna(subset=left_cols + right_cols).copy()

    if df.empty:
        return pd.DataFrame()

    left_points = df[left_cols].values
    right_points = df[right_cols].values
    centerline = (left_points + right_points) / 2.0

    df['centerline_X'] = centerline[:, 0]
    df['centerline_Y'] = centerline[:, 1]
    df['centerline_Z'] = centerline[:, 2]

    if orientation == 0:
        ideal_Y = df['centerline_Y'].mean()
        df['LPD_mm'] = (df['centerline_Y'] - ideal_Y).abs()
        df['position_along_path_mm'] = df['centerline_X'] - df['centerline_X'].min()
    elif orientation == 90:
        ideal_X = df['centerline_X'].mean()
        df['LPD_mm'] = (df['centerline_X'] - ideal_X).abs()
        df['position_along_path_mm'] = df['centerline_Y'] - df['centerline_Y'].min()
    else:
        df['LPD_mm'] = 0

    return df


def process_pass(pass_dir, layup_plan, paths_cfg):
    """Processes a single pass folder."""
    try:
        layer_name = os.path.basename(os.path.dirname(pass_dir))
        pass_name = os.path.basename(pass_dir)
        layer_id = int(layer_name.replace('Layer_', ''))
        pass_id = int(pass_name.replace('Pass_', ''))
    except (ValueError, IndexError):
        print(f"Warning: Could not parse layer/pass ID from path '{pass_dir}'. Skipping.")
        return

    orientation = get_orientation_from_plan(layup_plan, layer_id, pass_id)
    print(f"\n--- Processing {layer_name}/{pass_name} (Orientation: {orientation}°) ---")

    transformed_base_name = paths_cfg.get('transformed_laser_filename')
    input_file = get_latest_file_by_basename(pass_dir, transformed_base_name)

    if not input_file:
        print(f"  - No '{transformed_base_name}_*.csv' file found. Skipping.")
        return

    print(f"  - Reading: {os.path.basename(input_file)}")
    df = pd.read_csv(input_file)
    df_fidelity = calculate_fidelity_metrics(df, orientation)

    if df_fidelity.empty:
        print("  - No valid data to process.")
        return

    output_filename = os.path.join(pass_dir, "fidelity_metrics.csv")
    df_fidelity.to_csv(output_filename, index=False, float_format='%.4f')
    print(f"  - Saved fidelity metrics to: {os.path.basename(output_filename)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    df_fidelity = df_fidelity.sort_values('position_along_path_mm')

    ax1.plot(df_fidelity['position_along_path_mm'], df_fidelity['LPD_mm'], 'b-', label='Lateral Path Deviation (LPD)')
    ax1.set_ylabel('LPD (mm)')
    ax1.set_title(f'Path Fidelity Analysis for {layer_name}/{pass_name}')
    ax1.grid(True);
    ax1.legend()

    ax2.plot(df_fidelity['position_along_path_mm'], df_fidelity['centerline_Z'], 'r-',
             label='Vertical Profile (Z-height)')
    ax2.set_xlabel('Position Along Path (mm)')
    ax2.set_ylabel('World Z-Coordinate (mm)')
    ax2.grid(True);
    ax2.legend()

    plot_filename = os.path.join(pass_dir, 'fidelity_analysis_plot.png')
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    print(f"  - Analysis plot saved to: {os.path.basename(plot_filename)}")
    plt.close(fig)


def main():
    """
    Main function to automatically find the latest session and process all passes.
    """
    try:
        with open('config/layup_plan.json', 'r') as f:
            layup_plan = json.load(f)
        config = configparser.ConfigParser()
        config_path = layup_plan.get('config_file', 'config/config.ini')
        config.read(config_path)
        if not config.sections():
            raise FileNotFoundError(f"Config file not found or is empty: {config_path}")
        paths_cfg = config['FilePaths']
    except Exception as e:
        print(f"FATAL ERROR: Could not load configuration files. Details: {e}")
        return

    session_path = find_latest_session_dir(paths_cfg.get('output_directory', 'data/'))
    if not session_path:
        print("Error: Could not find any session directory to process.")
        return

    print(f"Analyzing latest session: {os.path.basename(session_path)}")
    pass_dirs = sorted(glob.glob(os.path.join(session_path, "Layer_*", "Pass_*")))
    if not pass_dirs:
        print("  - No 'Pass_XX' subdirectories found in this session.")
        return

    for pass_dir in pass_dirs:
        process_pass(pass_dir, layup_plan, paths_cfg)

    print("\nFidelity analysis for all passes is complete.")


if __name__ == '__main__':
    main()