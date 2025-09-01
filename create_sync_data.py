# create_sync_data.py

import os
import glob
import pandas as pd
import configparser
import sys
from datetime import datetime


def find_latest_session_dir(data_dir):
    """Finds the most recently modified session directory."""
    # Get all subdirectories in the data directory
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs:
        return None
    # Return the one that was modified most recently
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir


def process_single_pass(pass_dir, paths_cfg):
    """
    Finds, loads, merges, and saves the data for a single pass.
    """
    print(f"\n--- Processing Pass: {os.path.basename(pass_dir)} ---")

    # 1. Find the raw data files using glob to handle varying timestamps
    robot_path_pattern = os.path.join(pass_dir, f"{paths_cfg.get('robot_path_filename')}_*.csv")
    laser_profile_pattern = os.path.join(pass_dir, f"{paths_cfg.get('z_data_filename')}_*.csv")

    robot_files = glob.glob(robot_path_pattern)
    laser_files = glob.glob(laser_profile_pattern)

    if not robot_files or not laser_files:
        print("  - Warning: Missing robot path or laser profile file. Skipping pass.")
        return

    # Use the most recent file if multiple exist for some reason
    robot_path_file = max(robot_files, key=os.path.getctime)
    laser_profile_file = max(laser_files, key=os.path.getctime)

    # 2. Load the data
    robot_df = pd.read_csv(robot_path_file)
    laser_df = pd.read_csv(laser_profile_file)
    print(f"  - Loaded {len(robot_df)} robot poses and {len(laser_df)} laser profiles.")

    # 3. Prepare DataFrames for Merging
    # Rename timestamp columns to a common name for merging
    robot_df.rename(columns={'robot_timestamp_iso': 'timestamp'}, inplace=True)
    laser_df.rename(columns={'timestamp_iso': 'timestamp'}, inplace=True)

    # Convert the 'timestamp' column to the same timezone-aware datetime format
    robot_df['timestamp'] = pd.to_datetime(robot_df['timestamp'], format='ISO8601', utc=True)
    laser_df['timestamp'] = pd.to_datetime(laser_df['timestamp'], format='ISO8601', utc=True)

    # Clean any rows that failed timestamp conversion (resulting in NaT)
    robot_df.dropna(subset=['timestamp'], inplace=True)
    laser_df.dropna(subset=['timestamp'], inplace=True)

    if robot_df.empty or laser_df.empty:
        print("  - Warning: No valid data after cleaning timestamps. Skipping pass.")
        return

    # 4. Perform the Time-Based Merge
    merged_df = pd.merge_asof(
        left=laser_df.sort_values('timestamp'),
        right=robot_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    merged_df.dropna(inplace=True)
    print(f"  - Successfully merged {len(merged_df)} synchronized records.")

    # 5. Save the Merged File
    sync_base_name = paths_cfg.get('sync_filename')
    # Use the pass's original timestamp for consistency
    pass_timestamp = os.path.basename(robot_path_file).replace(f"{paths_cfg.get('robot_path_filename')}_", "").replace(
        ".csv", "")
    output_filename = os.path.join(pass_dir, f"{sync_base_name}_{pass_timestamp}.csv")

    print(f"  - Saving synchronized data to: {os.path.basename(output_filename)}")
    merged_df.to_csv(output_filename, index=False)


def main():
    print("--- Step 1: Synchronizing Raw Data for All Passes in a Layup ---")

    # Load Configuration
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    paths_cfg = config['FilePaths']
    output_dir = paths_cfg.get('output_directory', 'data/')

    # Determine which session to process
    if len(sys.argv) > 1:
        session_path = sys.argv[1]
        if not os.path.isdir(session_path):
            print(f"Error: Provided path '{session_path}' is not a valid directory.")
            return
    else:
        session_path = find_latest_session_dir(output_dir)
        if not session_path:
            print(f"Error: Could not find any session directories in '{output_dir}'.")
            return

    print(f"\nProcessing Layup Session: {os.path.basename(session_path)}")

    # Find all 'Pass_XX' subdirectories
    pass_dirs = sorted(glob.glob(os.path.join(session_path, "Layer_*", "Pass_*")))

    if not pass_dirs:
        print("  - No 'Pass_XX' subdirectories found in this session.")
        return

    # Process each Pass found
    for pass_dir in pass_dirs:
        process_single_pass(pass_dir, paths_cfg)

    print("\nSynchronization process for all passes is complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()