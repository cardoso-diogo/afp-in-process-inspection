# src/processing/data_saver.py

import csv
import numpy as np
from datetime import datetime, timezone
import os

def save_raw_scan_data(output_dir, base_z_name, base_x_name, timestamp, scan_data, x_coords, resolution):
    """
    Saves Z-profiles and X-coordinates to their respective, standard CSV files.
    """
    if not scan_data:
        print("Data Saver: No laser scan data to save.")
        return

    z_filename = os.path.join(output_dir, f"{base_z_name}_{timestamp}.csv")
    x_filename = os.path.join(output_dir, f"{base_x_name}_{timestamp}.csv")

    # --- Save Z-profiles ---
    print(f"Data Saver: Saving {len(scan_data)} raw Z-profiles to {z_filename}...")
    try:
        # Define a clear header. The first column is the timestamp.
        header_z = ['timestamp_iso'] + [f"Z_Point_{i}" for i in range(resolution)]

        with open(z_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header_z)

            # The writerow function handles the list correctly.
            # Each item in the list [pc_ts] + z_profile becomes a separate column.
            for pc_ts, z_profile in scan_data:
                writer.writerow([pc_ts] + z_profile)

        print(f"Data Saver: Z-data successfully saved.")

    except IOError as e:
        print(f"Data Saver: Error saving Z-data - {e}")

    # --- Save X-coordinates ---
    if x_coords:
        print(f"Data Saver: Saving reference X-coordinates to {x_filename}...")
        try:
            header_x = [f"X_Point_{i}" for i in range(len(x_coords))]
            with open(x_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header_x)
                writer.writerow(x_coords)
            print(f"Data Saver: X-coordinates successfully saved.")
        except IOError as e:
            print(f"Data Saver: Error saving X-coordinates - {e}")


def save_raw_robot_data(output_dir, base_name, timestamp, robot_data):
    """Saves the raw robot path data to a file."""
    if not robot_data: return

    filename = os.path.join(output_dir, f"{base_name}_{timestamp}.csv")
    print(f"Data Saver: Saving {len(robot_data)} raw robot poses to {filename}...")
    try:
        pose_keys = ['X', 'Y', 'Z', 'A', 'B', 'C']
        header = ['robot_timestamp_iso'] + [f'robot_{key}' for key in pose_keys]

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for ts_iso, value_dict in robot_data:
                row = [ts_iso] + [value_dict.get(key, 'NaN') for key in pose_keys]
                writer.writerow(row)
    except Exception as e:
        print(f"Data Saver: Error saving robot data - {e}")


def create_synchronized_file(output_dir, base_name, timestamp, scan_data, robot_data, x_coords):
    """Performs post-processing synchronization and saves the result."""
    if not all([scan_data, robot_data, x_coords]):
        print("Data Saver: Synchronization skipped due to missing data.")
        return

    filename = os.path.join(output_dir, f"{base_name}_{timestamp}.csv")
    print(f"Data Saver: Synchronizing data to {filename}...")
    try:
        robot_timestamps_iso = [item[0] for item in robot_data]
        robot_timestamps_unix = np.array([datetime.fromisoformat(ts).timestamp() for ts in robot_timestamps_iso])

        pose_keys = ['X', 'Y', 'Z', 'A', 'B', 'C']
        robot_poses = np.array([[item[1].get(key, np.nan) for key in pose_keys] for item in robot_data])

        header = ['pc_timestamp', 'robot_timestamp_estimated']
        header.extend([f'robot_{key}_interp' for key in pose_keys])
        header.extend([f"Z_Point_{i}" for i in range(len(x_coords))])
        header.extend([f"X_Point_{i}" for i in range(len(x_coords))])

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for pc_ts_iso, z_profile in scan_data:
                pc_ts_unix = datetime.fromisoformat(pc_ts_iso).timestamp()
                interp_pose = [np.interp(pc_ts_unix, robot_timestamps_unix, robot_poses[:, i], left=robot_poses[0, i],
                                         right=robot_poses[-1, i]) for i in range(len(pose_keys))]
                est_robot_ts_unix = np.interp(pc_ts_unix, robot_timestamps_unix, robot_timestamps_unix)
                est_robot_ts_iso = datetime.fromtimestamp(est_robot_ts_unix, tz=timezone.utc).isoformat()

                row = [pc_ts_iso, est_robot_ts_iso] + interp_pose + z_profile + x_coords
                writer.writerow(row)

        print(f"Data Saver: Synchronization successful.")
    except Exception as e:
        print(f"Data Saver: Synchronization failed - {e}")