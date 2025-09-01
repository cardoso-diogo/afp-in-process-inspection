# main.py

import configparser
import time
import os
import json
from datetime import datetime, timezone
import threading
import queue
import re
import numpy as np

from src.hardware.llt_controller import LLTController
# CameraController is no longer needed
# from src.hardware.camera_controller import CameraController
from src.processing import data_saver
from py_openshowvar import openshowvar, ENCODING

# --- KUKA Parser and Helper Functions ---
KUKA_STRUCTURE_REGEX = re.compile(r"([A-Z][A-Z0-9]*)\s+([-\d\.E\+]+)")


def parse_kuka_structure(data_string):
    """Parses a KUKA structure string into a Python dictionary."""
    parsed_data = {}
    try:
        content = data_string[data_string.find('{') + 1:data_string.rfind('}')]
        if ':' in content: content = content[content.find(':') + 1:]
        content = content.replace(',', '')
        matches = KUKA_STRUCTURE_REGEX.findall(content)
        for key, value_str in matches:
            parsed_data[key] = float(value_str)
    except Exception:
        pass
    return parsed_data


def kuka_polling_thread(kuka_cfg, shared_state_data, pose_queue, lock, stop_event):
    """
    Polls the robot using a smart, tiered frequency approach.
    """
    kuka_client = None
    trigger_var = kuka_cfg.get('trigger_variable');
    position_var = '$POS_ACT_MES'
    layer_var, pass_var = 'LAYERNUM', 'PASSNUM'
    try:
        kuka_client = openshowvar(kuka_cfg.get('robot_ip'), kuka_cfg.getint('robot_port'))
        if not kuka_client.can_connect:
            print("[KUKA THREAD] ERROR: Could not connect. Thread exiting.")
            with lock: shared_state_data['connection_ok'] = False
            return
        print("[KUKA THREAD] Connected and running.")
        with lock:
            shared_state_data['connection_ok'] = True
        loop_counter = 0
        POLL_SLOW_DATA_INTERVAL = 100
        while not stop_event.is_set():
            start_time = time.perf_counter()
            trigger_bytes = kuka_client.read(trigger_var, debug=False)
            position_bytes = kuka_client.read(position_var, debug=False)
            read_timestamp = datetime.now(timezone.utc)
            is_active = trigger_bytes and trigger_bytes.decode(errors='ignore').strip().upper() == 'TRUE'
            pose = parse_kuka_structure(position_bytes.decode(errors='ignore')) if position_bytes else {}
            with lock:
                shared_state_data['is_acquisition_active'] = is_active
            if pose:
                try:
                    pose_queue.put_nowait((read_timestamp, pose))
                except queue.Full:
                    pose_queue.get_nowait()
                    pose_queue.put_nowait((read_timestamp, pose))
            if loop_counter % POLL_SLOW_DATA_INTERVAL == 0:
                layer_bytes = kuka_client.read(layer_var, debug=False)
                pass_bytes = kuka_client.read(pass_var, debug=False)
                layer = int(float(layer_bytes.decode(errors='ignore').strip())) if layer_bytes else shared_state_data[
                    'actual_layer']
                pass_num = int(float(pass_bytes.decode(errors='ignore').strip())) if pass_bytes else shared_state_data[
                    'actual_pass']
                with lock:
                    shared_state_data['actual_layer'] = layer
                    shared_state_data['actual_pass'] = pass_num
            loop_counter += 1
            elapsed = time.perf_counter() - start_time
            time.sleep(max(0, 0.005 - elapsed))
    except Exception as e:
        print(f"[KUKA THREAD] An error occurred: {e}")
        with lock:
            shared_state_data['connection_ok'] = False
    finally:
        if kuka_client: kuka_client.close(); print("[KUKA THREAD] Disconnected.")


def find_closest_pose(pose_list, target_timestamp):
    """Finds the pose in a list with the timestamp closest to the target."""
    if not pose_list: return None, None
    time_diffs = np.array([abs((ts - target_timestamp).total_seconds()) for ts, pose in pose_list])
    closest_idx = np.argmin(time_diffs)
    min_diff_ms = time_diffs[closest_idx] * 1000
    return pose_list[closest_idx][1], min_diff_ms


def main():
    # --- State and Data Storage Variables ---
    all_scan_data, robot_path_data, reference_x_values = [], [], None
    kuka_shared_data = {'is_acquisition_active': False, 'actual_layer': 0, 'actual_pass': 0, 'connection_ok': False}
    pose_queue = queue.Queue(maxsize=1000)
    kuka_lock, data_lock = threading.Lock(), threading.Lock()
    stop_kuka_thread = threading.Event()

    # --- Load Plan and Config ---
    try:
        with open('config/layup_plan.json', 'r') as f:
            layup_plan = json.load(f)
        config_ini_path = os.path.join(layup_plan.get('config_file', 'config.ini'))
        config = configparser.ConfigParser();
        config.read(config_ini_path)
        paths_cfg, llt_cfg, kuka_cfg = (config['FilePaths'], config['LLT'], config['KUKA'])
    except Exception as e:
        print(f"ERROR: Could not load configuration. Details: {e}");
        return

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = os.path.join(paths_cfg.get('output_directory', 'data/'),
                                f"{layup_plan['layup_id']}_{session_timestamp}")
    os.makedirs(session_path, exist_ok=True)
    print(f"--- Layup Session Initialized. Data will be saved to: {session_path} ---")

    target_freq_hz = llt_cfg.getint('target_frequency_hz');
    exposure_time_ctrl = llt_cfg.getint('exposure_time')
    calculated_idle_time_ctrl = int((100_000 / target_freq_hz) - exposure_time_ctrl)

    # Initialize Hardware and KUKA Thread
    llt = LLTController(exposure_time_ctrl, calculated_idle_time_ctrl)
    kuka_thread = threading.Thread(target=kuka_polling_thread,
                                   args=(kuka_cfg, kuka_shared_data, pose_queue, kuka_lock, stop_kuka_thread),
                                   daemon=True)
    kuka_thread.start()

    acquisition_in_progress = False
    current_layer_id, current_pass_id = None, None

    try:
        if not llt.connect():
            print("ERROR: Failed to connect to LLT hardware.");
            return

        print("\nInitialization complete. Waiting for trigger...")
        time.sleep(2)

        while True:
            with kuka_lock:
                robot_state = kuka_shared_data.copy()
            if not robot_state['connection_ok']: print("KUKA connection lost."); break

            if robot_state['is_acquisition_active'] and not acquisition_in_progress:
                acquisition_in_progress = True
                while not pose_queue.empty(): pose_queue.get_nowait()
                with data_lock:
                    all_scan_data.clear(); robot_path_data.clear()
                reference_x_values = None
                llt.start_transfer()
                current_layer_id, current_pass_id = robot_state['actual_layer'], robot_state['actual_pass']
                print(
                    f"\n>>> TRIGGER HIGH. Starting acquisition for Layer {current_layer_id}, Pass {current_pass_id}...")

            elif not robot_state['is_acquisition_active'] and acquisition_in_progress:
                acquisition_in_progress = False
                llt.stop_transfer()
                print("\n<<< TRIGGER LOW. Acquisition stopped.")
                if current_layer_id is not None:
                    pass_path = os.path.join(session_path, f"Layer_{current_layer_id:02d}",
                                             f"Pass_{current_pass_id:02d}")
                    os.makedirs(pass_path, exist_ok=True)
                    print(f"--- Saving data for Layer {current_layer_id}, Pass {current_pass_id} ---")
                    with data_lock:
                        scans_to_save, path_to_save, x_coords_to_save = \
                            list(all_scan_data), list(robot_path_data), list(
                                reference_x_values) if reference_x_values else None
                    if any([scans_to_save, path_to_save]):
                        pass_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        data_saver.save_raw_robot_data(pass_path, paths_cfg['robot_path_filename'], pass_timestamp,
                                                       path_to_save)
                        data_saver.save_raw_scan_data(pass_path, paths_cfg['z_data_filename'],
                                                      paths_cfg['x_coords_filename'], pass_timestamp, scans_to_save,
                                                      x_coords_to_save, llt.resolution)
                    else:
                        print("No data collected for this pass. Skipping save.")
                    print("-------------------------------------------------\n")
                print("Waiting for next trigger...")

            if acquisition_in_progress:
                profile_data = llt.acquire_profile()
                if profile_data:
                    pc_timestamp_obj = datetime.now(timezone.utc)
                    pc_timestamp_iso = pc_timestamp_obj.isoformat()

                    poses_in_queue = []
                    while not pose_queue.empty(): poses_in_queue.append(pose_queue.get_nowait())

                    best_pose, time_diff = find_closest_pose(poses_in_queue, pc_timestamp_obj)

                    with data_lock:
                        x_data, z_data = profile_data
                        if reference_x_values is None: reference_x_values = x_data
                        all_scan_data.append((pc_timestamp_iso, z_data))
                        if best_pose: robot_path_data.append((pc_timestamp_iso, best_pose))

            else:  # Idle state
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n--- User interrupted program ---")
    finally:
        print("\n--- Cleaning up resources ---")
        stop_kuka_thread.set()
        if 'kuka_thread' in locals() and kuka_thread.is_alive(): kuka_thread.join(timeout=2)
        if 'llt' in locals() and hasattr(llt, 'disconnect'): llt.disconnect()
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()