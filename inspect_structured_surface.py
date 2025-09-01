# inspect_structured_surface.py

import os
import glob
import numpy as np
import configparser
import sys
import json
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

from enum import Enum

from src.processing import transformation


# --- Utility Functions ---
def find_latest_session_dir(data_dir):
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return max(subdirs, key=os.path.getmtime) if subdirs else None


def get_latest_file_in_pass(pass_dir, base_name):
    search_path = os.path.join(pass_dir, f"{base_name}_*.csv")
    files = glob.glob(search_path)
    return max(files, key=os.path.getctime) if files else None


def print_help():
    print("""
--- Interactive Inspector Controls ---
(3D Window must be in focus)
  - LEFT/RIGHT Arrow Keys: Move the slicer.
  - 'N' Key: Cycle through Layers and the 'Global View'.
  - 'O' Key: (In Global View only) Toggle slicer orientation (X/Y).
  - 'V' Key: Toggle 2D plot mode (Hybrid / Solid).
  - 'Q' Key: Quit the application.
------------------------------------""")


class PlotMode(Enum):
    HYBRID = 1
    SOLID = 2


# --- Data Loading and Computation Functions ---
def load_and_filter_raw_points(pass_dirs, paths_cfg, llt_calib_cfg):
    if not pass_dirs: return None, None
    x_coords_file = get_latest_file_in_pass(pass_dirs[0], paths_cfg.get('x_coords_filename'))
    if not x_coords_file: return None, None
    x_coords = pd.read_csv(x_coords_file, dtype=np.float64).values.flatten()
    T_tcp_llt = transformation.create_llt_hand_eye_matrix(llt_calib_cfg)
    all_world_points, profile_indices, current_profile_index = [], [], 0
    for pass_dir in pass_dirs:
        sync_file = get_latest_file_in_pass(pass_dir, paths_cfg.get('sync_filename'))
        if not sync_file: continue
        merged_df = pd.read_csv(sync_file)
        profile_cols = [col for col in merged_df.columns if col.startswith('Z_Point_')]
        for _, row in merged_df.iterrows():
            T_base_tcp = transformation.create_robot_pose_matrix(row['robot_X'], row['robot_Y'], row['robot_Z'],
                                                                 row['robot_A'], row['robot_B'], row['robot_C'])
            z_values = pd.to_numeric(row[profile_cols].values, errors='coerce')
            valid_mask = ~np.isnan(z_values)
            if not np.any(valid_mask): continue
            world_points_profile = transformation.map_llt_points_to_world_vectorized(x_coords[valid_mask],
                                                                                     z_values[valid_mask], T_base_tcp,
                                                                                     T_tcp_llt)
            all_world_points.append(world_points_profile)
            profile_indices.extend([current_profile_index] * len(world_points_profile))
            current_profile_index += 1
    if not all_world_points: return None, None
    unfiltered_points, unfiltered_indices = np.vstack(all_world_points), np.array(profile_indices)
    height_threshold = 60.0
    keep_mask = unfiltered_points[:, 2] > height_threshold
    filtered_points, filtered_indices = unfiltered_points[keep_mask], unfiltered_indices[keep_mask]
    removed_count = len(unfiltered_points) - len(filtered_points)
    if removed_count > 0: print(f"    - Filtered out {removed_count:,} low-lying noise points.")
    return filtered_points, filtered_indices


def compute_intra_profile_colors(raw_points, profile_indices):
    num_points = len(raw_points)
    colors = np.zeros((num_points, 3))
    cmap = plt.get_cmap("jet")
    unique_profiles = np.unique(profile_indices)
    for p_idx in tqdm(unique_profiles, desc="Coloring Profiles", leave=False):
        profile_mask = (profile_indices == p_idx)
        z_values_in_profile = raw_points[profile_mask, 2]
        if len(z_values_in_profile) < 20: continue
        z_min_local, z_max_local = np.percentile(z_values_in_profile, 5), np.percentile(z_values_in_profile, 95)
        if (z_max_local - z_min_local) < 1e-6:
            norm_z_values = np.full_like(z_values_in_profile, 0.5)
        else:
            clipped_z = np.clip(z_values_in_profile, z_min_local, z_max_local)
            norm_z_values = (clipped_z - z_min_local) / (z_max_local - z_min_local)
        colors[profile_mask] = cmap(norm_z_values)[:, :3]
    return colors


# --- The Main Visualizer Class ---
class LinkedSlicingVisualizer:
    def __init__(self, layer_structured_data, layer_raw_data, layer_colors, layer_orientations):
        # Data setup
        self.layer_structured_data, self.layer_raw_data, self.layer_colors, self.layer_orientations = layer_structured_data, layer_raw_data, layer_colors, layer_orientations
        self.layer_ids = sorted(self.layer_structured_data.keys())
        self.global_points = np.vstack([self.layer_raw_data[lid] for lid in self.layer_ids])
        self.global_colors = np.vstack([self.layer_colors[lid] for lid in self.layer_ids])

        # State Management
        self.view_mode_idx, self.global_slicer_orientation = 0, 'X'
        self.plot_mode = PlotMode.HYBRID
        self.slicer_thickness, self.slicer_step = 1.0, 1.0

        # Visualization Objects
        self.vis_3d = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd, self.slicer_vis = o3d.geometry.PointCloud(), o3d.geometry.LineSet()
        self.pcd_in_scene, self.slicer_in_scene = False, False

        # 2D Plot Setup - with separate objects for each mode
        plt.ion()
        self.fig_2d, self.ax_2d = plt.subplots(figsize=(10, 6))
        self.scatter_plot = self.ax_2d.scatter([], [], s=2, alpha=0.3, edgecolors='none', label='Raw Data')
        self.line_plot, = self.ax_2d.plot([], [], color='magenta', linewidth=1.0, label='Interpolated Surface')
        self.solid_fills = []  # To hold polygon objects for solid view
        self.ax_2d.grid(True);

    def run(self):
        self.vis_3d.create_window("3D Inspector", width=1280, height=720)
        self.vis_3d.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=100))
        self.vis_3d.register_key_callback(ord('V'), self.toggle_plot_mode)
        self.vis_3d.register_key_callback(ord('N'), self.cycle_view_mode)
        self.vis_3d.register_key_callback(ord('O'), self.toggle_slicer_orientation)
        self.vis_3d.register_key_callback(262, self.move_slicer_forward)
        self.vis_3d.register_key_callback(263, self.move_slicer_backward)
        self.vis_3d.register_key_callback(ord('Q'), self.quit)
        self.set_active_view(is_initial_run=True)
        print_help()
        self.vis_3d.run()
        self.vis_3d.destroy_window()
        plt.ioff();
        plt.close(self.fig_2d)
        print("\nInspector closed.")

    def set_active_view(self, is_initial_run=False):
        if self.pcd_in_scene: self.vis_3d.remove_geometry(self.pcd, reset_bounding_box=False)
        if self.view_mode_idx < len(self.layer_ids):
            layer_id = self.layer_ids[self.view_mode_idx]
            print(f"\n--- Switched to Layer {layer_id} (Orientation: {self.layer_orientations[layer_id]}°) ---")
            points, colors = self.layer_raw_data[layer_id], self.layer_colors[layer_id]
        else:
            print(f"\n--- Switched to Global View ---")
            points, colors = self.global_points, self.global_colors
        self.pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        if not self.slicer_in_scene: self.vis_3d.add_geometry(self.slicer_vis,
                                                              reset_bounding_box=False); self.slicer_in_scene = True
        self.vis_3d.add_geometry(self.pcd, reset_bounding_box=True)
        self.pcd_in_scene = True
        self.vis_3d.reset_view_point(True)
        self.reset_slicer_and_axes_for_new_view()

    def reset_slicer_and_axes_for_new_view(self):
        points = np.asarray(self.pcd.points)
        if len(points) == 0:
            self.bounds = {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0, 'z_min': 0, 'z_max': 0}
        else:
            self.bounds = {'x_min': points[:, 0].min(), 'x_max': points[:, 0].max(),
                           'y_min': points[:, 1].min(), 'y_max': points[:, 1].max(),
                           'z_min': points[:, 2].min(), 'z_max': points[:, 2].max()}

        # Determine plot limits from structured data for consistency
        if self.view_mode_idx < len(self.layer_ids):
            structured_data = self.layer_structured_data[self.layer_ids[self.view_mode_idx]]
            x_range = (np.nanmin(structured_data[:, :, 0]), np.nanmax(structured_data[:, :, 0]))
            y_range = (np.nanmin(structured_data[:, :, 1]), np.nanmax(structured_data[:, :, 1]))
            z_range = (np.nanmin(structured_data[:, :, 2]), np.nanmax(structured_data[:, :, 2]))
        else:  # Global view uses raw point bounds as there's no global structured data
            x_range, y_range, z_range = (self.bounds['x_min'], self.bounds['x_max']), (
            self.bounds['y_min'], self.bounds['y_max']), (self.bounds['z_min'], self.bounds['z_max'])

        orientation = self.get_current_slicer_orientation()
        if orientation == 'X':
            self.ax_2d.set_xlim(y_range[0] - 5, y_range[1] + 5)
        else:
            self.ax_2d.set_xlim(x_range[0] - 5, x_range[1] + 5)
        self.ax_2d.set_ylim(z_range[0] - 1, z_range[1] + 1)

        # Reset slicer position
        if self.view_mode_idx < len(self.layer_ids):
            layer_orientation_code = self.layer_orientations[self.layer_ids[self.view_mode_idx]]
            if layer_orientation_code == 0:
                self.slicer_pos = self.bounds['x_min']
            else:
                self.slicer_pos = self.bounds['y_min']
        else:
            if self.global_slicer_orientation == 'X':
                self.slicer_pos = self.bounds['x_min']
            else:
                self.slicer_pos = self.bounds['y_min']

        self.update_slicer_geometry();
        self.update_2d_plot()

    def update_slicer_geometry(self):
        orientation = self.get_current_slicer_orientation()
        padding = 50
        min_b = [self.bounds['x_min'] - padding, self.bounds['y_min'] - padding, self.bounds['z_min'] - padding]
        max_b = [self.bounds['x_max'] + padding, self.bounds['y_max'] + padding, self.bounds['z_max'] + padding]
        if orientation == 'X':
            min_b[0], max_b[0] = self.slicer_pos, self.slicer_pos + self.slicer_thickness
        else:
            min_b[1], max_b[1] = self.slicer_pos, self.slicer_pos + self.slicer_thickness
        slicer_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
        new_slicer_vis = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(slicer_bbox)
        new_slicer_vis.paint_uniform_color([1, 0, 0])
        self.slicer_vis.points, self.slicer_vis.lines, self.slicer_vis.colors = new_slicer_vis.points, new_slicer_vis.lines, new_slicer_vis.colors
        if self.slicer_in_scene: self.vis_3d.update_geometry(self.slicer_vis)

    def update_2d_plot(self):
        # Hide all plot elements initially
        self.scatter_plot.set_visible(False)
        self.line_plot.set_visible(False)
        for fill in self.solid_fills: fill.remove()
        self.solid_fills.clear()

        # Get orientation and data
        orientation = self.get_current_slicer_orientation()
        if self.view_mode_idx < len(self.layer_ids):
            layer_id = self.layer_ids[self.view_mode_idx]
            points, colors, structured_data = self.layer_raw_data[layer_id], self.layer_colors[layer_id], \
            self.layer_structured_data[layer_id]
            title_prefix = f"Layer {layer_id}"
        else:
            points, colors, structured_data = self.global_points, self.global_colors, None
            title_prefix = "Global View"

        # Set title and labels
        if orientation == 'X':
            title = f"{title_prefix} | Cross-Section at X = {self.slicer_pos:.2f} mm";
            xlabel = "World Y (mm)"
        else:
            title = f"{title_prefix} | Cross-Section at Y = {self.slicer_pos:.2f} mm";
            xlabel = "World X (mm)"
        self.ax_2d.set_title(title);
        self.ax_2d.set_xlabel(xlabel);
        self.ax_2d.set_ylabel("World Z (mm)")

        # Logic for HYBRID mode
        if self.plot_mode == PlotMode.HYBRID:
            self.scatter_plot.set_visible(True)
            self.line_plot.set_visible(True)
            # Update raw points
            if orientation == 'X':
                slice_mask = (points[:, 0] >= self.slicer_pos) & (
                            points[:, 0] < self.slicer_pos + self.slicer_thickness)
                raw_plot_data = points[slice_mask][:, 1:3]
            else:
                slice_mask = (points[:, 1] >= self.slicer_pos) & (
                            points[:, 1] < self.slicer_pos + self.slicer_thickness)
                raw_plot_data = points[slice_mask][:, [0, 2]]
            colors_in_slice = colors[slice_mask]
            if len(raw_plot_data) > 0:
                self.scatter_plot.set_offsets(raw_plot_data); self.scatter_plot.set_color(colors_in_slice)
            else:
                self.scatter_plot.set_offsets(np.empty((0, 2)))

            # Update interpolated line
            if structured_data is not None:
                if orientation == 'X':
                    slice_index = np.argmin(np.abs(structured_data[0, :, 0] - self.slicer_pos))
                    interp_cross = structured_data[:, slice_index, :];
                    main_axis, z_coords = interp_cross[:, 1], interp_cross[:, 2]
                else:
                    slice_index = np.argmin(np.abs(structured_data[:, 0, 1] - self.slicer_pos))
                    interp_cross = structured_data[slice_index, :, :];
                    main_axis, z_coords = interp_cross[:, 0], interp_cross[:, 2]
                valid_indices = ~np.isnan(z_coords)
                self.line_plot.set_data(main_axis[valid_indices], z_coords[valid_indices])
            else:
                self.line_plot.set_data([], [])

        # Logic for SOLID mode
        elif self.plot_mode == PlotMode.SOLID:
            if structured_data is None:
                self.ax_2d.text(0.5, 0.5, 'Solid View not available for Global View', ha='center', va='center',
                                transform=self.ax_2d.transAxes)
            else:
                if orientation == 'X':
                    slice_index = np.argmin(np.abs(structured_data[0, :, 0] - self.slicer_pos))
                    interp_cross = structured_data[:, slice_index, :];
                    main_axis, z_coords = interp_cross[:, 1], interp_cross[:, 2]
                else:
                    slice_index = np.argmin(np.abs(structured_data[:, 0, 1] - self.slicer_pos))
                    interp_cross = structured_data[slice_index, :, :];
                    main_axis, z_coords = interp_cross[:, 0], interp_cross[:, 2]
                valid = ~np.isnan(z_coords);
                main_axis, z_coords = main_axis[valid], z_coords[valid]
                if len(z_coords) > 2:
                    floor_z = np.percentile(z_coords, 10)
                    is_tow = (z_coords > floor_z + 0.1).astype(int)
                    tow_changes = np.diff(is_tow)
                    starts = np.where(tow_changes == 1)[0] + 1
                    ends = np.where(tow_changes == -1)[0]
                    if is_tow[0] == 1: starts = np.insert(starts, 0, 0)
                    if is_tow[-1] == 1: ends = np.append(ends, len(is_tow) - 1)
                    tow_colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(starts)))
                    for i, (start, end) in enumerate(zip(starts, ends)):
                        if end > start:
                            fill = self.ax_2d.fill_between(main_axis[start:end + 1], z_coords[start:end + 1], floor_z,
                                                           color=tow_colors[i], alpha=0.7)
                            self.solid_fills.append(fill)

        self.fig_2d.canvas.draw();
        self.fig_2d.canvas.flush_events()

    # --- Key Callback Handlers ---
    def get_current_slicer_orientation(self):
        if self.view_mode_idx < len(self.layer_ids):
            return 'X' if self.layer_orientations[self.layer_ids[self.view_mode_idx]] == 0 else 'Y'
        else:
            return self.global_slicer_orientation

    def cycle_view_mode(self, vis):
        self.view_mode_idx = (self.view_mode_idx + 1) % (len(self.layer_ids) + 1)
        self.set_active_view();
        return False

    def toggle_slicer_orientation(self, vis):
        if self.view_mode_idx == len(self.layer_ids):
            self.global_slicer_orientation = 'Y' if self.global_slicer_orientation == 'X' else 'X'
            print(f"--- Toggled Global Slicer to {self.global_slicer_orientation}-axis ---")
            self.reset_slicer_and_axes_for_new_view()
        return False

    def toggle_plot_mode(self, vis):
        if self.plot_mode == PlotMode.HYBRID:
            self.plot_mode = PlotMode.SOLID
        else:
            self.plot_mode = PlotMode.HYBRID
        print(f"--- Toggled 2D Plot to {self.plot_mode.name} View ---")
        self.update_2d_plot();
        return False

    def move_slicer(self, step):
        orientation = self.get_current_slicer_orientation()
        if orientation == 'X':
            self.slicer_pos = np.clip(self.slicer_pos + step, self.bounds['x_min'], self.bounds['x_max'])
        else:
            self.slicer_pos = np.clip(self.slicer_pos + step, self.bounds['y_min'], self.bounds['y_max'])
        self.update_slicer_geometry();
        self.update_2d_plot()

    def move_slicer_forward(self, vis):
        self.move_slicer(self.slicer_step); return False

    def move_slicer_backward(self, vis):
        self.move_slicer(-self.slicer_step); return False

    def quit(self, vis):
        self.vis_3d.close(); return False


def main():
    try:
        with open('config/layup_plan.json', 'r') as f:
            layup_plan = json.load(f)
        config = configparser.ConfigParser();
        config.read(layup_plan.get('config_file', 'config/config.ini'))
        paths_cfg, llt_calib_cfg = config['FilePaths'], config['LLT_HandEye']
    except Exception as e:
        print(f"FATAL ERROR: Could not load configuration or layup plan. Details: {e}"); return
    session_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_session_dir(
        paths_cfg.get('output_directory', 'data/'))
    if not session_path: print("Error: Could not find any session directory to inspect."); return
    print(f"\nPre-processing data for session: {os.path.basename(session_path)}")
    layer_structured_data, layer_raw_data, layer_colors, layer_orientations = {}, {}, {}, {}
    for layer_info in layup_plan.get('layers', []):
        layer_id = layer_info.get('layer_id');
        layer_dir = os.path.join(session_path, f"Layer_{layer_id:02d}")
        npy_file_path = os.path.join(layer_dir, f"structured_surface_layer_{layer_id:02d}.npy")
        if os.path.exists(npy_file_path):
            print(f"  - Loading and processing Layer {layer_id}...")
            layer_structured_data[layer_id] = np.load(npy_file_path)
            pass_dirs = sorted(glob.glob(os.path.join(layer_dir, "Pass_*")))
            raw_points, profile_indices = load_and_filter_raw_points(pass_dirs, paths_cfg, llt_calib_cfg)
            if raw_points is not None and len(raw_points) > 0:
                colors = compute_intra_profile_colors(raw_points, profile_indices)
                layer_raw_data[layer_id], layer_colors[layer_id] = raw_points, colors
                layer_orientations[layer_id] = layer_info['passes'][0].get('orientation', 0)
            else:
                print(f"  - Warning: No valid raw point data for Layer {layer_id}. It will be skipped.");
                if layer_id in layer_structured_data: del layer_structured_data[layer_id]
        else:
            print(
                f"  - Warning: Structured data file not found for Layer {layer_id}. Run 'create_structured_surface.py' first.")
    if not layer_structured_data: print("\nNo valid layer data found to inspect. Exiting."); return
    visualizer = LinkedSlicingVisualizer(layer_structured_data, layer_raw_data, layer_colors, layer_orientations)
    visualizer.run()


if __name__ == "__main__":
    main()