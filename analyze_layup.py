# analyze_layup.py

import os
import glob
import pandas as pd
import numpy as np
import configparser
import sys
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

pd.set_option('expand_frame_repr', False)


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


def get_orientation_from_plan(layup_plan, layer_id):
    """Gets the orientation for an entire layer from the plan."""
    try:
        for layer in layup_plan.get('layers', []):
            if layer.get('layer_id') == layer_id:
                # Assume all passes in a layer have the same orientation.
                # Return the orientation of the first pass found in the plan.
                return layer['passes'][0].get('orientation', 0.0)
    except (IndexError, KeyError):
        return None
    return None


def add_colored_fill_between(ax, x, y1, y2, color_metric, cmap, norm, orientation='horizontal', **kwargs):
    """
    Adds a PolyCollection to the axes to simulate a colored fill_between.
    It is significantly faster than looping through and calling fill_between for each segment.
    """
    x, y1, y2, color_metric = np.array(x), np.array(y1), np.array(y2), np.array(color_metric)

    verts = []
    for i in range(len(x) - 1):
        # Create polygon vertices for this segment
        if orientation == 'horizontal':
            verts.append([
                (x[i], y1[i]),
                (x[i + 1], y1[i + 1]),
                (x[i + 1], y2[i + 1]),
                (x[i], y2[i])
            ])
        else:  # vertical
            verts.append([
                (y1[i], x[i]),
                (y1[i + 1], x[i + 1]),
                (y2[i + 1], x[i + 1]),
                (y2[i], x[i])
            ])

    # Use the midpoint of the segment's metric to determine the color
    colors_metric_midpoints = (color_metric[:-1] + color_metric[1:]) / 2

    collection = PolyCollection(verts, cmap=cmap, norm=norm, **kwargs)
    collection.set_array(colors_metric_midpoints)

    ax.add_collection(collection)
    return collection


def plot_qc_metric(df, x_col, y_col, title, x_label, y_label, output_path, y_lim=None):
    """
    Generates and saves a line plot for a specific QC metric.
    """
    if y_col not in df.columns or df[y_col].isnull().all() or df.empty:
        # print(f"Skipping plot '{title}' due to missing or all-NaN data in column '{y_col}'.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df[x_col], df[y_col], marker='.', linestyle='-', markersize=4)
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add horizontal lines for mean and standard deviation
    mean_val = df[y_col].mean()
    std_val = df[y_col].std()
    plt.axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
    plt.axhline(mean_val + std_val, color='g', linestyle=':', label=f'Mean ± 1σ: {std_val:.3f}')
    plt.axhline(mean_val - std_val, color='g', linestyle=':')

    if y_lim:
        plt.ylim(y_lim)

    plt.legend()
    plt.tight_layout()
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    print("--- Final Layup Analysis (Orientation-Aware) ---")

    # 1. Load Configuration and Layup Plan
    try:
        with open('config/layup_plan.json', 'r') as f:
            layup_plan = json.load(f)
        config_ini_path = os.path.join(layup_plan.get('config_file', 'config.ini'))
        config = configparser.ConfigParser()
        config.read(config_ini_path)
        if not config.sections(): raise FileNotFoundError(f"Config file not found: {config_ini_path}")
        paths_cfg = config['FilePaths']
    except Exception as e:
        print(f"ERROR: Could not load configuration. Details: {e}")
        return

    output_dir = paths_cfg.get('output_directory', 'data/')
    session_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_session_dir(output_dir)
    if not session_path or not os.path.isdir(session_path):
        print(f"Error: Session directory not found: {session_path}")
        return
    print(f"\nProcessing Layup Session: {os.path.basename(session_path)}")

    # 2. Load all transformed feature files into a hierarchical dictionary
    layup_data = {}
    layer_dirs = sorted(glob.glob(os.path.join(session_path, "Layer_*")))
    if not layer_dirs:
        print("Error: No 'Layer_XX' subdirectories found in this session.")
        return

    for layer_dir in layer_dirs:
        layer_name = os.path.basename(layer_dir)
        layup_data[layer_name] = {}
        pass_dirs = sorted(glob.glob(os.path.join(layer_dir, "Pass_*")))
        for pass_dir in pass_dirs:
            pass_name = os.path.basename(pass_dir)
            transformed_file = get_latest_file_in_pass(pass_dir, paths_cfg.get('transformed_laser_filename'))
            if transformed_file:
                layup_data[layer_name][pass_name] = pd.read_csv(transformed_file)

    if not layup_data:
        print("Error: No transformed feature files could be loaded.")
        return

    # 3. Analyze and Plot
    all_layup_interface_results = []

    # --- Plotting Setup ---
    print("\n--- Generating Combined 2D Reconstruction Plot for All Layers ---")
    fig, ax = plt.subplots(figsize=(15, 8))
    gap_collection = None  # To hold the collection for the colorbar
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#c7c7c7', '#dbdb8d', '#9edae5']
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    pass_color_index = 0
    layer_hatch_index = 0

    for layer_name, layer_pass_data in sorted(layup_data.items()):
        print(f"\n--- Processing {layer_name} ---")
        try:
            layer_id = int(layer_name.replace('Layer_', ''))
            orientation = get_orientation_from_plan(layup_plan, layer_id)
            if orientation is None:
                print(f"  - Warning: Orientation for {layer_name} not found in plan. Skipping.");
                continue
        except ValueError:
            print(f"  - Warning: Could not parse layer ID from '{layer_name}'. Skipping.");
            continue

        if orientation == 0:
            merge_axis, cross_axis = 'X', 'Y'
        elif orientation == 90:
            merge_axis, cross_axis = 'Y', 'X'
        else:
            print(f"  - Warning: Orientation {orientation} for {layer_name} not supported. Skipping.");
            continue
        print(f"  - Orientation: {orientation} deg. Merging along World {merge_axis}-axis.")

        layer_min_merge = float('inf')
        layer_max_merge = float('-inf')
        merge_col_ref = f'current_tow_right_edge_world_{merge_axis}'

        for df in layer_pass_data.values():
            if merge_col_ref in df.columns and not df[merge_col_ref].empty:
                layer_min_merge = min(layer_min_merge, df[merge_col_ref].min())
                layer_max_merge = max(layer_max_merge, df[merge_col_ref].max())

        if layer_min_merge == float('inf') or layer_max_merge == float('-inf'):
            print(f"  - Warning: Could not determine data range for {layer_name}. Skipping filtering.")
            filtered_layer_pass_data = layer_pass_data # Use original data if range fails
        else:
            total_range = layer_max_merge - layer_min_merge
            relv_amount = total_range * 0.12
            filter_min = layer_min_merge + relv_amount
            filter_max = layer_max_merge - relv_amount

            filtered_layer_pass_data = {}
            for pass_name, df in layer_pass_data.items():
                if merge_col_ref in df.columns:
                    filtered_df = df[(df[merge_col_ref] >= filter_min) & (df[merge_col_ref] <= filter_max)].copy()
                    if not filtered_df.empty:
                        filtered_layer_pass_data[pass_name] = filtered_df
                else:
                     filtered_layer_pass_data[pass_name] = pd.DataFrame()

        for df in filtered_layer_pass_data.values():
            if all(c in df.columns for c in
                   ['current_tow_left_edge_world_X', 'current_tow_right_edge_world_Y', 'current_tow_right_edge_world_Z',
                    'current_tow_right_edge_world_X', 'current_tow_right_edge_world_Y',
                    'current_tow_right_edge_world_Z']):
                df['tow_width_mm'] = np.linalg.norm(df[['current_tow_left_edge_world_X',
                                                        'current_tow_left_edge_world_Y',
                                                        'current_tow_left_edge_world_Z']].values - df[
                                                        ['current_tow_right_edge_world_X',
                                                         'current_tow_right_edge_world_Y',
                                                         'current_tow_right_edge_world_Z']].values, axis=1)

                # --- Generate and save detailed QC plot for tow width ---
                plot_dir = os.path.join(session_path, "qc_plots")
                # Find the correct pass_name for the current df
                pass_name = [key for key, value in filtered_layer_pass_data.items() if value is df][0]
                tow_x_axis = f'current_tow_right_edge_world_{merge_axis}'

                plot_qc_metric(
                    df=df,
                    x_col=tow_x_axis,
                    y_col='tow_width_mm',
                    title=f'Tow Width Analysis: {layer_name}_{pass_name}',
                    x_label=f'Distance along World {merge_axis}-axis (mm)',
                    y_label='Tow Width (mm)',
                    output_path=os.path.join(plot_dir, f'{layer_name}_{pass_name}_tow_width.png'),
                    y_lim=(10, 15)
                )

        layer_interface_results = []
        sorted_pass_names = sorted(filtered_layer_pass_data.keys())
        if len(sorted_pass_names) >= 2:
            for i in range(len(sorted_pass_names) - 1):
                pass_n_name, pass_n1_name = sorted_pass_names[i], sorted_pass_names[i + 1]
                print(f"  - Comparing: {pass_n_name} -> {pass_n1_name}")
                df_n, df_n1 = filtered_layer_pass_data[pass_n_name], filtered_layer_pass_data[pass_n1_name]

                # Check if dataframes are empty after filtering
                if df_n.empty or df_n1.empty:
                    print(f"    - Skipping comparison due to empty data after filtering.")
                    continue

                prev_tow_df = df_n[
                    [f'current_tow_left_edge_world_{merge_axis}', f'current_tow_left_edge_world_{cross_axis}',
                     'current_tow_left_edge_world_Z']].copy()
                prev_tow_df.columns = ['merge_coord', 'left_cross_prev', 'left_z_prev']
                next_tow_df = df_n1[
                    [f'current_tow_right_edge_world_{merge_axis}', f'current_tow_right_edge_world_{cross_axis}',
                     'current_tow_right_edge_world_Z']].copy()
                next_tow_df.columns = ['merge_coord', 'right_cross_next', 'right_z_next']

                interface_df = pd.merge_asof(left=next_tow_df.sort_values('merge_coord').dropna(),
                                             right=prev_tow_df.sort_values('merge_coord').dropna(), on='merge_coord',
                                             direction='nearest', tolerance=1.0).dropna()
                if interface_df.empty: print("    - No matching points found."); continue

                interface_df['gap_width_mm'] = abs(interface_df['right_cross_next'] - interface_df['left_cross_prev'])

                gap_cmap = matplotlib.colormaps.get_cmap('bwr')
                gap_norm = Normalize(vmin=1.5, vmax=2.5)

                gap_collection = add_colored_fill_between(
                    ax,
                    x=interface_df['merge_coord'],
                    y1=interface_df['left_cross_prev'],
                    y2=interface_df['right_cross_next'],
                    color_metric=interface_df['gap_width_mm'],
                    cmap=gap_cmap,
                    norm=gap_norm,
                    orientation='horizontal' if orientation == 0 else 'vertical',
                    alpha=0.8
                )
                interface_df['height_step_mm'] = interface_df['right_z_next'] - interface_df['left_z_prev']

                as_laid_edge_cols = [f'current_tow_left_edge_world_{ax}' for ax in ['X', 'Y', 'Z']]
                post_compaction_edge_cols = [f'gap_edge_world_{ax}' for ax in ['X', 'Y', 'Z']]

                if all(c in df_n.columns for c in as_laid_edge_cols) and all(
                        c in df_n1.columns for c in post_compaction_edge_cols):
                    as_laid_df = df_n[as_laid_edge_cols].copy()
                    as_laid_df['merge_coord'] = df_n[f'current_tow_left_edge_world_{merge_axis}']

                    post_compaction_df = df_n1[post_compaction_edge_cols].copy()
                    post_compaction_df['merge_coord'] = df_n1[f'gap_edge_world_{merge_axis}']

                    stability_df = pd.merge_asof(
                        left=post_compaction_df.sort_values('merge_coord'),
                        right=as_laid_df.sort_values('merge_coord'),
                        on='merge_coord',
                        direction='nearest',
                        tolerance=1.0
                    ).dropna()

                    if not stability_df.empty:
                        p_as_laid = stability_df[as_laid_edge_cols].values
                        p_post_compaction = stability_df[post_compaction_edge_cols].values
                        stability_df['edge_shift_mm'] = np.linalg.norm(p_post_compaction - p_as_laid, axis=1)
                        interface_df = pd.merge(interface_df, stability_df[['merge_coord', 'edge_shift_mm']],
                                                on='merge_coord', how='left')
                else:
                    print(f"    - Warning: Required columns for edge stability not found. Skipping calculation.")
                    interface_df['edge_shift_mm'] = np.nan

                interface_df['interface_name'] = f"{layer_name}_{pass_n_name}_to_{pass_n1_name}"
                layer_interface_results.append(interface_df)

                plot_dir = os.path.join(session_path, "qc_plots")
                interface_plot_name = f"{layer_name}_{pass_n_name}_to_{pass_n1_name}"

                plot_qc_metric(
                    df=interface_df,
                    x_col='merge_coord',
                    y_col='gap_width_mm',
                    title=f'Gap Width Analysis: {pass_n_name} to {pass_n1_name}',
                    x_label=f'Distance along World {merge_axis}-axis (mm)',
                    y_label='Gap Width (mm)',
                    output_path=os.path.join(plot_dir, f'{interface_plot_name}_gap_width.png'),
                    y_lim=(-0.5, 2.0)
                )

                plot_qc_metric(
                    df=interface_df,
                    x_col='merge_coord',
                    y_col='height_step_mm',
                    title=f'Height Step Analysis: {pass_n_name} to {pass_n1_name}',
                    x_label=f'Distance along World {merge_axis}-axis (mm)',
                    y_label='Height Step (mm)',
                    output_path=os.path.join(plot_dir, f'{interface_plot_name}_height_step.png'),
                    y_lim=(-0.5, 0.5)
                )

                plot_qc_metric(
                    df=interface_df,
                    x_col='merge_coord',
                    y_col='edge_shift_mm',
                    title=f'Edge Stability Analysis: {pass_n_name} to {pass_n1_name}',
                    x_label=f'Distance along World {merge_axis}-axis (mm)',
                    y_label='Edge Shift (mm)',
                    output_path=os.path.join(plot_dir, f'{interface_plot_name}_edge_shift.png'),
                    y_lim=(-0.2, 1.0)
                )

        print(f"  - Adding {layer_name} tow bodies to the combined plot...")
        hatch = hatches[layer_hatch_index % len(hatches)];
        layer_hatch_index += 1
        for pass_name, df in sorted(filtered_layer_pass_data.items()):
            color = colors[pass_color_index % len(colors)];
            pass_color_index += 1
            if all(col in df.columns for col in
                   ['current_tow_right_edge_world_X', 'current_tow_right_edge_world_Y', 'current_tow_left_edge_world_Y',
                    'current_tow_left_edge_world_X']):
                label = f"{layer_name} {pass_name}"
                if orientation == 0:
                    ax.fill_between(df['current_tow_right_edge_world_X'], df['current_tow_right_edge_world_Y'],
                                    df['current_tow_left_edge_world_Y'], color=color, alpha=0.7, label=label,
                                    hatch=hatch, edgecolor='w')
                elif orientation == 90:
                    ax.fill_betweenx(df['current_tow_right_edge_world_Y'], df['current_tow_right_edge_world_X'],
                                     df['current_tow_left_edge_world_X'], color=color, alpha=0.7, label=label,
                                     hatch=hatch, edgecolor='w')
        all_layup_interface_results.extend(layer_interface_results)

    # --- Finalize and Save Plot ---
    #ax.set_title('Top-Down Reconstruction Analysis', fontsize=16)
    ax.set_xlabel('Robot World X-axis (mm)');
    ax.set_ylabel('Robot World Y-axis (mm)')
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax.grid(True);
    ax.set_aspect('equal', 'box');

    # 4. Save Final Report and Print Aggregate Statistics
    if all_layup_interface_results:
        final_report_df = pd.concat(all_layup_interface_results, ignore_index=True)
        report_base_name = paths_cfg.get('comparison_filename')
        session_name = os.path.basename(session_path)
        report_filename = os.path.join(session_path, f"{report_base_name}_{session_name}.csv")
        print(f"\nSaving final analysis report for all layers to {os.path.basename(report_filename)}")
        final_report_df.to_csv(report_filename, index=False, float_format='%.4f')

        print("\n--- Final QC Statistics (All Layers) ---")
        full_layup_pass_data = {}
        for layer_name, pass_data in layup_data.items():
            for pass_name, df in pass_data.items():
                full_layup_pass_data[f"{layer_name}_{pass_name}"] = df

        stats_string_for_plot = "--- QC Summary ---\n"

        stats_string_for_plot += "\nTow Widths (mean ± std dev):\n"
        for name, df in sorted(full_layup_pass_data.items()):
            if 'tow_width_mm' in df.columns and not df['tow_width_mm'].empty:
                clean_name = name.replace('_', ' ').replace('Pass ', 'P')
                stats_string_for_plot += f"  {clean_name}: {df['tow_width_mm'].mean():.3f} ± {df['tow_width_mm'].std():.3f} mm\n"

        stats_string_for_plot += "\nInterfaces (mean ± std dev):\n"
        for name, group in final_report_df.groupby('interface_name'):
            clean_name = name.replace('_to_', ' -> ').replace('Pass_', 'P')
            gap_mean, gap_std = group['gap_width_mm'].mean(), group['gap_width_mm'].std()
            step_mean, step_std = group['height_step_mm'].mean(), group['height_step_mm'].std()
            stats_string_for_plot += f"  {clean_name}:\n"
            stats_string_for_plot += f"    Gap: {gap_mean:.3f} ± {gap_std:.3f} mm\n"
            stats_string_for_plot += f"    Step: {step_mean:.3f} ± {step_std:.3f} mm\n"
            if 'edge_shift_mm' in group.columns and not group['edge_shift_mm'].isnull().all():
                shift_mean, shift_std = group['edge_shift_mm'].mean(), group['edge_shift_mm'].std()
                stats_string_for_plot += f"    Edge Shift: {shift_mean:.3f} ± {shift_std:.3f} mm\n"

        # Add the text box to the plot using the generated string
        at = AnchoredText(stats_string_for_plot.strip(),
                          loc='upper right', frameon=True,
                          bbox_to_anchor=(1.5, 1.0),
                          bbox_transform=ax.transAxes,
                          prop=dict(size=10, family='monospace'))
        at.patch.set(boxstyle="round,pad=0.5,rounding_size=0.2", facecolor='#f0f0f0', alpha=0.85)
        ax.add_artist(at)

        if gap_collection:
            cbar = fig.colorbar(gap_collection, ax=ax, orientation='vertical', pad=0.02, shrink=0.5)
            cbar.set_label('Gap / Overlap Width (mm)', fontsize=10)

        plt.tight_layout(rect=[0.05, 0, 0.9, 1])

        plot_filename = os.path.join(session_path, "layup_reconstruction_all_layers.png")
        print(f"\nSaving combined plot to {os.path.basename(plot_filename)}")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show(block=True)
        plt.close(fig)

        print("\nTow Widths:")
        for name, df in sorted(full_layup_pass_data.items()):
            if 'tow_width_mm' in df.columns and not df['tow_width_mm'].empty:
                print(
                    f"  {name}: Mean Width: {df['tow_width_mm'].mean():.4f} mm (Std Dev: {df['tow_width_mm'].std():.4f} mm)")

        print("\nInterfaces:")
        for name, group in final_report_df.groupby('interface_name'):
            print(f"  {name}:")
            print(
                f"    - Mean Gap Width:    {group['gap_width_mm'].mean():.4f} mm (Std Dev: {group['gap_width_mm'].std():.4f} mm)")
            print(
                f"    - Mean Height Step:  {group['height_step_mm'].mean():.4f} mm (Std Dev: {group['height_step_mm'].std():.4f} mm)")
            if 'edge_shift_mm' in group.columns and not group['edge_shift_mm'].isnull().all():
                print(
                    f"    - Mean Edge Shift:   {group['edge_shift_mm'].mean():.4f} mm (Std Dev: {group['edge_shift_mm'].std():.4f} mm)")
        print("--------------------------")
    else:
        plt.tight_layout()
        plot_filename = os.path.join(session_path, "layup_reconstruction_all_layers.png")
        print(f"\nSaving combined plot to {os.path.basename(plot_filename)}")
        plt.savefig(plot_filename, dpi=300)
        plt.show(block=True)
        plt.close(fig)

    print("\nFull layup analysis complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {e}");
        import traceback;

        traceback.print_exc()