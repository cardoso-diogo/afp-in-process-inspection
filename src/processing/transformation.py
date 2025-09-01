# src/processing/transformation.py

import numpy as np
from scipy.spatial.transform import Rotation


def create_robot_pose_matrix(x, y, z, rz, ry, rx):
    """Creates the 4x4 T_base_tcp matrix from robot coordinates."""
    r = Rotation.from_euler('zyx', [rz, ry, rx], degrees=True)
    T = np.identity(4)
    T[0:3, 0:3] = r.as_matrix()
    T[0:3, 3] = [x, y, z]
    return T

def create_llt_hand_eye_matrix(llt_calib):  # <-- REMOVED layup_angle_deg
    """
    Creates the 4x4 T_tcp_llt matrix.
    This matrix describes the fixed relationship from
    the TCP frame to the Sensor's measurement frame.
    """
    # Get the fixed physical mounting rotation
    fixed_rot_z = llt_calib.getfloat('fixed_rot_z_deg', 0.0)
    fixed_rot_y = llt_calib.getfloat('fixed_rot_y_deg', 0.0)
    fixed_rot_x = llt_calib.getfloat('fixed_rot_x_deg', 0.0)
    fixed_rotation = Rotation.from_euler('zyx', [fixed_rot_z, fixed_rot_y, fixed_rot_x], degrees=True)

    # Apply the base correction needed to align the sensor's X-scan
    # with the robot's Y-axis for a 0-degree layup. This is a fixed property of the setup.
    base_correction_rotation = Rotation.from_euler('z', 90, degrees=True)

    # The final rotation is the combination of these two fixed effects.
    final_rotation = base_correction_rotation * fixed_rotation

    # Create the final transformation matrix
    transform = np.identity(4)
    transform[0:3, 0:3] = final_rotation.as_matrix()
    transform[0:3, 3] = [
        llt_calib.getfloat('offset_x_tcp_to_llt'),
        llt_calib.getfloat('offset_y_tcp_to_llt'),
        llt_calib.getfloat('offset_z_tcp_to_llt')
    ]
    return transform

def map_llt_point_to_world(scanner_x, scanner_z, T_base_tcp, T_tcp_llt):
    """
    Transforms a point from the LLT frame (x_sensor, z_sensor) to the robot base frame.
    """
    # The point in the sensor's 2D measurement plane is (scanner_x, scanner_z).
    # We represent this as a 3D point in the sensor's frame: (scanner_x, 0, scanner_z).
    p_llt_homogeneous = np.array([scanner_x, 0, scanner_z, 1])

    # Chain the transformations: P_world = T_base_tcp * T_tcp_llt * P_llt
    p_world = T_base_tcp @ T_tcp_llt @ p_llt_homogeneous
    return p_world[:3]

def map_camera_pixel_to_world(pixel_y, T_base_tcp, cam_calib):
    """
    Transforms a camera pixel to the robot base frame using the corrected axis mapping.
    """
    # The camera's vertical centerline corresponds to a constant X in the TCP frame.
    tcp_x = cam_calib.getfloat('x_offset_for_line')

    # The camera's Y-pixel maps to the Y-coordinate in the TCP frame.
    tcp_y = (pixel_y * -cam_calib.getfloat('mm_per_pixel')) + cam_calib.getfloat('y_offset_at_pixel_0')

    # We assume the measurement line lies on the TCP's XY plane.
    tcp_z = 0

    p_tcp_homogeneous = np.array([tcp_x, tcp_y, tcp_z, 1])

    # The second part of the transformation remains the same
    p_world = T_base_tcp @ p_tcp_homogeneous
    return p_world[:3]


def map_llt_points_to_world_vectorized(x_coords, z_coords, T_base_tcp, T_tcp_llt):
    """
    Transforms an array of 2D laser points to 3D world coordinates in a
    vectorized and efficient manner.

    Args:
        x_coords (np.ndarray): 1D array of X coordinates from the laser profile.
        z_coords (np.ndarray): 1D array of Z coordinates from the laser profile.
        T_base_tcp (np.ndarray): 4x4 transformation matrix from robot base to TCP.
        T_tcp_llt (np.ndarray): 4x4 transformation matrix from TCP to LLT sensor.

    Returns:
        np.ndarray: An (N, 3) array of transformed 3D points in the world frame.
    """
    num_points = len(x_coords)
    if num_points == 0:
        return np.array([])

    # Create an (N, 4) array of homogeneous coordinates for all points
    # Points are in the LLT frame, so Y is 0.
    llt_points_homogeneous = np.ones((num_points, 4))
    llt_points_homogeneous[:, 0] = x_coords
    llt_points_homogeneous[:, 1] = 0  # Y is always 0 in the 2D sensor frame
    llt_points_homogeneous[:, 2] = z_coords

    # Calculate the full transformation from base to LLT
    T_base_llt = T_base_tcp @ T_tcp_llt

    # Apply the transformation to all points at once.
    # The result of the matmul is an (N, 4) array. We transpose the points
    # to (4, N) for the matmul and then transpose the result back to (N, 4).
    world_points_homogeneous = (T_base_llt @ llt_points_homogeneous.T).T

    # Return only the first 3 columns (X, Y, Z), discarding the homogeneous 'w'
    return world_points_homogeneous[:, :3]