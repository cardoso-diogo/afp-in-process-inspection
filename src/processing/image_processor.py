# src/processing/image_processor.py

import cv2
import numpy as np
import yaml


def load_calibration(calibration_file_path):
    """Loads camera intrinsic parameters from an OpenCV-style .yml file."""
    try:
        fs = cv2.FileStorage(calibration_file_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise IOError(f"Failed to open: {calibration_file_path}")
        cam_matrix = fs.getNode('camera_matrix').mat()
        dist_coeffs = fs.getNode('dist_coeffs').mat()
        fs.release()
        if cam_matrix is None or dist_coeffs is None:
            raise ValueError("camera_matrix or dist_coeffs not found.")
        return cam_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return None, None


def undistort_image(image, camera_matrix, dist_coeffs):
    """Applies lens undistortion to an image."""
    if camera_matrix is None or dist_coeffs is None:
        return image
    h, w = image.shape[:2]
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_cam_matrix)
    x, y, w, h = roi
    return undistorted_img[y:y + h, x:x + w]


def find_tows(image, config):
    """
    Finds all valid tow contours in an image using a simple global binary threshold.
    Applies the constraint that there can be at most 2 tows.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    _, binary_img = cv2.threshold(
        blurred_img,
        config['binary_threshold'],
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((5, 5), np.uint8)
    opened_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    validated_tows = []
    for cnt in contours:
        if cv2.contourArea(cnt) < config['min_contour_area']: continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < config['min_aspect_ratio']: continue
        validated_tows.append({'contour': cnt, 'x': x, 'y': y, 'w_px': w, 'h_px': h, 'cy': y + h // 2})

    if len(validated_tows) > 2:
        return []

    return sorted(validated_tows, key=lambda t: t['cy'])


def process_last_tow(detected_tows, mm_per_pixel):
    """
    Takes a list of tows, processes ONLY the last one, and returns its measurements.
    """
    if not detected_tows:
        return None

    last_tow = detected_tows[-1]

    return {
        'cam_tow_width_mm': last_tow['w_px'] * mm_per_pixel,
        'cam_tow_top_edge_y_px': last_tow['y'],
        'cam_tow_bottom_edge_y_px': last_tow['y'] + last_tow['h_px']
    }


def visualize_image_processing(image, tows):
    """Draws detected contours and highlights the last one being measured."""
    vis_img = image.copy()

    if not tows:
        cv2.putText(vis_img, "No Tows Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Draw all detected tows in blue for context
        for i, tow in enumerate(tows):
            x, y, w, h = tow['x'], tow['y'], tow['w_px'], tow['h_px']
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
            # Label all tows for clarity
            cv2.putText(vis_img, f"Tow {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        last_tow = tows[-1]
        x, y, w, h = last_tow['x'], last_tow['y'], last_tow['w_px'], last_tow['h_px']
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(vis_img, "Last Tow (Measured)", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Image Processing Visualization", vis_img)