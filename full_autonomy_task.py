#!/usr/bin/env python3

# ----------------------------------------------- AI GENERATED SCRIPT -------------------------------------------------

import pyzed.sl as sl
import numpy as np
import cv2
import math
import time
from loco_lib import *
import gps_lib # Assuming this is available
from sklearn.cluster import DBSCAN

BEFORE_HILL_COORD = (0,0,0)
ICY_CRATER_COORD = (0,0,0)
LAVA_TUBE_COORD = (0,0,0)

TARGET_MARKER_1_ID = 10
TARGET_MARKER_2_ID = 11

def light_switch(on=True):
    if on:
        #turn the light on
        pass
    else:
        #turn it off
        pass
    ret = False
    return ret

def move_to_coordinates(lat, long, alt):
    ret = False
    return ret

def find_peak(point_cloud_mat, max_distance=10.0, min_altitude=0.3, eps=0.5, min_cluster_size=20):
    """
    Finds the highest point (peak) of distinct objects in the environment.

    Args:
        point_cloud_mat (sl.Mat): The ZED point cloud data (XYZ format).
        max_distance (float): The maximum distance from the rover to search for peaks (in meters).
        min_altitude (float): The minimum height a point must have to be considered,
                              used to filter out the ground (in meters).
        eps (float): The maximum distance between two points for them to be considered
                     in the same neighborhood (DBSCAN parameter). Controls how "spread out" an object can be.
        min_cluster_size (int): The number of points an object needs to have to be
                                considered a valid cluster (DBSCAN parameter).

    Returns:
        list: A list of dictionaries, where each dictionary represents a peak point.
              Example: [{'x': 1.5, 'y': 3.2, 'altitude': 1.2, 'distance_from_rover': 3.5}]
              Returns an empty list if no peaks are found.
    """
    # 1. Get Point Cloud Data into a NumPy array
    # Note: Assumes ZED is configured with COORIDNATE_SYSTEM.RIGHT_HANDED_Z_UP, so Z is altitude.
    pc_data_full = point_cloud_mat.get_data()
    
    # Reshape to a list of (X, Y, Z, Color) points and keep only XYZ
    points = pc_data_full.reshape(-1, 4)[:, :3]
    
    # 2. Filter Points
    # Remove non-finite values (NaN, Inf)
    valid_points_mask = np.isfinite(points).all(axis=1)
    points = points[valid_points_mask]

    # Calculate distance from rover (origin) for all points
    distances = np.linalg.norm(points, axis=1)

    # Apply distance and altitude filters
    # The altitude is the Z-coordinate
    mask = (distances < max_distance) & (points[:, 2] > min_altitude)
    candidate_points = points[mask]

    if candidate_points.shape[0] < min_cluster_size:
        # Not enough points to even form one cluster
        return []

    # 3. Cluster the points to identify distinct objects
    # DBSCAN is great because we don't need to know the number of objects beforehand.
    db = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(candidate_points)
    labels = db.labels_
    
    # The number of clusters found (ignoring noise points, which are labeled -1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if not unique_labels:
        return []

    # 4. Find the highest point (peak) for each cluster
    peak_points = []
    for label in unique_labels:
        # Get all points belonging to this cluster
        cluster_points = candidate_points[labels == label]
        
        # Find the index of the point with the highest altitude (max Z value)
        peak_index_in_cluster = np.argmax(cluster_points[:, 2])
        
        # Get the peak point's coordinates
        peak_point_coords = cluster_points[peak_index_in_cluster]
        
        # Calculate its distance from the rover
        peak_distance = np.linalg.norm(peak_point_coords)

        # Format the result and add to our list
        peak_info = {
            'x': round(peak_point_coords[0], 2),
            'y': round(peak_point_coords[1], 2),
            'altitude': round(peak_point_coords[2], 2),
            'distance_from_rover': round(peak_distance, 2)
        }
        peak_points.append(peak_info)
        
    return peak_points

def install_antenna(): #just drops the antenna
    ret = False
    return ret

def record_or_send_coordinates(send=False):
    ret = False
    return ret

# def find_coldest_point():
#     pass
def avoid_obstacles(point_cloud_mat, res, m2p):
    """
    Analyzes a ZED point cloud to find obstacles and calculates a safe path.

    Args:
        point_cloud_mat (sl.Mat): The ZED point cloud data (XYZ format).
        res (sl.Resolution): The resolution of the point cloud.
        m2p (int): The meters-to-pixels conversion factor for the map.

    Returns:
        tuple: (obstacle_in_front, best_angle, obstacle_density_in_best_path)
               - obstacle_in_front (bool): True if an obstacle is directly ahead.
               - best_angle (float): The recommended steering angle (90 is straight).
               - obstacle_density_in_best_path (float): Mean value of the clearest path.
    """
    # --- 1. Create Bird's Eye View Map from Point Cloud ---
    bult2 = point_cloud_mat.get_data().reshape(res.width * res.height, 4)[:, :3]
    bult2 = np.column_stack((bult2[:, 0], bult2[:, 2], -bult2[:, 1]))

    z_filter = -0.35
    height_mask = (bult2[:, 1] > z_filter) & (bult2[:, 1] < 3.5)
    distance_mask = (abs(bult2[:, 2]) < 4) & (abs(bult2[:, 2]) > 0.5)
    bult2 = bult2[height_mask & distance_mask]

    bult2 = np.column_stack((bult2[:, 0], bult2[:, 2]))
    bult2 = (bult2 * m2p * 5).astype(int)
    bult2 += np.array([10 * m2p, 14 * m2p])

    map_height, map_width = 20 * m2p, 15 * m2p
    twod_map = np.zeros((map_height, map_width), dtype=np.uint8)
    valid_points = (bult2[:, 0] >= 0) & (bult2[:, 0] < map_height) & \
                   (bult2[:, 1] >= 0) & (bult2[:, 1] < map_width)
    bult2 = bult2[valid_points]
    twod_map[bult2[:, 0], bult2[:, 1]] = 255

    # --- 2. Process the Map for Clarity ---
    rows, cols = twod_map.shape
    gradient = np.linspace(1, 0, rows).reshape(-1, 1)
    twod_map_processed = (twod_map.astype(np.float32) * np.repeat(gradient, cols, axis=1)).astype(np.uint8)
    
    twod_map_processed = cv2.medianBlur(twod_map_processed, 3)
    _, twod_map_processed = cv2.threshold(twod_map_processed, 130, 255, cv2.THRESH_BINARY)
    twod_map_processed = cv2.morphologyEx(twod_map_processed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    twod_map_processed = cv2.flip(cv2.rotate(twod_map_processed, cv2.ROTATE_90_CLOCKWISE), 1)

    # --- 3. Scan for Obstacles and Find Best Path ---
    minimum_mean = 2
    height, width = twod_map_processed.shape
    radius = height
    p1 = (width // 2, height)  # Origin point (rover)

    # First, check directly in front
    front_check_contour = np.array([p1, (p1[0] - 30, 0), (p1[0] + 30, 0)])
    front_mean, _ = getContourStat(cv2.convexHull(front_check_contour), twod_map_processed)
    obstacle_in_front = front_mean[0][0] > minimum_mean

    # Scan a 100-degree arc for the clearest path
    angle_differince = 20
    angle_slide = 5
    interval_means = []
    for angle_start in range(140, 40, -angle_slide):
        angle_end = angle_start - angle_differince
        
        p2_x = int(math.cos(math.radians(angle_start)) * radius + width / 2)
        p2_y = int(height - math.sin(math.radians(angle_start)) * radius)
        p3_x = int(math.cos(math.radians(angle_end)) * radius + width / 2)
        p3_y = int(height - math.sin(math.radians(angle_end)) * radius)

        hull_points = np.array([p1, (p2_x, p2_y), (p3_x, p3_y)])
        convex = cv2.convexHull(hull_points)
        mean_val, _ = getContourStat(convex, twod_map_processed)
        interval_means.append((mean_val[0][0], (angle_start + angle_end) / 2))

    # Find the path with the lowest obstacle density (clearest path)
    interval_means.sort(key=lambda x: x[0])
    best_path_density, best_angle = interval_means[0]

    # For visualization, create a colored map
    twod_map_colored = cv2.applyColorMap(twod_map, cv2.COLORMAP_JET)
    twod_map_colored = cv2.flip(cv2.rotate(twod_map_colored, cv2.ROTATE_90_CLOCKWISE), 1)
    
    return obstacle_in_front, best_angle, best_path_density, twod_map_processed, twod_map_colored

def getContourStat(contour,image):
    mask = np.zeros(image.shape,dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    cx, cy = image.shape[1]//2, image.shape[0]-1
    
    y, x = np.where(mask == 255)
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Normalize distances to 0-1 range
    max_dist = np.max(distances)
    if max_dist > 0:
        weights = 1 - (distances / max_dist)  # Points closer to center have higher weights
    else:
        weights = np.ones_like(distances)
    
    # Create weighted mask
    weighted_mask = np.zeros_like(mask, dtype=np.float32)
    weighted_mask[y, x] = weights * 255  # Scale weights to 0-255 range
    
    # Convert to uint8 for OpenCV
    weighted_mask = weighted_mask.astype(np.uint8)
    
    mean,stddev = cv2.meanStdDev(image,mask=weighted_mask)
    return mean, stddev

try:
    #from getGps import get_gps
    GPS_AVAILABLE = True
except ImportError:
    print("GPS library not found. GPS features will be disabled.")
    GPS_AVAILABLE = False

def go_between_arucos(zed, detector):
    marker_timeout = 5.0  # seconds
    arrival_threshold = 0.5 # meters
    
    # --- State Variables ---
    last_seen_markers = {}
    runtime_params = sl.RuntimeParameters(confidence_threshold=50)
    image = sl.Mat()
    point_cloud = sl.Mat()

    # Low-res point cloud for obstacle avoidance
    res = sl.Resolution(720, 404)
    point_cloud2 = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    m2p = 25

    print(f"Mission Started: Go between ArUco markers {TARGET_MARKER_1_ID} and {TARGET_MARKER_2_ID}.")

    while True:
        # --- 1. Grab Data from ZED Camera ---
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            break
        
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        zed.retrieve_measure(point_cloud2, sl.MEASURE.XYZ, sl.MEM.CPU, res)

        frame = image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # ZED images are BGRA
        display_frame = frame.copy()
        current_time = time.time()
        
        # --- 2. Detect ArUco Markers ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None:
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                if marker_id in [TARGET_MARKER_1_ID, TARGET_MARKER_2_ID]:
                    center = np.mean(corner[0], axis=0).astype(int)
                    err, pc_value = point_cloud.get_value(center[0], center[1])
                    if err == sl.ERROR_CODE.SUCCESS and np.isfinite(pc_value[2]):
                        last_seen_markers[marker_id] = {
                            'position': pc_value[:3],
                            'timestamp': current_time
                        }
                        # Draw on frame for feedback
                        cv2.aruco.drawDetectedMarkers(display_frame, [corner], np.array([[marker_id]]))

        # --- 3. Call Obstacle Avoidance Function ---
        obstacle_in_front, escape_angle, _, obs_map, obs_map_color = avoid_obstacles(point_cloud2, res, m2p)
        
        # --- 4. High-Level Decision Making ---
        knows_marker_1 = TARGET_MARKER_1_ID in last_seen_markers and \
                         current_time - last_seen_markers[TARGET_MARKER_1_ID]['timestamp'] < marker_timeout
        knows_marker_2 = TARGET_MARKER_2_ID in last_seen_markers and \
                         current_time - last_seen_markers[TARGET_MARKER_2_ID]['timestamp'] < marker_timeout

        # --- BEHAVIOR 1: Navigate to Midpoint ---
        if knows_marker_1 and knows_marker_2:
            pos1 = np.array(last_seen_markers[TARGET_MARKER_1_ID]['position'])
            pos2 = np.array(last_seen_markers[TARGET_MARKER_2_ID]['position'])
            midpoint_3d = (pos1 + pos2) / 2
            distance_to_midpoint = np.linalg.norm(midpoint_3d)

            # Check for arrival
            if distance_to_midpoint < arrival_threshold:
                print(f"SUCCESS: Arrived at midpoint. Distance: {distance_to_midpoint:.2f}m")
                #velocity_control_loco(0.0, 0.0) # STOP
                return True # Mission Complete

            # Check for obstacles before moving
            if obstacle_in_front:
                print(f"Path to midpoint blocked. Executing avoidance. Best angle: {escape_angle:.1f}")
                # Use escape_angle to control rover motors
                # e.g., turn_speed = (90 - escape_angle) * -0.01
                # velocity_control_loco(turn_speed, 0.1) # Turn and move slowly
            else:
                # Path is clear, navigate to midpoint
                angle_to_midpoint_rad = math.atan2(midpoint_3d[0], midpoint_3d[2])
                angle_to_midpoint_deg = math.degrees(angle_to_midpoint_rad)
                print(f"Navigating to midpoint. Dist: {distance_to_midpoint:.2f}m, Angle: {angle_to_midpoint_deg:.1f}")
                # Use angle_to_midpoint_deg to control rover
                # e.g., turn_speed = angle_to_midpoint_deg * 0.01
                # velocity_control_loco(turn_speed, 0.4) # Move towards target
        
        # --- BEHAVIOR 2: Search for Second Marker ---
        elif knows_marker_1 or knows_marker_2:
            print("Found one marker, searching for the other. Turning...")
            #velocity_control_loco(-0.2, 0.0) # Turn left slowly
            
        # --- BEHAVIOR 3: Explore / Default Obstacle Avoidance ---
        else:
            if obstacle_in_front:
                print(f"No targets in sight. Avoiding obstacle. Best angle: {escape_angle:.1f}")
                # Use escape_angle to control rover
                # e.g., turn_speed = (90 - escape_angle) * -0.01
                # velocity_control_loco(turn_speed, 0.1)
            else:
                print("No targets or obstacles. Exploring...")
                #velocity_control_loco(0.0, 0.2) # Move forward slowly

        # --- 5. Visualization ---
        cv2.imshow("Mission View", cv2.resize(display_frame, (800, 450)))
        cv2.imshow("Obstacle Map (Binary)", obs_map)
        cv2.imshow("Obstacle Map (Color)", obs_map_color)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Mission aborted by user.")
            return False
    ret = False
    return ret

def explore_tube():
    ret = False
    return ret

def find_covered_length(point_cloud_mat, rover_height=0.5, ceiling_height=2.5, forward_check_distance=5.0, side_width=2.0):
    """
    Analyzes the point cloud to detect if the rover is in a covered section like a tube.

    Args:
        point_cloud_mat (sl.Mat): The ZED point cloud data (XYZ format).
        rover_height (float): The height of the rover itself. We search for roofs above this height.
        ceiling_height (float): The maximum height to check for a roof.
        forward_check_distance (float): How far in front of the rover to check for a roof.
        side_width (float): How wide the search area should be (total width, centered on rover).

    Returns:
        dict: A dictionary containing information about the roof.
              Example:
              {
                  'is_covered': True,
                  'point_count': 1253,
                  'avg_roof_height': 2.1,
                  'horizontal_offset': -0.15,
                  'coverage_percentage': 0.85
              }
              'is_covered' is the key boolean.
              'horizontal_offset' tells you how centered you are under the roof.
              'coverage_percentage' is an estimate of how much of the "sky box" is filled.
    """
    # 1. Get Point Cloud Data into a NumPy array
    # Assumes ZED is configured with Z-UP, Y-FORWARD coordinate system.
    # Z = altitude, Y = forward distance, X = horizontal distance
    pc_data_full = point_cloud_mat.get_data()
    points = pc_data_full.reshape(-1, 4)[:, :3]
    
    # Remove non-finite values (NaN, Inf)
    valid_points_mask = np.isfinite(points).all(axis=1)
    points = points[valid_points_mask]

    # 2. Define the "Sky Box" volume above the rover
    # This is a 3D box where we expect to find a roof.
    x_min = -side_width / 2.0
    x_max = side_width / 2.0
    y_min = 0.0  # Start check from right in front of the rover
    y_max = forward_check_distance
    z_min = rover_height
    z_max = ceiling_height

    # 3. Filter points to find only those inside the "Sky Box"
    roof_points_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                       (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & \
                       (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                       
    roof_points = points[roof_points_mask]
    
    num_roof_points = roof_points.shape[0]

    # 4. Analyze the detected roof points
    
    # Define a threshold for what constitutes "covered". This needs tuning based on
    # camera resolution and distance. A good start is to estimate the number of points
    # a solid surface would generate.
    # Let's use a density threshold instead of a raw count.
    sky_box_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    
    # This is a very rough density metric. For a 1MP point cloud, a density of 
    # 500 points per cubic meter for a nearby surface is reasonable.
    density_threshold = 500 # points per m^3
    point_count_threshold = int(density_threshold * sky_box_volume)

    is_covered = num_roof_points > point_count_threshold
    
    # 5. Calculate bonus metrics if covered
    avg_roof_height = 0.0
    horizontal_offset = 0.0
    coverage_percentage = 0.0

    if is_covered:
        # Average height of the detected roof
        avg_roof_height = np.mean(roof_points[:, 2])
        
        # How centered is the rover under the roof?
        # A negative value means the roof is more to the left.
        # A positive value means the roof is more to the right.
        horizontal_offset = np.mean(roof_points[:, 0])

        # A rough estimate of how "complete" the roof is.
        # This is a simplified metric. A more advanced method would project the points
        # onto the XY plane and calculate the area of the 2D convex hull.
        # For now, let's just use the ratio of points to a "full" roof.
        # A "full" roof might have 10x the threshold points.
        coverage_percentage = min(1.0, num_roof_points / (point_count_threshold * 10.0))

    # 6. Return the results in a structured dictionary
    result = {
        'is_covered': is_covered,
        'point_count': num_roof_points,
        'point_count_threshold': point_count_threshold,
        'avg_roof_height': round(avg_roof_height, 2),
        'horizontal_offset': round(horizontal_offset, 2),
        'coverage_percentage': round(coverage_percentage, 2)
    }
    
    return result

def enter_airlock():
    ret = False
    return ret

def complete_task(zed, detector):

    #mission flow:

    light_switch()

    AIRLOCK_COORD = record_or_send_coordinates()

    move_to_coordinates(BEFORE_HILL_COORD)

    PEAK_COORD = find_peak()

    move_to_coordinates(PEAK_COORD)

    install_antenna()

    record_or_send_coordinates(True)

    move_to_coordinates(ICY_CRATER_COORD)

    #find_coldest_point()
    #record_or_send_coordinates(True)

    move_to_coordinates(LAVA_TUBE_COORD)

    go_between_arucos(zed, detector)

    explore_tube()

    find_covered_length()

    #after return message
    move_to_coordinates(AIRLOCK_COORD)

    light_switch(False)

    #task end

# ===================================================================
# MAIN EXECUTION BLOCK
# ===================================================================
if __name__=="__main__":

    print("Initializing ZED camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD1080,
        camera_fps=30,
        depth_mode=sl.DEPTH_MODE.ULTRA,
        coordinate_units=sl.UNIT.METER,
        depth_minimum_distance=0.3,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    )
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera open failed: {status}")
        exit()

    print("Initializing ArUco detector...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("Initializing rover locomotion...")
    #start_bus()

    complete_task(zed, detector)    