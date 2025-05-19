#!/usr/bin/env python3
import pyzed.sl as sl
import numpy as np
import cv2
import math
import time
from loco_lib import *
#from getGps import get_gps
import gps_lib



tracking_begin_time = 0


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

def track_aruco():
    global tracking_begin_time
    zed = sl.Camera()
    
    # Add a set to track visited markers (ones we've been within 2 meters of)
    visited_markers = set()
    
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA for better accuracy
    init_params.coordinate_units = sl.UNIT.METER  # Set units to meters
    init_params.depth_minimum_distance = 0.3      # Minimum depth in meters
    init_params.coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera open failed: {status}")
        return
    
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50

    try:
        runtime_params.texture_confidence_threshold = 100
    except AttributeError:
        try:
            runtime_params.textureness_confidence_threshold = 100
        except AttributeError:
            print("Warning: Could not set texture confidence threshold")
    
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    
    # Detection parameters
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 0.7
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 1
    parameters.errorCorrectionRate = 0.8
    parameters.maxErroneousBitsInBorderRate = 0.4

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Enhanced smoothing parameters
    prev_centers = {}
    smoothing_factor = 0.9
    history_length = 10
    center_history = {}
    
    # For outlier rejection
    max_movement_threshold = 30
    
    # For tracking last seen markers
    last_seen_markers = {}
    marker_timeout = 5.0  # seconds before considering a marker lost
    
    # Create a separate window for the map

    map_size = 500  # Size of the map window


    print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")


    res = sl.Resolution()
    res.width = 720
    res.height = 404
    m2p = int(25)

    point_cloud2 = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    #pnp = np.array(point_cloud)
    #print(pnp.shape)
    start_bus()
    
    try:
        gpsjson = gps_lib.read_gps()
        initial_coordinates = (gpsjson["lat"],gpsjson["lon"])
        rover_pos = (map_size // 2, map_size // 2)
        while True:
            markers = {}
            gpsjson = gps_lib.read_gps()
            current_coordinates = (gpsjson["lat"],gpsjson["lon"])
            # Grab a new frame from the ZED
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                break
            
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            
            # Retrieve depth map and point cloud
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            pnp = np.array(point_cloud)
            print('!!!!',pnp.shape)

            # Convert ZED image to OpenCV format
            frame = image.get_data()
            
            # Fix color channels
            if len(frame.shape) == 3 and frame.shape[2] >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ensure frame is valid and has the right format
            if frame is None or frame.size == 0:
                print("Warning: Invalid frame received from ZED camera")
                continue
                
            # Check frame dimensions and channels
            if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA format
                print(f"Converting RGBA frame to BGR: {frame.shape}")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(gray)
            
            # Get current GPS position if available
            current_gps = None
            if GPS_AVAILABLE:
                try:
                    current_gps = gps_lib.read_gps()
             
                except Exception as e:
                    print(f"Error getting GPS: {e}")
            
            # Create map image
            map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Draw coordinate grid on map
            grid_spacing = 50  # pixels
            for i in range(0, map_size, grid_spacing):
                cv2.line(map_img, (i, 0), (i, map_size), (200, 200, 200), 1)
                cv2.line(map_img, (0, i), (map_size, i), (200, 200, 200), 1)
            
            # Draw rover position at center of map
            rover_pos = gps_lib.calculate_vector(initial_coordinates[0], initial_coordinates[1], current_coordinates[0], current_coordinates[1])[2]
            print("Calculated vector:",gps_lib.calculate_vector(initial_coordinates[0], initial_coordinates[1], current_coordinates[0], current_coordinates[1]))
            print("Coordinates:" ,current_coordinates)
            print("Initial Coordinates:" ,initial_coordinates)
            rover_pos = (map_size // 2 + int(rover_pos[0])*20, map_size // 2 + int(rover_pos[1])*20)
            #print(rover_pos)
            cv2.circle(map_img, rover_pos, 10, (0, 0, 255), -1)  # Red circle for rover
            
            # Draw rover direction indicator (assuming forward is up)
            cv2.line(map_img, rover_pos, (rover_pos[0], rover_pos[1] - 30), (0, 0, 255), 2)
            
            # Add compass directions
            cv2.putText(map_img, "N", (map_size//2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(map_img, "S", (map_size//2, map_size-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(map_img, "E", (map_size-20, map_size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(map_img, "W", (10, map_size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add scale indicator (assuming 1 meter = 20 pixels)
            scale_length = 40  # 2 meters
            cv2.line(map_img, (20, map_size-20), (20+scale_length, map_size-20), (0, 0, 0), 2)
            cv2.putText(map_img, "20m", (20, map_size-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Add GPS coordinates if available
            if current_coordinates:
                gps_text = f"GPS: {current_coordinates[0]:.8f}, {current_coordinates[1]:.8f}"
                cv2.putText(map_img, gps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Current time for marker timeout
            current_time = time.time()
            
            # Validate detected markers
            if ids is not None:
                valid_corners = []
                valid_ids = []
                
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    
                    # Calculate marker area
                    x1, y1 = corner[0][0]
                    x2, y2 = corner[0][1]
                    x3, y3 = corner[0][2]
                    x4, y4 = corner[0][3]
                    
                    area = abs(x1*y2+x2*y3+x3*y4+x4*y1-y1*x2-y2*x3-y3*x4-y4*x1)
                    
                    # Validate marker shape
                    sides = [
                        np.linalg.norm(corner[0][0] - corner[0][1]),
                        np.linalg.norm(corner[0][1] - corner[0][2]),
                        np.linalg.norm(corner[0][2] - corner[0][3]),
                        np.linalg.norm(corner[0][3] - corner[0][0])
                    ]
                    
                    min_side = min(sides)
                    max_side = max(sides)
                    
                    is_square = max_side / min_side < 2.0
                    min_area_threshold = 50
                    
                    if area > min_area_threshold and is_square:
                        valid_corners.append(corner)
                        valid_ids.append([marker_id])
                
                if valid_corners:
                    corners = valid_corners
                    ids = np.array(valid_ids)
                else:
                    ids = None
            
            # Process detected markers or show last remembered positions
            detected_markers = set()
            
            # If markers are detected, process them
            if ids is not None:
        
                
                # Draw all markers
                cv2.aruco.drawDetectedMarkers(display_frame, corners, ids, (0, 0, 255))
                
                # Process each marker
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    if marker_id==2:
                        detected_markers.add(marker_id)
                    marker_last_seen = dict()
                    # Update last seen time
                    marker_last_seen[marker_id] = current_time
                    
                    # Calculate center point of the marker
                    current_center = np.mean(corner[0], axis=0).astype(int)
                    center_x, center_y = current_center
                    markers[marker_id] =  center_x
                    # Get 3D position from point cloud
                    try:
                        # Get the point cloud data
                        center_x_int = int(center_x)
                        center_y_int = int(center_y)
                        marker_point = point_cloud.get_value(center_x_int, center_y_int)
                        
                        #print(np.array(point_cloud).shape)
                        # Handle ZED SDK specific format
                        if isinstance(marker_point, tuple) and len(marker_point) >= 2:
                            status = marker_point[0]
                            point_data = marker_point[1]
                            if status == sl.ERROR_CODE.SUCCESS and isinstance(point_data, np.ndarray) and len(point_data) >= 3:
                                x, y, z = point_data[0], point_data[1], point_data[2]
                                cur_gps = gps_lib.read_gps()
                                x,y= float(f"{x:.7f}"),float(f"{y:.7f}")
                                lx, ly = gps_lib.add_vector_components_to_coordinates(cur_gps[0],cur_gps[1],x,y)
                                lx,ly= float(f"{lx:.8f}"),float(f"{ly:.8f}") # Coordinates of aruco code
                                
                                
                                
                                # Check if depth is valid
                                if not np.isnan(z) and z > 0 and z < 100:
                                    # Calculate Euclidean distance
                                    distance = np.sqrt(x*x + y*y + z*z)
                                    
                                    # Apply smoothing for stability
                                    if marker_id not in center_history:
                                        center_history[marker_id] = []
                                    
                                    if marker_id in prev_centers:
                                        prev_center = prev_centers[marker_id]
                                        movement = np.linalg.norm(current_center - prev_center)
                                        
                                        if movement > max_movement_threshold:
                                            current_center = prev_center
                                    
                                    center_history[marker_id].append(current_center)
                                    
                                    if len(center_history[marker_id]) > history_length:
                                        center_history[marker_id].pop(0)
                                    
                                    weights = np.linspace(0.5, 1.0, len(center_history[marker_id]))
                                    weights = weights / np.sum(weights)
                                    
                                    smoothed_center = np.zeros(2)
                                    for j, center in enumerate(center_history[marker_id]):
                                        smoothed_center += weights[j] * center
                                    
                                    smoothed_center = smoothed_center.astype(int)
                                    prev_centers[marker_id] = smoothed_center
                                    
                                    # Store marker position for when it's not visible
                                    last_seen_markers[marker_id] = {
                                        'position': (x, y, z),
                                        'distance': distance,
                                        'screen_pos': smoothed_center
                                    }
                                    
                                    # Draw circle at center
                                    cv2.circle(display_frame, tuple(smoothed_center), 6, (0, 255, 0), -1)
                                    
                                    # Display information near the marker
                                    # Create a semi-transparent info box
                                    info_box_width = 180
                                    info_box_height = 80
                                    
                                    # Position the info box near the marker
                                    box_x = smoothed_center[0] + 20
                                    box_y = smoothed_center[1] - 40
                                    
                                    # Ensure the box stays within frame boundaries
                                    if box_x + info_box_width > display_frame.shape[1]:
                                        box_x = smoothed_center[0] - info_box_width - 20
                                    if box_y + info_box_height > display_frame.shape[0]:
                                        box_y = smoothed_center[1] - info_box_height - 20
                                    if box_y < 0:
                                        box_y = 0
                                    if box_x < 0:
                                        box_x = 0
                                    
                                    # Draw semi-transparent background
                                    overlay = display_frame.copy()
                                    cv2.rectangle(overlay, (box_x, box_y), 
                                                (box_x + info_box_width, box_y + info_box_height), 
                                                (0, 0, 0), -1)
                                    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
                                    
                                    # Draw info text
                                    cv2.putText(display_frame, f"ID: {marker_id}", 
                                              (box_x + 10, box_y + 20), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                                    cv2.putText(display_frame, f"Dist: {distance:.2f}m", 
                                              (box_x + 10, box_y + 45), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                                    cv2.putText(display_frame, f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})", 
                                              (box_x + 10, box_y + 70), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    
                                    # Draw line from marker to info box
                                    cv2.line(display_frame, tuple(smoothed_center), 
                                           (box_x, box_y + info_box_height//2), 
                                           (0, 255, 0), 1)
                                    
                                    # Calculate 2D position for map (top-down view)
                                    # ZED camera: +X right, +Y down, +Z forward
                                    # Map: center is rover, up is forward
                                    map_scale = 60  # pixels per meter
                                    
                                    # Mirror the X coordinate to fix map orientation
                                    marker_map_x = int(rover_pos[0] + x * map_scale)  # X is left-right (mirrored)
                                    marker_map_y = int(rover_pos[1] - z * map_scale)  # Z is forward-backward
                                    
                                    # Ensure marker stays within map boundaries
                                    marker_map_x = max(10, min(map_size-10, marker_map_x))
                                    marker_map_y = max(10, min(map_size-10, marker_map_y))
                                    
                                    # Draw marker on map
                                    cv2.circle(map_img, (marker_map_x, marker_map_y), 8, (0, 128, 255), -1)
                                    
                                    # Draw line from rover to marker
                                    cv2.line(map_img, rover_pos, (marker_map_x, marker_map_y), (0, 128, 255), 2)
                                    
                                    # Add marker ID and distance on map
                                    cv2.putText(map_img, f"{marker_id}: {distance:.1f}m", 
                                              (marker_map_x + 10, marker_map_y), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    except Exception as e:
                        print(f"Error processing marker {marker_id}: {e}")
            
            # Check for markers that were seen before but not in this frame
            for marker_id, data in last_seen_markers.items():
                if marker_id not in detected_markers and marker_id==2:
                    # Check if the marker was seen recently
                    if marker_id in marker_last_seen and current_time - marker_last_seen[marker_id] < marker_timeout:
                        # Draw faded marker on the map
                        x, y, z = data['position']
                        distance = data['distance']
                        
                        # Calculate map position
                        map_scale = 20  # pixels per meter
                        marker_map_x = int(rover_pos[0] + x * map_scale)  # Mirrored X
                        marker_map_y = int(rover_pos[1] - z * map_scale)  # Unchanged Z
                        
                        # Ensure marker stays within map boundaries
                        marker_map_x = max(10, min(map_size-10, marker_map_x))
                        marker_map_y = max(10, min(map_size-10, marker_map_y))
                        
                        # Draw faded marker on map (gray instead of orange)
                        cv2.circle(map_img, (marker_map_x, marker_map_y), 8, (150, 150, 150), -1)
                        
                        # Draw dashed line from rover to marker
                        dash_length = 5
                        dash_gap = 5
                        dx = marker_map_x - rover_pos[0]
                        dy = marker_map_y - rover_pos[1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        if distance > 0:
                            dx = dx / distance
                            dy = dy / distance
                            
                            for i in range(0, int(distance), dash_length + dash_gap):
                                start_x = int(rover_pos[0] + i * dx)
                                start_y = int(rover_pos[1] + i * dy)
                                end_x = int(start_x + dash_length * dx)
                                end_y = int(start_y + dash_length * dy)
                                
                                # Ensure points are within map boundaries
                                start_x = max(0, min(map_size-1, start_x))
                                start_y = max(0, min(map_size-1, start_y))
                                end_x = max(0, min(map_size-1, end_x))
                                end_y = max(0, min(map_size-1, end_y))
                                
                                cv2.line(map_img, (start_x, start_y), (end_x, end_y), (150, 150, 150), 1)
                        
                        # Add marker ID and "last seen" text
                        seconds_ago = int(current_time - marker_last_seen[marker_id])
                        cv2.putText(map_img, f"{marker_id} ({seconds_ago}s ago)", 
                                  (marker_map_x + 10, marker_map_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # Display the main frame and map
            



            z_filter = -0.35
            zed.retrieve_measure(point_cloud2, sl.MEASURE.XYZ,sl.MEM.CPU, res)
            bult2 = point_cloud2.get_data().reshape(res.width*res.height, 4)[:,:3]
            #print(np.array(bult2))
            bult2x, bult2y, bult2z = bult2[:,0] , bult2[:,1], bult2[:,2]
            # print(bult2x[110])
            # print(bult2y[110])
            # print(bult2z[110])
            bult2 = np.column_stack((bult2x,bult2z,-bult2y)) # Rotate x axis 90 degrees because point cloud and video orientation are different orientation
            # print(bult2[110])

            height_mask = (bult2[:,1] > z_filter) & (bult2[:,1] < 3.5)  # Ignore low obstacles and floor 
            distance_mask = (abs(bult2[:,2]) < 4) & (abs(bult2[:,2]) > 0.5)
            bult2 = bult2[height_mask & distance_mask]
    
            bult2 = np.column_stack((bult2[:,0], bult2[:,2]))  
            
            bult2 = (bult2 * m2p*5).astype(int)
            
            bult2 += np.array([10*m2p, 14*m2p])
               
            twod_map = np.zeros((20*m2p, 15*m2p), dtype=np.uint8) # 2D Birds view
            valid_points = (bult2[:,0] >= 0) & (bult2[:,0] < 20*m2p) & \
                         (bult2[:,1] >= 0) & (bult2[:,1] < 15*m2p)
            bult2 = bult2[valid_points]
            

            twod_map[bult2[:, 0], bult2[:, 1]] = 255
            
            # Create vertical gradient (1 at bottom, 0 at top)
            rows, cols = twod_map.shape
            gradient = np.linspace(1, 0, rows).reshape(-1, 1)  # Vertical gradient
            gradient = np.repeat(gradient, cols, axis=1)       # Extend to all columns

            # Apply gradient to the map (convert to float for multiplication)
            twod_map = twod_map.astype(np.float32) * gradient
            twod_map = twod_map.astype(np.uint8)  # Convert back to uint8
            
            # Save a colored visualization copy before thresholding
            twod_map_colored = cv2.applyColorMap(twod_map, cv2.COLORMAP_JET)
            
            # Continue with processing for obstacle detection
            twod_map = cv2.medianBlur(twod_map, 3)  # Decrease noise
            ret, twod_map = cv2.threshold(twod_map, 130, 255, cv2.THRESH_BINARY)
            twod_map = cv2.morphologyEx(twod_map, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            twod_map = cv2.flip(cv2.rotate(twod_map, cv2.ROTATE_90_CLOCKWISE), 1)
            twod_map_colored = cv2.flip(cv2.rotate(twod_map_colored, cv2.ROTATE_90_CLOCKWISE), 1)

            minimum_mean = 2    # Minimum mean of an interval angle to be considered as obstacle
            heigth,width = twod_map.shape

            angle_differince = 20   # Angle interval to search
            angle_slide = 5 # Step of the interval
            radius = heigth


            p1 = (width//2,heigth)  # origin

            p2_x = int(math.cos(math.radians(90+angle_differince//2))*radius + width/2)

            p2_y = int(heigth - math.sin(math.radians(90+angle_differince//2))*radius) 

            if p2_x < 0:
                p2_x = 0

            elif p2_x > width:
                p2_x = width

            if p2_y < 0:
                p2_y = 0

            elif p2_y > heigth:
                p2_y = heigth

            p2 = (p2_x,p2_y)    # Left dot of triangle

            p3_x = int(math.cos(math.radians(90-angle_differince//2))*radius + width/2)

            p3_y = int(heigth - math.sin(math.radians(90-angle_differince//2))*radius)

            if p3_x < 0:
                p3_x = 0

            elif p3_x > width:
                p3_x = width

            if p3_y < 0:
                p3_y = 0
                
            elif p3_y > heigth:
                p3_y = heigth

            p3 = (p3_x,p3_y)    # Right dot of triangle

            hull_points = np.array([p1,p2,p3])

            convex = cv2.convexHull(hull_points)

            # print(time.time()-tracking_begin_time)
            #print((((getContourStat(convex, twod_map)[0][0][0]) > minimum_mean), (2 in markers.keys()) , (int(time.time()-tracking_begin_time)>5)))
            
            # Check if there's an obstacle in front of the rover
            obstacle_detected = (getContourStat(convex, twod_map)[0][0][0]) > minimum_mean
            sees_aruco = 2 in markers.keys()
            marker_remembered = 2 in last_seen_markers and current_time - marker_last_seen.get(2, 0) < marker_timeout
            tracking_timed_out = int(time.time() - tracking_begin_time) > 5
            
            # Improved decision logic:
            # 1. If we see or remember an ArUco marker AND there's no obstacle, go to the marker
            # 2. If there's an obstacle, do obstacle avoidance
            # 3. If no marker and no obstacle, explore
            
            if (sees_aruco or marker_remembered) and not obstacle_detected:
                try:
                    # Try using visible marker coordinates first
                    if sees_aruco:
                        print("ArUco marker visible - navigating directly")
                        x = markers[2]
                        center_y = display_frame.shape[0] // 2  # Estimate center y if not available
                        center_visible = True
                    # If not visible but remembered, use last known position
                    elif marker_remembered:
                        print("Using remembered ArUco position")
                        x = last_seen_markers[2]['screen_pos'][0]
                        center_y = last_seen_markers[2]['screen_pos'][1]
                        center_visible = False
                    else:
                        continue
                    
                    # Skip this marker if we've already visited it
                    if 2 in visited_markers:
                        print("Marker 2 already visited, ignoring")
                        # Let the obstacle avoidance take over
                        #velocity_control_loco(0.0, 0.2)  # Just move forward slowly
                        continue
                    
                    tracking_begin_time = time.time()
                    
                    # Get current GPS position
                    current_gps = gps_lib.read_gps()
                    
                    # Calculate ArUco marker's GPS coordinates
                    try:
                        # Get the point cloud data at marker center
                        if center_visible:
                            center_x_int = int(x)
                            center_y_int = int(center_y)
                            marker_point = point_cloud.get_value(center_x_int, center_y_int)
                        else:
                            # Use last known position data
                            x, y, z = last_seen_markers[2]['position']
                            marker_distance = last_seen_markers[2]['distance']
                            marker_point = (sl.ERROR_CODE.SUCCESS, np.array([x, y, z]))
                        
                        if isinstance(marker_point, tuple) and len(marker_point) >= 2:
                            status = marker_point[0]
                            point_data = marker_point[1]
                            
                            if status == sl.ERROR_CODE.SUCCESS and isinstance(point_data, np.ndarray) and len(point_data) >= 3:
                                x, y, z = point_data[0], point_data[1], point_data[2]
                                # Calculate Euclidean distance in 3D space
                                marker_distance = np.sqrt(x*x + y*y + z*z)
                                
                                # Convert relative coordinates to GPS coordinates
                                marker_lat, marker_lon = gps_lib.add_vector_components_to_coordinates(
                                    current_gps[0], current_gps[1], x, y)
                                
                                print(f"ArUco marker GPS coordinates: {marker_lat:.8f}, {marker_lon:.8f}")
                                print(f"Distance to marker: {marker_distance:.2f} meters")
                                
                                # If already close enough to the marker (within 2 meters), mark as visited
                                if marker_distance <= 2.0:
                                    print("Within 2 meters of marker 2, marking as visited")
                                    visited_markers.add(2)  # Add marker 2 to visited set
                                    # Stop briefly to register the visit
                                    #velocity_control_loco(0.0, 0.0)
                                    time.sleep(1.0)
                                    continue  # Exit this section and let obstacle avoidance take over
                                else:
                                    # No obstacle detected, proceed with navigation
                                    print("Navigating to marker")
                                    
                                    # Get angle to target
                                    target_coords = [marker_lat, marker_lon]
                                    current_coords = (current_gps[0], current_gps[1])
                                    angle = gps_lib.calculate_vector(current_coords[0], current_coords[1], 
                                                                   target_coords[0], target_coords[1])[1]
                                    
                                    # Set initial angle
                                    from loco_lib import set_angle
                                    print(f"Setting angle: {angle} degrees")
                                    set_angle(angle)
                                    
                                    # Move forward for a short time
                                    move_duration = 3.0  # shorter duration to check obstacles more frequently
                                    #velocity_control_loco(0.0, 0.4)  # Move forward
                                    time.sleep(move_duration)
                                    
                                    # Stop briefly before next loop iteration
                                    #velocity_control_loco(0.0, 0.0)
                                    time.sleep(0.5)
                    except Exception as e:
                        print(f"Error calculating marker GPS coordinates: {e}")
                        #velocity_control_loco(0.0, 0.2)  # Move forward slowly as fallback
                        continue

                except Exception as e:
                    print(f"Error in ArUco tracking: {e}")
                    #velocity_control_loco(0.0, 0.2)  # Move forward slowly as fallback
                    continue
            
            # If obstacle detected, do obstacle avoidance
            elif obstacle_detected:
                # Decide which algorithm to apply (Aruco tracking with caution and free obstacle avoidence)
                
                angle_1 = 140
                angle_2 = angle_1-angle_differince
                
                
                interval_means = []
                while True:
                    
                    p2_x = int(math.cos(math.radians(angle_1))*radius + width/2)

                    p2_y = int(heigth - math.sin(math.radians(angle_1))*radius)

                    if p2_x < 0:
                        p2_x = 0
                    
                    elif p2_x > width:
                        p2_x = width

                    if p2_y < 0:
                        p2_y = 0
                    
                    elif p2_y > heigth:
                        p2_y = heigth

                    p2 = (p2_x,p2_y)

                    p3_x = int(math.cos(math.radians(angle_2))*radius + width/2)

                    p3_y = int(heigth - math.sin(math.radians(angle_2))*radius)

                    if p3_x < 0:
                        p3_x = 0
                    
                    elif p3_x > width:
                        p3_x = width

                    if p3_y < 0:
                        p3_y = 0
                    
                    elif p3_y > heigth:
                        p3_y = heigth

                    p3 = (p3_x,p3_y)

                    hull_points = np.array([p1,p2,p3])

                    convex = cv2.convexHull(hull_points)
                    interval_means.append((getContourStat(convex, twod_map)[0][0][0],angle_1,angle_2))            

                    angle_1 -= angle_slide
                    angle_2 -= angle_slide

                    if angle_2 <= 40:
                        break
                
                
                
                # print(interval_means[len(interval_means)//2-2:len(interval_means)//2+2])
                for index in range(len(interval_means)//2-2,len(interval_means)//2+2):
                    interval_means.insert(index-(len(interval_means)//2-2), interval_means.pop(index))

                interval_means = sorted(interval_means, key= lambda x: x[0], reverse=False)
                




                mean1, a1, a2 = interval_means[0]
                mean2, a3, a4 = interval_means[1]
                mean3, a5, a6 = interval_means[2]

                a1 = (a1+a2)/2
                a3 = (a3+a4)/2
                a5 = (a5+a6)/2

                if mean1 > minimum_mean:
                    mean1_k = 0
                    mean = mean1
                    avg_angle = a1
                
                else:

                    mean1_k = 1-(mean1/minimum_mean)
                
                    if mean2 > minimum_mean:
                        mean2_k = 0
                        mean = mean1

                        avg_angle = a1

                    
                    else:
                        mean2_k = 1-(mean2/minimum_mean)

                        if mean3 > minimum_mean:
                            mean3_k = 0

                            avg_angle = np.average([a1,a3],weights=[mean1_k,mean2_k])
                            mean = (mean1+mean2)/2
                        
                        else:
                            mean3_k = 1-(mean3/minimum_mean)
                            avg_angle = np.average([a1,a3,a5],weights=[mean1_k,mean2_k,mean3_k])
                            mean = (mean1+mean2+mean3)/3



                avg_angle = avg_angle
                mean = round(mean,3)
                
                p4_x = int(math.cos(math.radians(avg_angle))*radius + width/2)

                p4_y = int(heigth - math.sin(math.radians(avg_angle))*radius)

                if p4_x < 0:
                    p4_x = 0
                    
                elif p4_x > width:
                    p4_x = width

                if p4_y < 0:
                    p4_y = 0
                
                elif p4_y > heigth:
                    p4_y = heigth

                p4 = (p4_x,p4_y)
                cv2.line(twod_map, p1,p4, (255),4)
                

                time.sleep(0.1)      

                if mean<minimum_mean:
                    if avg_angle<85:
                        pass #velocity_control_loco(0.25,0.0)        
                    elif avg_angle>95:
                        pass #velocity_control_loco(-0.25,0.0)        
                    else:
                        pass #velocity_control_loco(0.0,0.35)     
                    print("Engelden kacma!", mean, avg_angle)

                          
                else:
                    print('Cannot pass!!: ', mean, avg_angle)
                    #velocity_control_loco(0.0,-0.2)
                     

            # If no ArUco and no obstacles, just move forward slowly to explore
            else:
                print("No obstacles or ArUco markers - exploring")
                #velocity_control_loco(0.0, 0.2)  # Move forward slowly

            cv2.imshow("Bird's Eye View (Binary)", twod_map)
            cv2.imshow("Bird's Eye View (Gradient)", twod_map_colored)
            cv2.imshow('ArUco Tracking', cv2.resize(display_frame,(800,400)))
            cv2.imshow("Map", map_img)  
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    track_aruco()

