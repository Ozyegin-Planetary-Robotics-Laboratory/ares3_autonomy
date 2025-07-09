import cv2
import numpy as np

# --- 1. DEFINE YOUR MULTIPLE CUSTOM PATTERNS HERE ---
# A dictionary where keys are the pattern names (strings) and
# values are the NumPy arrays representing the pattern (0=black, 1=white).
CUSTOM_PATTERNS = {
    "airlock": np.array([
        [1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1]
    ], dtype=np.uint8),

    "lava_tube": np.array([
        [1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1]
    ], dtype=np.uint8),
}

# --- 2. HELPER FUNCTION (Unchanged) ---
def match_custom_pattern(marker_image, custom_pattern):
    pattern_size = custom_pattern.shape[0]
    full_marker_grid_size = pattern_size + 2
    if marker_image.shape[0] < full_marker_grid_size or marker_image.shape[1] < full_marker_grid_size: return False, None
    cell_size_x = marker_image.shape[1] // full_marker_grid_size
    cell_size_y = marker_image.shape[0] // full_marker_grid_size
    extracted_bits = np.zeros((pattern_size, pattern_size), dtype=np.uint8)
    for y in range(pattern_size):
        for x in range(pattern_size):
            cell_center_x = int((x + 1.5) * cell_size_x)
            cell_center_y = int((y + 1.5) * cell_size_y)
            if 0 <= cell_center_y < marker_image.shape[0] and 0 <= cell_center_x < marker_image.shape[1]:
                if marker_image[cell_center_y, cell_center_x] > 127: extracted_bits[y, x] = 1
            else: return False, None
    for i in range(4):
        rotated_pattern = np.rot90(custom_pattern, k=i)
        if np.array_equal(extracted_bits, rotated_pattern): return True, i
    return False, None

# --- 3. MAIN DETECTION SCRIPT (Updated with console output for clarity) ---
def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    print("Starting camera feed. Press 'q' to quit.")
    print(f"Recognizing {len(CUSTOM_PATTERNS)} patterns: {list(CUSTOM_PATTERNS.keys())}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, _, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        all_corners = []
        if corners: all_corners.extend(corners)
        if rejected: all_corners.extend(rejected)

        # --- NEW: List to hold all confirmed detections for this frame ---
        detections_in_frame = []

        if len(all_corners) > 0:
            # === THIS IS THE KEY LOOP THAT HANDLES MULTIPLE MARKERS ===
            # It runs for every single potential marker found in the frame.
            for marker_corners in all_corners:
                if marker_corners.shape[0] != 1 or marker_corners.shape[1] != 4: continue
                
                warp_size = 160 
                dst_pts = np.array([[0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]], dtype='float32')
                M = cv2.getPerspectiveTransform(marker_corners.astype('float32'), dst_pts)
                warped = cv2.warpPerspective(gray, M, (warp_size, warp_size))
                _, warped_thresh = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
                
                found_match_name = None
                found_rotation = None
                for name, pattern_array in CUSTOM_PATTERNS.items():
                    is_match, rotation = match_custom_pattern(warped_thresh, pattern_array)
                    if is_match:
                        found_match_name = name
                        found_rotation = rotation
                        break 

                if found_match_name:
                    # --- NEW: Add the successful detection to our list ---
                    detections_in_frame.append({'name': found_match_name, 'rotation': found_rotation*90})

                    # Draw green box and label for the matched marker
                    cv2.polylines(frame, [marker_corners.astype(int)], True, (0, 255, 0), 2, cv2.LINE_AA)
                    text_pos = (int(marker_corners[0][0][0]), int(marker_corners[0][0][1]) - 10)
                    cv2.putText(frame, f"{found_match_name}", text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.polylines(frame, [marker_corners.astype(int)], True, (0, 0, 255), 1)

        # --- NEW: Print the list of all detections from this frame to the console ---
        if detections_in_frame:
            print(f"Detected {len(detections_in_frame)} markers: {detections_in_frame}")

        cv2.imshow('Multi-Pattern ArUco Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()