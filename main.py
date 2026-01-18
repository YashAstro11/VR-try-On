import cv2
import time
import mediapipe as mp
import numpy as np
import math
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def create_clothing_templates():
    """Create properly sized clothing templates if they don't exist"""
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    # Clothing templates with proper aspect ratios
    templates = [
        {"name": "tshirt", "width": 400, "height": 520, "color": (100, 100, 200), "type": "tshirt"},
        {"name": "dress_shirt", "width": 400, "height": 560, "color": (255, 255, 255), "type": "shirt"},
        {"name": "jacket", "width": 450, "height": 608, "color": (50, 50, 50), "type": "jacket"},
        {"name": "polo_shirt", "width": 380, "height": 475, "color": (0, 128, 0), "type": "polo"}
    ]
    
    for template in templates:
        filename = f"{assets_dir}/{template['name']}.png"
        if not os.path.exists(filename):
            create_single_template(template, filename)

def create_single_template(template, filename):
    """Create a single clothing template"""
    width, height = template["width"], template["height"]
    img = np.zeros((height, width, 4), dtype=np.uint8)
    color = template["color"]
    
    if template["type"] == "tshirt":
        # T-shirt shape
        create_tshirt_shape(img, width, height, color)
    elif template["type"] == "shirt":
        # Dress shirt shape
        create_shirt_shape(img, width, height, color)
    elif template["type"] == "jacket":
        # Jacket shape
        create_jacket_shape(img, width, height, color)
    elif template["type"] == "polo":
        # Polo shirt shape
        create_polo_shape(img, width, height, color)
    
    cv2.imwrite(filename, img)
    print(f"Created: {filename}")

def create_tshirt_shape(img, width, height, color):
    """Create T-shirt shape"""
    h, w = height, width
    
    # Main body
    body_top = h // 6
    body_bottom = h - h // 10
    cv2.rectangle(img, (w//4, body_top), (w*3//4, body_bottom), color, -1)
    
    # Neck hole
    neck_radius = w // 8
    cv2.circle(img, (w//2, body_top), neck_radius, (0, 0, 0, 0), -1)
    
    # Sleeves
    sleeve_width = w // 4
    sleeve_height = h // 4
    cv2.ellipse(img, (0, body_top + sleeve_height//2), (sleeve_width, sleeve_height), 
                0, 270, 450, color, -1)
    cv2.ellipse(img, (w, body_top + sleeve_height//2), (sleeve_width, sleeve_height), 
                0, 90, 270, color, -1)

def create_shirt_shape(img, width, height, color):
    """Create dress shirt shape"""
    h, w = height, width
    
    # Main body
    body_top = h // 8
    body_bottom = h - h // 15
    cv2.rectangle(img, (w//4, body_top), (w*3//4, body_bottom), color, -1)
    
    # Collar
    collar_points = np.array([
        [w//3, body_top],
        [w//2, body_top - h//12],
        [w*2//3, body_top]
    ], np.int32)
    cv2.fillPoly(img, [collar_points], color)
    
    # Sleeves (more formal)
    sleeve_width = w // 5
    sleeve_height = h // 5
    cv2.rectangle(img, (w//4 - sleeve_width, body_top), 
                 (w//4, body_top + sleeve_height), color, -1)
    cv2.rectangle(img, (w*3//4, body_top), 
                 (w*3//4 + sleeve_width, body_top + sleeve_height), color, -1)

def create_jacket_shape(img, width, height, color):
    """Create jacket shape"""
    h, w = height, width
    
    # Main body (wider)
    body_top = h // 7
    body_bottom = h - h // 12
    cv2.rectangle(img, (w//5, body_top), (w*4//5, body_bottom), color, -1)
    
    # Collar
    collar_height = h // 10
    cv2.rectangle(img, (w//3, body_top - collar_height//2), 
                 (w*2//3, body_top), color, -1)
    
    # Large sleeves
    sleeve_width = w // 4
    sleeve_height = h // 4
    cv2.ellipse(img, (0, body_top + sleeve_height//2), (sleeve_width, sleeve_height), 
                0, 270, 450, color, -1)
    cv2.ellipse(img, (w, body_top + sleeve_height//2), (sleeve_width, sleeve_height), 
                0, 90, 270, color, -1)

def create_polo_shape(img, width, height, color):
    """Create polo shirt shape"""
    h, w = height, width
    
    # Main body
    body_top = h // 6
    body_bottom = h - h // 10
    cv2.rectangle(img, (w//4, body_top), (w*3//4, body_bottom), color, -1)
    
    # Polo collar
    collar_width = w // 6
    collar_height = h // 12
    cv2.rectangle(img, (w//2 - collar_width//2, body_top - collar_height//2),
                 (w//2 + collar_width//2, body_top + collar_height//2), color, -1)
    
    # Sleeves
    sleeve_width = w // 5
    sleeve_height = h // 5
    cv2.ellipse(img, (0, body_top + sleeve_height//2), (sleeve_width, sleeve_height), 
                0, 270, 450, (0, 0, 0, 0), -1)
    cv2.ellipse(img, (w, body_top + sleeve_height//2), (sleeve_width, sleeve_height), 
                0, 90, 270, (0, 0, 0, 0), -1)

# Create clothing templates
create_clothing_templates()

# Load clothing images
clothing_images = {
    "T-Shirt": cv2.imread("assets/tshirt.png", cv2.IMREAD_UNCHANGED),
    "Shirt": cv2.imread("assets/dress_shirt.png", cv2.IMREAD_UNCHANGED),
    "Jacket": cv2.imread("assets/jacket.png", cv2.IMREAD_UNCHANGED),
    "Polo": cv2.imread("assets/polo_shirt.png", cv2.IMREAD_UNCHANGED),
}

# Verify images loaded correctly
for name, img in clothing_images.items():
    if img is None:
        print(f"Error: Could not load {name} image")
        # Create a placeholder
        clothing_images[name] = np.zeros((100, 100, 4), dtype=np.uint8)
    else:
        print(f"Loaded: {name} - Size: {img.shape[1]}x{img.shape[0]}")

clothing_keys = list(clothing_images.keys())
selected_index = 0
selected_clothing = clothing_keys[selected_index] if clothing_keys else None

# UI button state
button_positions = {}
hover_start_time = None
hovered_button = None
CLICK_HOLD_DURATION = 1.5
zoom_scale = 1.0

# Alignment adjustments
vertical_adjust = -30
horizontal_adjust = 0
size_adjust = 1.0

# Get distance between two landmarks
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Improved body box detection with better alignment
def get_upper_body_box(landmarks, frame_shape):
    h, w = frame_shape
    
    try:
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Check visibility
        if any(lm.visibility < 0.5 for lm in [ls, rs, lh, rh]):
            return None
        
        # Calculate shoulder width and body height
        shoulder_width = abs(ls.x - rs.x) * w
        body_height = (max(lh.y, rh.y) - min(ls.y, rs.y)) * h
        
        # Calculate center point between shoulders
        center_x = (ls.x + rs.x) / 2 * w
        center_y = (ls.y + rs.y) / 2 * h
        
        # Adjust box dimensions for better fit
        box_width = shoulder_width * 2.2  # Wider to cover arms
        box_height = body_height * 1.6    # Taller to cover properly
        
        # Position box - centered on shoulders
        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height * 0.4)  # Start above shoulders
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height * 0.6)  # Extend below hips
        
        return (x1, y1, x2 - x1, y2 - y1)
    
    except (IndexError, KeyError):
        return None

# Optimized overlay function
def overlay_clothing(frame, clothing_img, box, scale=1.0):
    if box is None:
        return
        
    x, y, w, h = box
    
    # Apply scale
    w, h = int(w * scale), int(h * scale)
    
    # Adjust position to keep center aligned
    x = x - (w - box[2]) // 2
    y = y - (h - box[3]) // 2
    
    # Apply manual adjustments
    x += horizontal_adjust
    y += vertical_adjust
    w = int(w * size_adjust)
    h = int(h * size_adjust)
    
    # Boundary checks
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return
    
    if clothing_img.shape[2] != 4:
        return
    
    try:
        # Resize clothing image
        clothing_resized = cv2.resize(clothing_img, (w, h), interpolation=cv2.INTER_AREA)
        
        # Extract alpha channel
        alpha_channel = clothing_resized[:, :, 3] / 255.0
        alpha_channel = np.expand_dims(alpha_channel, axis=2)
        
        # Get the region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Ensure dimensions match
        if roi.shape[:2] == clothing_resized[:, :, :3].shape[:2]:
            # Blend images
            blended = (1 - alpha_channel) * roi.astype(float) + alpha_channel * clothing_resized[:, :, :3].astype(float)
            frame[y:y+h, x:x+w] = blended.astype(np.uint8)
            
    except Exception as e:
        print(f"Overlay error: {e}")

# Enhanced button drawing with hover feedback
def draw_buttons(frame):
    x_start = 20
    y = 20
    w = 120
    h = 40
    spacing = 20
    
    for i, key in enumerate(clothing_images.keys()):
        x = x_start + i * (w + spacing)
        
        # Button colors based on state
        if key == selected_clothing:
            color = (0, 255, 0)  # Green for selected
        elif key == hovered_button:
            color = (255, 255, 0)  # Yellow for hover
        else:
            color = (255, 0, 0)  # Red for normal
        
        # Draw button with border
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Draw progress bar for hover-to-click
        if key == hovered_button and hover_start_time:
            elapsed = time.time() - hover_start_time
            progress = min(elapsed / CLICK_HOLD_DURATION, 1.0)
            progress_width = int(w * progress)
            cv2.rectangle(frame, (x, y + h - 5), (x + progress_width, y + h), (0, 255, 255), -1)
        
        # Center text
        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        button_positions[key] = (x, y, w, h)

# Draw status information
def draw_status(frame, zoom_scale, hand_count):
    status_text = f"Zoom: {zoom_scale:.1f}x | Hands: {hand_count}"
    cv2.putText(frame, status_text, (20, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show adjustment info
    adj_text = f"Adj: V{vertical_adjust} H{horizontal_adjust} S{size_adjust:.1f}"
    cv2.putText(frame, adj_text, (20, frame.shape[0] - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if selected_clothing:
        cv2.putText(frame, f"Selected: {selected_clothing}", 
                    (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Enhanced gesture detection
def detect_gestures(hand_landmarks, pose_landmarks, frame_shape):
    gestures = {
        "remove_clothing": False,
    }
    
    h, w = frame_shape
    
    # Remove clothing gesture: hand above head
    if pose_landmarks:
        nose_y = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * h
        for hand in hand_landmarks:
            wrist_y = hand.landmark[0].y * h
            if wrist_y < nose_y - 50:  # Added threshold
                gestures["remove_clothing"] = True
                break
    
    return gestures

# Start video capture with error handling
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# FPS calculation
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

print("Virtual Try-On Application Started!")
print("Controls:")
print("- Hover over buttons for 1.5 seconds to select clothing")
print("- Use two hands with index fingers to zoom")
print("- Raise hand above head to remove clothing")
print("- Press SPACE to reset selection")
print("- W/S: Move clothing up/down")
print("- A/D: Move clothing left/right") 
print("- +/-: Adjust clothing size")
print("- R: Reset adjustments")
print("- ESC: Exit")

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame")
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    draw_buttons(frame)
    
    hand_count = 0
    if hand_results.multi_hand_landmarks:
        hand_count = len(hand_results.multi_hand_landmarks)
        landmarks_list = [hand.landmark for hand in hand_results.multi_hand_landmarks]

        # Zoom with 2 hands
        if len(landmarks_list) == 2:
            d = distance(landmarks_list[0][8], landmarks_list[1][8])
            zoom_scale = np.clip(d * 10, 0.8, 2.0)
        else:
            zoom_scale = 1.0

        for handLms in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            index_tip = handLms.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # Hover-to-click
            hit = None
            for item, pos in button_positions.items():
                x, y, bw, bh = pos
                if x < cx < x + bw and y < cy < y + bh:
                    hit = item
                    break
                    
            if hit == hovered_button:
                if hover_start_time and time.time() - hover_start_time > CLICK_HOLD_DURATION:
                    selected_clothing = hit
                    hovered_button = None
                    hover_start_time = None
                    print(f"Selected: {selected_clothing}")
            else:
                hovered_button = hit
                hover_start_time = time.time() if hit else None

            # Enhanced gesture detection
            gestures = detect_gestures(hand_results.multi_hand_landmarks, 
                                     pose_results.pose_landmarks, frame.shape[:2])
            if gestures["remove_clothing"]:
                selected_clothing = None
                print("Clothing removed (hand above head)")

    # Draw pose landmarks and overlay clothing
    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        if selected_clothing and selected_clothing in clothing_images:
            landmarks = pose_results.pose_landmarks.landmark
            box = get_upper_body_box(landmarks, frame.shape[:2])
            if box:
                overlay_clothing(frame, clothing_images[selected_clothing], box, zoom_scale)

    # Draw status and FPS
    draw_status(frame, zoom_scale, hand_count)
    
    # Calculate FPS
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()
    
    cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "W/S:Move Up/Down  A/D:Move Left/Right  +/-:Size  R:Reset", 
                (frame.shape[1] - 600, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Virtual Try-On", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # Spacebar to reset
        selected_clothing = None
        zoom_scale = 1.0
        print("Selection reset")
    elif key == ord('w'):  # Move up
        vertical_adjust -= 5
        print(f"Vertical adjustment: {vertical_adjust}")
    elif key == ord('s'):  # Move down  
        vertical_adjust += 5
        print(f"Vertical adjustment: {vertical_adjust}")
    elif key == ord('a'):  # Move left
        horizontal_adjust -= 5
        print(f"Horizontal adjustment: {horizontal_adjust}")
    elif key == ord('d'):  # Move right
        horizontal_adjust += 5
        print(f"Horizontal adjustment: {horizontal_adjust}")
    elif key == ord('+'):  # Increase size
        size_adjust += 0.1
        print(f"Size adjustment: {size_adjust:.1f}")
    elif key == ord('-'):  # Decrease size
        size_adjust = max(0.5, size_adjust - 0.1)
        print(f"Size adjustment: {size_adjust:.1f}")
    elif key == ord('r'):  # Reset adjustments
        vertical_adjust = -30
        horizontal_adjust = 0
        size_adjust = 1.0
        print("Adjustments reset")

cap.release()
cv2.destroyAllWindows()
print("Application closed successfully!")