import cv2
import time
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load transparent clothing images
clothing_images = {
    "T-Shirt": cv2.imread("assets/tshirt.png", cv2.IMREAD_UNCHANGED),
    "Shirt": cv2.imread("assets/dress_shirt.png", cv2.IMREAD_UNCHANGED),
    "Jacket": cv2.imread("assets/jacket.png", cv2.IMREAD_UNCHANGED),
    "Polo": cv2.imread("assets/polo_shirt.png", cv2.IMREAD_UNCHANGED),
}
clothing_keys = list(clothing_images.keys())
selected_index = 0
selected_clothing = clothing_keys[selected_index]

# UI button state
button_positions = {}
hover_start_time = None
hovered_button = None
CLICK_HOLD_DURATION = 1.5
zoom_scale = 1.0

# Get distance between two landmarks
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Get body box for overlay
def get_upper_body_box(landmarks, frame_shape):
    h, w = frame_shape
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    if ls.visibility < 0.5 or rs.visibility < 0.5 or lh.visibility < 0.5:
        return None
    x1 = int(min(ls.x, rs.x) * w)
    x2 = int(max(ls.x, rs.x) * w)
    y1 = int(min(ls.y, rs.y) * h)
    y2 = int(lh.y * h)
    return x1, y1, x2 - x1, y2 - y1

# Overlay clothing image
def overlay_clothing(frame, clothing_img, box, scale=1.0):
    x, y, w, h = box
    w, h = int(w * scale), int(h * scale)
    x -= (w - box[2]) // 2
    y -= (h - box[3]) // 2
    if clothing_img.shape[2] != 4:
        return
    clothing_resized = cv2.resize(clothing_img, (w, h), interpolation=cv2.INTER_AREA)
    for i in range(h):
        for j in range(w):
            if y+i >= frame.shape[0] or x+j >= frame.shape[1] or x+j < 0 or y+i < 0:
                continue
            alpha = clothing_resized[i, j, 3] / 255.0
            if alpha > 0:
                frame[y+i, x+j] = (1 - alpha) * frame[y+i, x+j] + alpha * clothing_resized[i, j, :3]

# Draw selection buttons
def draw_buttons(frame):
    x_start = 20
    y = 20
    w = 120
    h = 40
    spacing = 20
    for i, key in enumerate(clothing_images.keys()):
        x = x_start + i * (w + spacing)
        color = (0, 255, 0) if key == selected_clothing else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.putText(frame, key, (x + 10, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        button_positions[key] = (x, y, w, h)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    draw_buttons(frame)

    if hand_results.multi_hand_landmarks:
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
            else:
                hovered_button = hit
                hover_start_time = time.time() if hit else None

            # Remove gesture: hand above head
            if pose_results.pose_landmarks:
                nose_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * h
                if cy < nose_y:
                    selected_clothing = None

    if pose_results.pose_landmarks and selected_clothing:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark
        box = get_upper_body_box(landmarks, frame.shape[:2])
        if box:
            overlay_clothing(frame, clothing_images[selected_clothing], box, zoom_scale)

    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
