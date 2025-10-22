#!/usr/bin/env python3
"""
Real-time emoji display: Thinking vs Eureka moment detection!
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuration
SMILE_LIFT_THRESHOLD = -0.008  # Adjust this! Higher = need bigger smile
SMILE_WIDTH_THRESHOLD = 0.15  # Alternative: wider mouth detection
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Load emoji images
try:
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")

    if smiling_emoji is None:
        raise FileNotFoundError("smile.jpg not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("plain.png not found")
    if hands_up_emoji is None:
        raise FileNotFoundError("air.jpg not found")

    # Resize emojis
    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    
except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    print("\nExpected files:")
    print("- smile.jpg (for EUREKA moment)")
    print("- plain.png (for THINKING pose)")
    print("- air.jpg (for hands up)")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("Controls:")
print("  Press 'q' to quit")
print("  THINKING")
print("  EUREKA")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "NEUTRAL"
        is_smiling = False
        hand_near_face = False
        finger_pointing_up = False

        # Get frame dimensions
        h, w, c = frame.shape

        # Check facial expression
        results_face = face_mesh.process(image_rgb)
        face_y_center = None
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                # Get face center (nose tip)
                nose_tip = face_landmarks.landmark[1]
                face_y_center = nose_tip.y
                
                # Check smile by detecting mouth corner lift (works with closed mouth!)
                left_corner = face_landmarks.landmark[61]   # Left mouth corner
                right_corner = face_landmarks.landmark[291]  # Right mouth corner
                upper_lip_center = face_landmarks.landmark[0]  # Upper lip center
                
                # Calculate if corners are higher than center (smile!)
                left_lift = upper_lip_center.y - left_corner.y
                right_lift = upper_lip_center.y - right_corner.y
                avg_lift = (left_lift + right_lift) / 2
                
                # Also check mouth width (smiles are wider)
                mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
                
                # DEBUG: Show smile detection values
                cv2.putText(frame, f'Smile lift: {avg_lift:.3f}', (10, h-100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Detect smile: corners lifted OR mouth wider
                if avg_lift > SMILE_LIFT_THRESHOLD or mouth_width > SMILE_WIDTH_THRESHOLD:
                    is_smiling = True
        
        # Check hand gestures
        results_hands = hands.process(image_rgb)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Get key hand landmarks
                index_tip = hand_landmarks.landmark[8]  # Index fingertip
                index_base = hand_landmarks.landmark[5]  # Index finger base
                wrist = hand_landmarks.landmark[0]
                
                # DEBUG: Show hand position
                cv2.putText(frame, f'Wrist Y: {wrist.y:.2f}', (10, h-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                if face_y_center is not None:
                    cv2.putText(frame, f'Face Y: {face_y_center:.2f}', (10, h-40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(frame, f'Distance: {abs(wrist.y - face_y_center):.2f}', (10, h-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Check if finger is pointing UP
                if index_tip.y < index_base.y - 0.1:  # Tip is above base
                    finger_pointing_up = True
                    # Draw special indicator
                    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                    cv2.circle(frame, (cx, cy), 20, (255, 0, 255), 3)
                    cv2.putText(frame, 'POINTING UP!', (cx+25, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Check if hand is near face (thinking pose)
                if face_y_center is not None:
                    # More sensitive detection: hand in upper 70% and closer to face
                    if wrist.y < 0.7 and abs(wrist.y - face_y_center) < 0.3:
                        hand_near_face = True
                        cx, cy = int(wrist.x * w), int(wrist.y * h)
                        cv2.putText(frame, 'HAND NEAR FACE', (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Alternative: check if index finger or middle finger is near face
                middle_tip = hand_landmarks.landmark[12]
                if face_y_center is not None:
                    if (abs(index_tip.y - face_y_center) < 0.15 or 
                        abs(middle_tip.y - face_y_center) < 0.15):
                        hand_near_face = True
        
        # Check for hands up (original feature)
        results_pose = pose.process(image_rgb)
        both_hands_up = False
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            if (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y):
                both_hands_up = True
        
        # Determine state with priority
        if finger_pointing_up and is_smiling:
            current_state = "EUREKA"
            emoji_to_display = smiling_emoji
            emoji_name = "EUREKA!"
        elif hand_near_face and not is_smiling:
            current_state = "THINKING"
            emoji_to_display = hands_up_emoji  # air.jpg for thinking pose
            emoji_name = "THINKING"
        elif both_hands_up:
            current_state = "CELEBRATION"
            emoji_to_display = hands_up_emoji
            emoji_name = "CELEBRATION"
        elif is_smiling:
            current_state = "HAPPY"
            emoji_to_display = smiling_emoji
            emoji_name = "HAPPY"
        else:
            current_state = "NEUTRAL"
            emoji_to_display = straight_face_emoji
            emoji_name = "NEUTRAL"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Display state with larger text
        cv2.putText(camera_frame_resized, emoji_name, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Show detection status
        status_y = 80
        if finger_pointing_up:
            cv2.putText(camera_frame_resized, 'Finger UP', (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            status_y += 25
        if is_smiling:
            cv2.putText(camera_frame_resized, 'Smiling', (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            status_y += 25
        if hand_near_face:
            cv2.putText(camera_frame_resized, 'Hand near face', (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()