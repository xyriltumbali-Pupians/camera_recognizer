import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def draw_3d_cube(img, center, size=100):
    cx, cy = center
    s = size // 2

    # Define 8 vertices of a cube in 3D (x, y, z)
    # Using a simple orthographic projection for visualization
    nodes = np.array([
        [cx-s, cy-s, 0], [cx+s, cy-s, 0], [cx+s, cy+s, 0], [cx-s, cy+s, 0],
        [cx-s, cy-s, size], [cx+s, cy-s, size], [cx+s, cy+s, size], [cx-s, cy+s, size]
    ])

    edges = [
        (0,1), (1,2), (2,3), (3,0), # Back face
        (4,5), (5,6), (6,7), (7,4), # Front face
        (0,4), (1,5), (2,6), (3,7)  # Connecting lines
    ]

    for start, end in edges:
        p1 = (nodes[start][0], nodes[start][1])
        p2 = (nodes[end][0], nodes[end][1])
        cv2.line(img, p1, p2, (0, 255, 0), 3)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm8 = hand_landmarks.landmark[8]
            h, w, c = img.shape
            cx, cy = int(lm8.x * w), int(lm8.y * h)

            draw_3d_cube(img, (cx, cy))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("3D Air Drawing", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()