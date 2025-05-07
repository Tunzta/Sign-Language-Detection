import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load trained model from pickle
try:
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
except Exception as e:
    print("Error loading model:", e)
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for video stream
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Label mapping
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

# Set confidence threshold
confidence_threshold = 0.75

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark coordinates
            x_, y_, data_aux = [], [], []
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Predict character and its confidence
            probabilities = model.predict_proba([np.asarray(data_aux)])[0]
            confidence = np.max(probabilities)
            predicted_class = np.argmax(probabilities)

            # If confidence is above the threshold, display the prediction
            if confidence > confidence_threshold:
                predicted_character = labels_dict[predicted_class]
            else:
                predicted_character = ''  # Do not display anything if confidence is too low

            # Draw bounding box and prediction
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            if predicted_character:  # Only display text if we have a valid prediction
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
