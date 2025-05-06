import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model đã huấn luyện từ pickle
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Khởi tạo video capture
cap = cv2.VideoCapture(2)

# Khởi tạo mediapipe cho nhận diện bàn tay
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary ánh xạ các label
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Đọc frame từ video capture
    ret, frame = cap.read()

    # Kiểm tra nếu có frame
    if not ret:
        break

    H, W, _ = frame.shape

    # Chuyển đổi frame sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý kết quả từ Mediapipe
    results = hands.process(frame_rgb)

    # Kiểm tra nếu có bàn tay trong ảnh
    if results.multi_hand_landmarks:
        # Vẽ landmarks của bàn tay trên ảnh
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Thu thập các điểm landmark của bàn tay
        for hand_landmarks in results.multi_hand_landmarks:
            x_.clear()
            y_.clear()
            data_aux.clear()

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Chuẩn hóa giá trị các điểm
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Xác định tọa độ để vẽ hình chữ nhật quanh bàn tay
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Dự đoán ký tự từ mô hình
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Vẽ hình chữ nhật quanh bàn tay và hiển thị ký tự dự đoán
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Hiển thị frame với các landmarks vẽ lên
    cv2.imshow('frame', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
