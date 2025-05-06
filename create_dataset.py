import os
import pickle
import mediapipe as mp
import cv2

# Khởi tạo module MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo đối tượng hands với các tham số cấu hình
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'  # Thư mục chứa dữ liệu ảnh

# Danh sách để lưu trữ dữ liệu ảnh và nhãn
data = []
labels = []

# Duyệt qua tất cả các thư mục con trong thư mục dữ liệu
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Danh sách chứa các dữ liệu landmarks của một ảnh

        # Đọc ảnh
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue  # Bỏ qua nếu ảnh không thể đọc được

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển ảnh từ BGR sang RGB
        results = hands.process(img_rgb)  # Xử lý ảnh để tìm landmarks

        # Kiểm tra nếu có hand landmarks trong ảnh
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []  # Danh sách chứa các tọa độ x của landmarks
                y_ = []  # Danh sách chứa các tọa độ y của landmarks

                # Lấy các tọa độ x, y của các landmarks trên tay
                for i in range(len(hand_landmarks.landmark)):
                    x_ = hand_landmarks.landmark[i].x
                    y_ = hand_landmarks.landmark[i].y

                # Chuẩn hóa các landmarks bằng cách trừ đi giá trị nhỏ nhất của x và y
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Trừ giá trị nhỏ nhất của x
                    data_aux.append(y - min(y_))  # Trừ giá trị nhỏ nhất của y

            # Lưu dữ liệu và nhãn vào danh sách
            data.append(data_aux)
            labels.append(dir_)

# Lưu dữ liệu và nhãn vào file pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data has been saved in 'data.pickle'")
