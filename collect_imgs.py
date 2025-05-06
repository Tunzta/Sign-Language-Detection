import cv2
import os

# Cấu hình
DATA_DIR = 'data'
number_of_classes = 3      # Số lớp cần thu thập
dataset_size = 100         # Số ảnh mỗi lớp
camera_index = 0           # Thay đổi nếu camera không hoạt động

# Tạo thư mục dữ liệu
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Mở camera
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Cannot open camera. Check the connection or try a different index (1, 2, ...).")
    exit()

for class_id in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Collecting data for class {class_id}")

    # Chờ người dùng nhấn Q để bắt đầu
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from camera.")
            continue

        msg = f"Class {class_id} - Press Q to start or ESC to quit"
        cv2.putText(frame, msg, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC
            print("Exiting program.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Bắt đầu thu thập ảnh
    count = 0
    while count < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        # Hiển thị số lượng ảnh đã thu thập
        msg = f"Class {class_id} - Saving image {count + 1}/{dataset_size}"
        cv2.putText(frame, msg, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('frame', frame)

        img_path = os.path.join(class_dir, f'{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC để thoát giữa chừng
            print("Exiting during image collection.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"Finished collecting {dataset_size} images for class {class_id}")

cap.release()
cv2.destroyAllWindows()
