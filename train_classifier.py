import pickle
import joblib  # Để lưu mô hình thay vì pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Tải dữ liệu
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Lấy dữ liệu và nhãn
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Kiểm tra nếu dữ liệu và nhãn có cùng kích thước
assert len(data) == len(labels), "Dữ liệu và nhãn không cùng kích thước!"

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Khởi tạo mô hình RandomForest
model = RandomForestClassifier()

# Huấn luyện mô hình
model.fit(x_train, y_train)

# Dự đoán với tập kiểm tra
y_predict = model.predict(x_test)

# Tính toán độ chính xác
score = accuracy_score(y_predict, y_test)

# In kết quả
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Lưu mô hình sử dụng joblib thay vì pickle (lưu trữ mô hình dễ dàng hơn)
joblib.dump(model, 'model.joblib')

# Hoặc nếu bạn vẫn muốn sử dụng pickle:
# with open('model.p', 'wb') as f:
#     pickle.dump({'model': model}, f)
