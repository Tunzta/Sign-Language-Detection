import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Clean the data: ensure each item is a flat list of 42 numbers
clean_data = []
clean_labels = []

for i, (item, label) in enumerate(zip(data_dict['data'], data_dict['labels'])):
    if isinstance(item, (list, np.ndarray)) and len(item) == 42 and all(isinstance(x, (float, int)) for x in item):
        clean_data.append(item)
        clean_labels.append(label)
    else:
        print(f"Skipping malformed item at index {i} (type: {type(item)}, length: {len(item) if hasattr(item, '__len__') else 'N/A'})")

# Convert to NumPy arrays
data = np.array(clean_data, dtype=float)
labels = np.array(clean_labels)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
