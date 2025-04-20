import pickle
from collections import Counter

file_path = 'data.pickle'

with open(file_path, 'rb') as file:
    data = pickle.load(file)
    labels = data.get('labels', [])

if labels:
    label_counts = Counter(labels)
    print("Số lượng mẫu theo nhãn:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
else:
    print("Không tìm thấy nhãn trong dữ liệu.")
