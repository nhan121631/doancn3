import os
import pickle
import mediapipe as mp
import cv2

DATA_DIR = './data_3'
DATA_FILE = 'data.pickle'

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Đọc dữ liệu cũ nếu có
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        saved = pickle.load(f)
        data = saved['data']
        labels = saved['labels']
else:
    data = []
    labels = []

# Lấy danh sách nhãn đã có
existing_labels = set(labels)

# Lặp qua các nhãn mới
new_dirs = [d for d in os.listdir(DATA_DIR) if d not in existing_labels]
print(f"🟡 Nhãn mới phát hiện: {new_dirs}")

for dir_ in new_dirs:
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            all_hand_data = []

            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                single_hand = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    single_hand.append(lm.x - min(x_))
                    single_hand.append(lm.y - min(y_))

                all_hand_data.append(single_hand)

            if len(all_hand_data) == 1:
                all_hand_data.append([0.0] * 42)

            if len(all_hand_data) >= 2:
                data_aux = all_hand_data[0] + all_hand_data[1]
                data.append(data_aux)
                labels.append(dir_)

# Ghi đè file pickle với dữ liệu mới
with open(DATA_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"✅ Đã cập nhật {len(new_dirs)} nhãn mới vào {DATA_FILE}")
