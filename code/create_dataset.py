import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data_3'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
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

            # Nếu chỉ có 1 tay → padding 0 cho đủ 2 tay
            if len(all_hand_data) == 1:
                all_hand_data.append([0.0] * 42)

            # Nếu có đúng 2 tay → nối lại
            if len(all_hand_data) >= 2:
                data_aux = all_hand_data[0] + all_hand_data[1]
                data.append(data_aux)
                labels.append(dir_)
                
file_path = 'data.pickle'
# if os.path.exists(file_path): 
#     os.remove(file_path)
# Save file
with open(file_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
