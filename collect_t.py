import os
import sys
import cv2

label_index = int(sys.argv[1])
DATA_DIR = './data_3'
dataset_size = 100

cap = cv2.VideoCapture(0)
save_path = os.path.join(DATA_DIR, str(label_index))
os.makedirs(save_path, exist_ok=True)

print(f"Collecting data for label {label_index}")

# Chờ người dùng nhấn 'q' để bắt đầu
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Hiển thị thông báo trên khung hình
    message = "Press 'q' to collect images"
    cv2.putText(frame, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Bắt đầu thu thập ảnh
counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    if not ret:
        break

    # Hiển thị số lượng ảnh đã thu thập
    cv2.putText(frame, f"Collecting...: {counter+1}/{dataset_size}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imwrite(os.path.join(save_path, f'{counter}.jpg'), frame)
    counter += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):  # Cho phép dừng sớm
        break

cap.release()
cv2.destroyAllWindows()
