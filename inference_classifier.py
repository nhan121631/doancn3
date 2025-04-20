import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import pickle
import mediapipe as mp
import numpy as np
import time
import subprocess 
from tkinter import messagebox, simpledialog
import os




LABEL_FILE = "labels.txt"
DATA_DIR = "./data_3"

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Load labels from file
def load_labels():
    labels = {}
    with open(LABEL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':')
                labels[int(key)] = value
    return labels

labels_dict = load_labels()

# Ghi thêm label mới vào file
def save_label(index, label):
    with open(LABEL_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{index}:{label}")

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🖐️ Nhận diện ngôn ngữ ký hiệu")

        self.output_text = ""
        self.last_prediction = ""
        self.last_time = time.time()
        self.wait_time = 1.5

        self.create_widgets()

        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def create_widgets(self):
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(padx=10, pady=10)

        control_frame = ttk.LabelFrame(self.root, text="Kết quả")
        control_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(control_frame, text="Cụm từ nhận diện", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10)

        self.text_display = tk.Text(control_frame, height=2, font=("Arial", 14, "bold"))
        self.text_display.pack(padx=10, pady=5, fill="x")

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=5)

        clear_btn = ttk.Button(btn_frame, text="🗑️ Clear", command=self.clear_text)
        clear_btn.pack(side=tk.LEFT, padx=5)

        icon_img = Image.open("logo-vku.jpg").resize((120, 60))
        self.icon_tk = ImageTk.PhotoImage(icon_img)
        
        collect_btn = ttk.Button(btn_frame, text="Thu thập ảnh", command=self.start_collecting_images)
        collect_btn.pack(side=tk.LEFT, padx=5)
        
        logo_label = ttk.Label(control_frame, image=self.icon_tk)
        logo_label.pack(side=tk.LEFT, padx=5)
        
        text_DA = ttk.Label(text="Đồ án chuyên ngành 3", font=('Arial', 12, 'bold'))
        text_DA.pack(side=tk.LEFT, padx=5)

    def clear_text(self):
        self.output_text = ""
        self.text_display.delete("1.0", tk.END)

    def delete_last(self):
        self.output_text = self.output_text[:-1]
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert(tk.END, self.output_text)
    
    def reload_model_and_labels(self):
        global model, labels_dict
        # Reload the model
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']
        # Reload the labels
        labels_dict = load_labels()
    def create_dataset(self):
        try:
            messagebox.showinfo("Thông báo", "Creating dataset...")
            subprocess.run(["python", "create_dataset_t.py"], check=True)
            messagebox.showinfo("Xong", "Đã tạo xong dataset")
        except subprocess.CalledProcessError:
            messagebox.showerror("Lỗi", "Lỗi khi chạy create_dataset.py")

    def train(self):
        try:
            messagebox.showinfo("Thông báo", "Trainning...")
            subprocess.run(["python", "train_classifier.py"], check=True)
            self.reload_model_and_labels()
            messagebox.showinfo("Xong", "Đã train xong")
        except subprocess.CalledProcessError:
            messagebox.showerror("Lỗi", "Lỗi khi chạy train_classifier.py")
            
    def start_collecting_images(self):
    # Nhập ký tự
        label = simpledialog.askstring("Thu thập", "Nhập ký tự cần thu thập:")
        if not label:
            return

        labels = load_labels()
        if label in labels.values():
            messagebox.showinfo("Thông báo", f"Ký tự '{label}' đã tồn tại.")
            return

        # Tìm số label mới chưa có
        if labels:
            new_index = max(labels.keys()) + 1
        else:
            new_index = 0

        # Ghi vào file
        save_label(new_index, label)
        os.makedirs(os.path.join(DATA_DIR, str(new_index)), exist_ok=True)

        # Gọi collect_t.py và truyền index
        self.cap.release()
        try:
            subprocess.run(["python", "collect_t.py", str(new_index)], check=True)
            messagebox.showinfo("Xong", f"Đã thu thập ảnh cho '{label}' (label {new_index})")
            self.create_dataset()
            self.train()
        except subprocess.CalledProcessError:
            messagebox.showerror("Lỗi", "Lỗi khi chạy collect_t.py")
        # Mở lại webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap.open(0)

        self.running = True
        self.update_frame()
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        #frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        H, W, _ = frame.shape
        data_aux = []
        all_hand_data = []
        bbox_x, bbox_y = [], []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                x_, y_, single_hand = [], [], []
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    single_hand.append(lm.x - min(x_))
                    single_hand.append(lm.y - min(y_))
                bbox_x.extend(x_)
                bbox_y.extend(y_)
                all_hand_data.append(single_hand)

            # Đảm bảo có đủ dữ liệu 2 tay (nếu chỉ 1 tay thì thêm tay giả)
            if len(all_hand_data) == 1:
                all_hand_data.append([0.0] * 42)

            if len(all_hand_data) >= 2:
                data_aux = all_hand_data[0] + all_hand_data[1]
                if len(data_aux) == 84:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    if predicted_character == self.last_prediction:
                        if time.time() - self.last_time >= self.wait_time:
                            if predicted_character == 'DELETE':
                                self.delete_last()
                            elif predicted_character == 'Space':
                                self.output_text += ' '
                                print('space')
                            else:
                                self.output_text += predicted_character
                            self.text_display.delete("1.0", tk.END)
                            self.text_display.insert(tk.END, self.output_text)
                            self.last_time = time.time()
                    else:
                        self.last_prediction = predicted_character
                        self.last_time = time.time()

                    # Vẽ bounding box và nhãn
                    x1 = int(min(bbox_x) * W) - 10
                    y1 = int(min(bbox_y) * H) - 10
                    x2 = int(max(bbox_x) * W) + 10
                    y2 = int(max(bbox_y) * H) + 10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

        # Convert ảnh để hiển thị lên Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        if self.running:
            self.root.after(10, self.update_frame)

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
