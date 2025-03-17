import cv2
import numpy as np
import os
import mediapipe as mp

# Tăng số frame cho mỗi sequence
DATA_PATH = os.path.join('MP_Data')
actions = [
    'want_rice',        # Muốn ăn cơm
    'want_drink',       # Muốn uống nước
    'change_dish',      # Muốn đổi món ăn
    'dessert',          # Món tráng miệng
    'satisfied',        # Hài lòng món ăn
    'not_satisfied',    # Không hài lòng
    'ask_time',         # Muốn hỏi mấy giờ
    'clear_tray'        # Muốn phục vụ dọn khay
]

action_labels = {
    'want_rice': 'Muốn ăn cơm',
    'want_drink': 'Muốn uống nước',
    'change_dish': 'Muốn đổi món ăn',
    'dessert': 'Món tráng miệng',
    'satisfied': 'Hài lòng món ăn',
    'not_satisfied': 'Không hài lòng',
    'ask_time': 'Muốn hỏi mấy giờ',
    'clear_tray': 'Muốn phục vụ dọn khay'
}

no_sequences = 30  # Số video cho mỗi hành động
sequence_length = 20  # Giảm xuống 20 frames thay vì 40

# Tạo thư mục
for action in actions:
    dirpath = os.path.join(DATA_PATH, action)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    for sequence in range(no_sequences):
        sequence_path = os.path.join(dirpath, str(sequence))
        if not os.path.exists(sequence_path):
            os.makedirs(sequence_path)

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for action in actions:
    # Hiển thị hướng dẫn cho mỗi hành động
    print(f'\nChuẩn bị thu thập dữ liệu cho: {action_labels[action]}')
    print('Hướng dẫn thực hiện:')
    
    # Hướng dẫn cụ thể cho từng hành động
    if action == 'want_rice':
        print('- Giơ tay ngang ngực')
        print('- Làm động tác múc cơm')
        print('- Thực hiện chậm và rõ ràng')
    elif action == 'want_drink':
        print('- Giơ tay lên miệng')
        print('- Làm động tác uống nước')
        print('- Giữ động tác 2-3 giây')
    # Thêm hướng dẫn cho các hành động khác...
    
    input('Nhấn Enter khi sẵn sàng...')
    
    for sequence in range(no_sequences):
        print(f'\nChuẩn bị cho video {sequence + 1}/{no_sequences}')
        
        # Đếm ngược lâu hơn
        for i in range(7):  # Tăng thời gian đếm ngược
            print(f'{7-i}...')
            cv2.waitKey(1000)
        
        # Hiển thị "BẮT ĐẦU!"
        print("BẮT ĐẦU!")
        
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Xử lý frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Vẽ landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
            
            # Hiển thị thông tin chi tiết hơn
            cv2.putText(image, f'Thu thap: {action_labels[action]}', (10,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Video: {sequence + 1}/{no_sequences}', (10,60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Frame: {frame_num + 1}/{sequence_length}', (10,90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Thêm thanh tiến trình
            progress = int((frame_num / sequence_length) * 400)
            cv2.rectangle(image, (10, 100), (410, 120), (0, 0, 0), 2)
            cv2.rectangle(image, (10, 100), (10 + progress, 120), (0, 255, 0), -1)
            
            cv2.imshow('Thu thap du lieu', image)
            
            # Lưu keypoints
            if results.multi_hand_landmarks:
                keypoints = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
hands.close() 