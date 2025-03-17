import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv3D, MaxPooling3D, Flatten, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt


# Tắt cảnh báo
tf.get_logger().setLevel('ERROR')

# Cấu hình
DATA_PATH = os.path.join('MP_Data')
actions = [
    'want_rice', 'want_drink', 'change_dish', 'dessert',
    'satisfied', 'not_satisfied', 'ask_time', 'clear_tray'
]
sequence_length = 20

# Chuẩn bị dữ liệu
sequences, labels = [], []
for action in actions:
    print(f"Đang xử lý dữ liệu cho hành động: {action}")
    for sequence in range(30):
        window = []
        success = True
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            except Exception as e:
                success = False
                break
        if success and len(window) == sequence_length:
            sequences.append(window)
            labels.append(actions.index(action))

print(f"\nSố lượng mẫu thu thập được: {len(sequences)}")

# Chuyển đổi thành numpy array
X = np.array(sequences)
y = np.array(labels)
print(f"Shape của dữ liệu X: {X.shape}")
print(f"Shape của nhãn y: {y.shape}")

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vẽ biểu đồ phân bố dữ liệu
labels, counts = np.unique(y_train, return_counts=True)
labels_test, counts_test = np.unique(y_test, return_counts=True)

plt.figure(figsize=(10, 5))

# Vẽ biểu đồ cho tập train
plt.subplot(1, 2, 1)
plt.bar(labels, counts, color='blue', alpha=0.7)
plt.xticks(labels, actions, rotation=45)
plt.xlabel("Hành động")
plt.ylabel("Số mẫu")
plt.title("Số lượng mẫu trong tập Train")

# Vẽ biểu đồ cho tập test
plt.subplot(1, 2, 2)
plt.bar(labels_test, counts_test, color='red', alpha=0.7)
plt.xticks(labels_test, actions, rotation=45)
plt.xlabel("Hành động")
plt.ylabel("Số mẫu")
plt.title("Số lượng mẫu trong tập Test")

plt.tight_layout()
plt.show()


# Reshape dữ liệu cho mô hình
X_train = X_train.reshape(-1, sequence_length, 63, 1, 1)
X_test = X_test.reshape(-1, sequence_length, 63, 1, 1)

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Tạo mô hình với cấu trúc đơn giản hơn
model = Sequential([
    # Input layer
    Input(shape=(sequence_length, 63, 1, 1)),
    
    # First Conv3D block
    Conv3D(32, (3, 3, 1), activation='relu', padding='same'),
    MaxPooling3D((1, 2, 1)),
    
    # Second Conv3D block
    Conv3D(64, (3, 3, 1), activation='relu', padding='same'),
    MaxPooling3D((1, 2, 1)),
    
    # Flatten layer
    Flatten(),
    
    # Dense layers
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(actions), activation='softmax')
])

# Biên dịch mô hình
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

print("\nCấu trúc mô hình:")
print(model.summary())

# Callbacks
callbacks = [
    ModelCheckpoint(
        'model3d.h5',
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
]

print("\nBắt đầu huấn luyện mô hình...")

# Huấn luyện
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Lưu mô hình
model.save('model3d.h5')
print("\nĐã lưu mô hình thành công!") 
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
np.save('y_train.npy', y_train)