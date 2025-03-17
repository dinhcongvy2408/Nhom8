import os
import numpy as np

DATA_PATH = 'MP_Data'
actions = ['want_rice', 'want_drink', 'change_dish', 'dessert',
           'satisfied', 'not_satisfied', 'ask_time', 'clear_tray']

total_sequences = 0
print("Kiểm tra dữ liệu đã thu thập:")
print("-" * 50)

for action in actions:
    action_sequences = 0
    for sequence in range(30):
        path = os.path.join(DATA_PATH, action, str(sequence))
        if os.path.exists(path):
            files = os.listdir(path)
            if len(files) == 20:  # Kiểm tra đủ 20 frames
                action_sequences += 1
    total_sequences += action_sequences
    print(f"{action}: {action_sequences}/30 sequences hoàn chỉnh")

print("-" * 50)
print(f"Tổng số sequences hoàn chỉnh: {total_sequences}/{8*30}") 