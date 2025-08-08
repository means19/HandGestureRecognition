"""
Gesture Classes Configuration
Định nghĩa các class cho Hand Gesture Recognition

Cách sử dụng:
- Thêm/sửa/xóa gesture classes trong GESTURE_CLASSES
- Key: Phím bấm hoặc class_id (string)
- Value: Tên gesture (string)
"""

GESTURE_CLASSES = {
    '0': 'fist',           # Nắm đấm
    '1': 'point',          # Một ngón
    '2': 'two',            # Hai ngón (peace/victory)
    '3': 'palm',           # Bàn tay
}

# Danh sách tất cả gesture names (để kiểm tra duplicate)
ALL_GESTURE_NAMES = list(GESTURE_CLASSES.values())

# Các phím shortcut đặc biệt (không phải số)
SPECIAL_KEYS = {
    'a': 'thumbs_up',
}

def get_gesture_name(class_id):
    """Trả về gesture name từ class_id"""
    return GESTURE_CLASSES.get(class_id, 'unknown')

def get_all_classes():
    """Trả về dict tất cả classes"""
    return GESTURE_CLASSES.copy()

def get_class_count():
    """Trả về số lượng classes"""
    return len(GESTURE_CLASSES)

def is_valid_class_id(class_id):
    """Kiểm tra class_id có hợp lệ không"""
    return class_id in GESTURE_CLASSES

def print_all_classes():
    """In ra tất cả classes để tham khảo"""
    print("=== GESTURE CLASSES ===")
    for key, value in GESTURE_CLASSES.items():
        key_desc = f"'{key}'" if key.isalpha() else key
        print(f"Press {key_desc}: {value}")
    print("=======================")

if __name__ == "__main__":
    print_all_classes()
