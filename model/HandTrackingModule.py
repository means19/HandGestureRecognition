import cv2
import mediapipe as mp
import time
import os
import csv
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.gesture_classes import GESTURE_CLASSES, get_gesture_name, is_valid_class_id, print_all_classes
from datetime import datetime


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def findAllHands(self, img, draw=True):
        """Trả về vị trí của tất cả các tay được phát hiện"""
        allHands = []
        if self.results.multi_hand_landmarks:
            for handIdx, handLms in enumerate(self.results.multi_hand_landmarks):
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                
                # Xác định tay trái hay tay phải
                handType = "Unknown"
                if self.results.multi_handedness:
                    handType = self.results.multi_handedness[handIdx].classification[0].label
                
                allHands.append({
                    "type": handType,
                    "landmarks": lmList
                })
                
                if draw:
                    for landmark in lmList:
                        cv2.circle(img, (landmark[1], landmark[2]), 5, (255, 0, 255), cv2.FILLED)
                    # Vẽ label tay trái/phải
                    if len(lmList) > 0:
                        cv2.putText(img, handType, (lmList[0][1], lmList[0][2]-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return allHands


class DatasetCollector:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.csv_file = os.path.join(dataset_path, "hand_gestures_dataset.csv")
        self.create_dataset_folder()
        self.initialize_csv()
        # Sử dụng gesture_classes từ config file
        self.gesture_classes = GESTURE_CLASSES
        
    def create_dataset_folder(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            
    def initialize_csv(self):
        """Tạo file CSV với header nếu chưa tồn tại"""
        if not os.path.exists(self.csv_file):
            # Tạo header cho CSV
            header = ['timestamp', 'class_id', 'gesture_name', 'num_hands']
            
            # Thêm cột cho landmarks của 2 tay (21 landmarks mỗi tay, mỗi landmark có x,y)
            for hand_idx in range(2):  # Tối đa 2 tay
                hand_label = f"hand_{hand_idx}"
                header.append(f"{hand_label}_type")  # Left/Right/None
                for landmark_idx in range(21):  # 21 landmarks mỗi tay
                    header.extend([
                        f"{hand_label}_landmark_{landmark_idx}_x",
                        f"{hand_label}_landmark_{landmark_idx}_y"
                    ])
            
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                
            print(f"Created CSV file: {self.csv_file}")
            
    def save_gesture_data(self, hands_data, class_id):
        if is_valid_class_id(class_id):
            gesture_name = get_gesture_name(class_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Chuẩn bị dữ liệu cho CSV row
            row_data = [timestamp, class_id, gesture_name, len(hands_data)]
            
            # Xử lý dữ liệu cho 2 tay (padding với None nếu thiếu)
            for hand_idx in range(2):
                if hand_idx < len(hands_data):
                    hand = hands_data[hand_idx]
                    row_data.append(hand['type'])
                    
                    # Thêm landmarks (21 landmarks, mỗi landmark có x, y)
                    for landmark_idx in range(21):
                        if landmark_idx < len(hand['landmarks']):
                            landmark = hand['landmarks'][landmark_idx]
                            row_data.extend([landmark[1], landmark[2]])  # x, y
                        else:
                            row_data.extend([None, None])  # Padding nếu thiếu landmarks
                else:
                    # Tay không tồn tại - padding với None
                    row_data.append(None)  # hand_type
                    for landmark_idx in range(21):
                        row_data.extend([None, None])  # x, y
            
            # Ghi vào CSV
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)
                
            print(f"Saved to CSV: {gesture_name} - {len(hands_data)} hand(s)")
            return True
        return False
    
    def get_dataset_summary(self):
        """Trả về thống kê dataset"""
        if not os.path.exists(self.csv_file):
            return "No dataset found"
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                rows = list(reader)
                
            if len(rows) <= 1:  # Chỉ có header hoặc empty
                return "Dataset is empty"
            
            # Đếm số lượng mỗi class
            class_counts = {}
            for row in rows[1:]:  # Bỏ qua header
                gesture_name = row[2]
                class_counts[gesture_name] = class_counts.get(gesture_name, 0) + 1
            
            summary = f"Total samples: {len(rows)-1}\n"
            for gesture, count in class_counts.items():
                summary += f"{gesture}: {count} samples\n"
            
            return summary
            
        except Exception as e:
            return f"Error reading dataset: {str(e)}"


