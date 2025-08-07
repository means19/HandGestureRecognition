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
import HandTrackingModule as htm


def main():
    """Main application function for hand gesture recognition data collection."""
    # Initialize variables
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector(maxHands=2)  # Detect up to 2 hands
    collector = htm.DatasetCollector()
    
    # Tracking mode
    tracking_mode = False
    
    # Print welcome message and instructions
    print_welcome_message(collector)
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                continue
                
            img = cv2.flip(img, 1)  # Horizontal flip for more natural view
            
            # Find hands
            img = detector.findHands(img, draw=tracking_mode)
            allHands = detector.findAllHands(img, draw=tracking_mode)
            
            # Draw UI elements
            img = draw_ui_elements(img, tracking_mode, allHands, pTime)
            
            # Calculate FPS
            pTime = update_fps(img, pTime)
            
            # Display the image
            cv2.imshow("Hand Gesture Data Collection", img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if handle_key_press(key, tracking_mode, allHands, collector):
                if key == ord('q'):
                    break
                elif key == ord('k'):
                    tracking_mode = not tracking_mode
                    print(f"Tracking mode: {'ON' if tracking_mode else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


def print_welcome_message(collector):
    """Print welcome message and instructions."""
    print("=" * 60)
    print("🤚 HAND GESTURE RECOGNITION DATA COLLECTION 🤚")
    print("=" * 60)
    print("📋 AVAILABLE GESTURE CLASSES:")
    print_all_classes()
    print("-" * 60)
    print("📋 COMMANDS:")
    print("  • Press 'k' to toggle tracking mode ON/OFF")
    print("  • Press the corresponding key to save gesture data")
    print("  • Press 's' to show dataset summary")
    print("  • Press 'c' to clear screen")
    print("  • Press 'q' to quit")
    print("-" * 60)
    print(f"📁 Dataset location: {collector.csv_file}")
    print("-" * 60)
    print(collector.get_dataset_summary())
    print("=" * 60)


def draw_ui_elements(img, tracking_mode, allHands, pTime):
    """Draw UI elements on the image."""
    height, width = img.shape[:2]
    
    # Status indicator
    status_color = (0, 255, 0) if tracking_mode else (0, 0, 255)
    status_text = "🟢 TRACKING ON" if tracking_mode else "🔴 TRACKING OFF"
    cv2.putText(img, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Hand count
    hand_count_text = f"👋 Hands: {len(allHands)}"
    cv2.putText(img, hand_count_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Hand types
    for i, hand in enumerate(allHands):
        hand_type_text = f"Hand {i+1}: {hand['type']}"
        cv2.putText(img, hand_type_text, (10, 115 + i*35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Instructions at bottom
    instruction_text = "Press 'k' for tracking | Keys from config to save | 's' for summary | 'q' to quit"
    text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(img, instruction_text, (10, height - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def update_fps(img, pTime):
    """Calculate and display FPS."""
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    cv2.putText(img, f'FPS: {int(fps)}', (img.shape[1]-120, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cTime


def handle_key_press(key, tracking_mode, allHands, collector):
    """Handle keyboard input."""
    if key == ord('q'):
        return True
    elif key == ord('k'):
        return True
    elif key == ord('c'):
        os.system('cls' if os.name == 'nt' else 'clear')
        print_welcome_message(collector)
    elif key == ord('s'):  # Show dataset summary
        print("\n" + "=" * 60)
        print("📊 DATASET SUMMARY:")
        print("=" * 60)
        print(collector.get_dataset_summary())
        print("=" * 60)
    else:
        # Check if pressed key corresponds to any gesture class
        pressed_char = chr(key) if key < 128 else None
        if pressed_char and is_valid_class_id(pressed_char):
            save_gesture_data(tracking_mode, allHands, collector, pressed_char)
        elif pressed_char and pressed_char.isdigit():
            # Handle numeric keys for backward compatibility
            save_gesture_data(tracking_mode, allHands, collector, pressed_char)
    
    return False


def save_gesture_data(tracking_mode, allHands, collector, class_id):
    """Save gesture data with validation."""
    if not tracking_mode:
        print("⚠️  Turn ON tracking mode first (press 'k')")
        return
    
    if len(allHands) == 0:
        print("⚠️  No hands detected! Position your hand(s) in view.")
        return
    
    # Save the data
    gesture_name = get_gesture_name(class_id)
    collector.save_gesture_data(allHands, class_id)
    print(f"✅ Gesture data saved: '{gesture_name}' (class '{class_id}') with {len(allHands)} hand(s)")


if __name__ == "__main__":
    main()