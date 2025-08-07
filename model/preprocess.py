import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# Thêm thư mục config vào path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.gesture_classes import GESTURE_CLASSES, get_gesture_name


class DataPreprocessor:
    """Lớp xử lý tiền xử lý dữ liệu cho Hand Gesture Recognition"""
    
    def __init__(self, csv_file_path=None):
        # Tự động tìm đường dẫn đúng
        if csv_file_path is None:
            # Lấy thư mục hiện tại của file preprocess.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Đi lên 1 cấp để về thư mục HandGestureRecognition
            project_root = os.path.dirname(current_dir)
            # Tạo đường dẫn đến dataset
            self.csv_file_path = os.path.join(project_root, 'dataset', 'hand_gestures_dataset.csv')
        else:
            self.csv_file_path = csv_file_path
            
        self.scaler = None
        self.label_encoder = None
        self.df = None
        self.feature_columns = []
    
    def load_data(self):
        """Đọc dữ liệu từ CSV file"""
        print(f"📁 Loading data from {self.csv_file_path}...")
        
        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(self.csv_file_path):
            print(f"❌ File not found: {self.csv_file_path}")
            
            # Thử tìm file ở các vị trí khác
            possible_paths = [
                'dataset/hand_gestures_dataset.csv',  # Từ thư mục model
                '../dataset/hand_gestures_dataset.csv',  # Từ thư mục model
                '../../dataset/hand_gestures_dataset.csv',  # Nếu có nested folders
                os.path.join(os.getcwd(), 'dataset', 'hand_gestures_dataset.csv'),  # Từ current working directory
            ]
            
            print("🔍 Searching for dataset file in alternative locations...")
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                print(f"   Checking: {abs_path}")
                if os.path.exists(abs_path):
                    print(f"✅ Found dataset at: {abs_path}")
                    self.csv_file_path = abs_path
                    break
            else:
                # Nếu không tìm thấy ở đâu cả
                print("\n❌ Dataset file not found in any expected location!")
                print("📋 Please check:")
                print("   1. File exists: dataset/hand_gestures_dataset.csv")
                print("   2. You have collected some data using the tracking app")
                print("   3. Current working directory:", os.getcwd())
                raise FileNotFoundError(f"Dataset file not found: {self.csv_file_path}")
        
        # Đọc file CSV
        self.df = pd.read_csv(self.csv_file_path)
        print(f"✅ Loaded {len(self.df)} samples from {self.csv_file_path}")
        print(f"📊 Columns: {len(self.df.columns)} total")
        
        return self.df
    
    def explore_data(self):
        """Khám phá dữ liệu cơ bản"""
        if self.df is None:
            self.load_data()
            
        print("\n" + "="*60)
        print("📊 DATA EXPLORATION")
        print("="*60)
        
        # Thông tin cơ bản
        print(f"📈 Dataset shape: {self.df.shape}")
        print(f"📋 Columns: {len(self.df.columns)}")
        
        # Thống kê gesture classes
        print("\n🤚 Gesture Distribution:")
        gesture_counts = self.df['gesture_name'].value_counts()
        for gesture, count in gesture_counts.items():
            print(f"  • {gesture}: {count} samples")
        
        # Thống kê số tay
        print("\n👋 Number of Hands:")
        hand_counts = self.df['num_hands'].value_counts()
        for num_hands, count in hand_counts.items():
            print(f"  • {num_hands} hand(s): {count} samples")
        
        # Kiểm tra missing values
        print("\n❓ Missing Values:")
        missing_data = self.df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        if len(missing_cols) > 0:
            print(f"  Found missing values in {len(missing_cols)} columns")
            for col, count in missing_cols.items():
                print(f"    • {col}: {count} missing")
        else:
            print("  ✅ No missing values found")
    
    def check_data_sufficiency(self):
        """Kiểm tra xem dữ liệu có đủ để training không"""
        if self.df is None:
            self.load_data()
        
        print("\n" + "="*60)
        print("🔍 DATA SUFFICIENCY CHECK")
        print("="*60)
        
        total_samples = len(self.df)
        gesture_counts = self.df['gesture_name'].value_counts()
        min_samples_per_class = gesture_counts.min()
        
        print(f"📊 Total samples: {total_samples}")
        print(f"📉 Minimum samples per class: {min_samples_per_class}")
        
        # Đề xuất
        recommended_min = 50  # Tối thiểu 50 samples mỗi class
        
        if total_samples < 100:
            print("⚠️  WARNING: Dataset is very small!")
            print(f"   Recommended: At least 100 total samples")
            print(f"   Current: {total_samples} samples")
        
        if min_samples_per_class < recommended_min:
            print("⚠️  WARNING: Some classes have too few samples!")
            print(f"   Recommended: At least {recommended_min} samples per class")
            for gesture, count in gesture_counts.items():
                if count < recommended_min:
                    needed = recommended_min - count
                    print(f"   • {gesture}: {count} samples (need {needed} more)")
        
        if total_samples >= 100 and min_samples_per_class >= recommended_min:
            print("✅ Dataset looks good for training!")
        else:
            print("\n💡 SUGGESTIONS:")
            print("   1. Collect more data using your tracking app")
            print("   2. Focus on classes with fewer samples")
            print("   3. Consider data augmentation techniques")
            print("   4. Start with a simple model for proof of concept")
        
        return total_samples >= 100 and min_samples_per_class >= recommended_min
    
    def prepare_features(self, use_both_hands=False, normalize_method='minmax'):
        """
        Chuẩn bị features từ landmark data
        
        Args:
            use_both_hands (bool): Sử dụng cả 2 tay hay chỉ hand_0
            normalize_method (str): 'minmax', 'standard', hoặc 'none'
        """
        if self.df is None:
            self.load_data()
        
        print(f"\n🔧 Preparing features (use_both_hands={use_both_hands}, normalize={normalize_method})...")
        
        # Lấy các cột landmark
        if use_both_hands:
            # Sử dụng cả 2 tay
            hand_0_cols = [col for col in self.df.columns if 'hand_0_landmark' in col]
            hand_1_cols = [col for col in self.df.columns if 'hand_1_landmark' in col]
            self.feature_columns = hand_0_cols + hand_1_cols
        else:
            # Chỉ sử dụng hand_0
            self.feature_columns = [col for col in self.df.columns if 'hand_0_landmark' in col]
        
        print(f"📊 Selected {len(self.feature_columns)} feature columns")
        
        # Lấy features và xử lý missing values
        X = self.df[self.feature_columns].fillna(0).values
        
        # Chuẩn hóa dữ liệu
        if normalize_method == 'minmax':
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
        elif normalize_method == 'standard':
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            self.scaler = None
        
        # Lấy labels
        y = self.df['gesture_name'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✅ Features shape: {X_scaled.shape}")
        print(f"✅ Labels shape: {y_encoded.shape}")
        print(f"📋 Classes: {list(self.label_encoder.classes_)}")
        
        return X_scaled, y_encoded
    
    def split_data(self, X, y, test_size=0.2, val_size=0.0, random_state=42):
        """
        Chia dữ liệu thành train/validation/test
        
        Args:
            X: Features
            y: Labels
            test_size: Tỷ lệ test set
            val_size: Tỷ lệ validation set (set to 0 for small datasets)
            random_state: Random seed
        """
        print(f"\n📂 Splitting data...")
        
        # Kiểm tra số lượng samples cho mỗi class
        unique, counts = np.unique(y, return_counts=True)
        min_samples = min(counts)
        
        print(f"📊 Class distribution: {dict(zip(unique, counts))}")
        print(f"📉 Minimum samples per class: {min_samples}")
        
        # Nếu dataset quá nhỏ hoặc có class với ít samples, không dùng stratify
        use_stratify = min_samples >= 2 and len(X) >= 10
        
        if not use_stratify:
            print("⚠️ Small dataset detected - using simple random split (no stratification)")
            stratify_param = None
        else:
            stratify_param = y
        
        # Nếu val_size = 0 hoặc dataset quá nhỏ, chỉ chia train/test
        if val_size == 0 or len(X) < 20:
            print(f"📂 Splitting into train/test only (train: {1-test_size:.1f}, test: {test_size:.1f})")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
            )
            X_val, y_val = None, None
        else:
            # Chia train và temp (chứa validation + test)
            print(f"📂 Splitting into train/val/test ({1-test_size-val_size:.1f}/{val_size:.1f}/{test_size:.1f})")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size+val_size, random_state=random_state, stratify=stratify_param
            )
            
            # Chia temp thành validation và test
            val_ratio = val_size / (test_size + val_size)
            temp_stratify = y_temp if use_stratify else None
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=1-val_ratio, random_state=random_state, stratify=temp_stratify
            )
        
        print(f"✅ Train set: {X_train.shape[0]} samples")
        if X_val is not None:
            print(f"✅ Validation set: {X_val.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessors(self, save_dir='../model/saved_models'):
        """Lưu scaler và label encoder"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if self.scaler is not None:
            scaler_path = os.path.join(save_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"💾 Saved scaler to {scaler_path}")
        
        if self.label_encoder is not None:
            encoder_path = os.path.join(save_dir, 'label_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"💾 Saved label encoder to {encoder_path}")
    
    def load_preprocessors(self, save_dir='../model/saved_models'):
        """Load scaler và label encoder"""
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        encoder_path = os.path.join(save_dir, 'label_encoder.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"📥 Loaded scaler from {scaler_path}")
        
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"📥 Loaded label encoder from {encoder_path}")


def main():
    """Hàm main để test preprocessing"""
    # Khởi tạo preprocessor
    preprocessor = DataPreprocessor()
    
    # Khám phá dữ liệu
    preprocessor.explore_data()
    
    # Kiểm tra tính đủ của dữ liệu
    is_sufficient = preprocessor.check_data_sufficiency()
    
    # Chuẩn bị features (chỉ dùng 1 tay)
    X, y = preprocessor.prepare_features(use_both_hands=False, normalize_method='minmax')
    
    # Chia dữ liệu (không dùng validation set cho dataset nhỏ)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y, test_size=0.2, val_size=0.0)
    
    # Lưu preprocessors
    preprocessor.save_preprocessors()
    
    if not is_sufficient:
        print("\n⚠️  Dataset may be too small for robust training!")
        print("   Consider collecting more data before training a model.")
    
    print("\n🎉 Preprocessing completed successfully!")
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = main()