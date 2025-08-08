"""
Statistics tracking and management
"""
import json
import os
from datetime import datetime
from config.settings import Config

class StatisticsManager:
    """Manages application statistics and logging"""
    
    def __init__(self):
        self.total_frames = 0
        self.detection_count = 0
        self.start_time = None
        self.session_start = datetime.now()
        
        # Create logs directory
        os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    def start_session(self):
        """Start timing the session"""
        self.start_time = datetime.now()
    
    def update_frame_stats(self, has_detection=False):
        """Update frame statistics"""
        self.total_frames += 1
        if has_detection:
            self.detection_count += 1
    
    def get_session_stats(self):
        """Get current session statistics"""
        if self.start_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
        else:
            total_time = 0
        
        detection_rate = (
            (self.detection_count / self.total_frames) * 100 
            if self.total_frames > 0 else 0
        )
        
        avg_fps = self.total_frames / total_time if total_time > 0 else 0
        
        return {
            'total_time': total_time,
            'total_frames': self.total_frames,
            'detection_count': self.detection_count,
            'detection_rate': detection_rate,
            'avg_fps': avg_fps
        }
    
    def save_session_stats(self, additional_data=None):
        """Save session statistics to file"""
        try:
            stats = {
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                },
                'performance': self.get_session_stats(),
                'config': {
                    'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
                    'prediction_history_size': Config.PREDICTION_HISTORY_SIZE,
                    'smoothing_window': Config.SMOOTHING_WINDOW
                }
            }
            
            # Add additional data if provided
            if additional_data:
                stats.update(additional_data)
            
            # Save to file
            filename = f"session_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(Config.LOG_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"✓ Statistics saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"✗ Error saving statistics: {e}")
            return None
    
    def print_session_summary(self):
        """Print session summary to console"""
        stats = self.get_session_stats()
        
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Total time: {stats['total_time']:.2f} seconds")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Frames with hand detection: {stats['detection_count']}")
        print(f"Detection rate: {stats['detection_rate']:.1f}%")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print("="*50)

class FPSCounter:
    """Simple FPS counter"""
    
    def __init__(self):
        self.fps_counter = 0
        self.fps_start_time = None
        self.current_fps = 0
    
    def start(self):
        """Start FPS counting"""
        self.fps_start_time = datetime.now()
    
    def update(self):
        """Update FPS counter"""
        if self.fps_start_time is None:
            self.start()
        
        self.fps_counter += 1
        current_time = datetime.now()
        
        elapsed = (current_time - self.fps_start_time).total_seconds()
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_fps(self):
        """Get current FPS"""
        return self.current_fps