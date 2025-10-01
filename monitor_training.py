# Monitor Training Progress for Small Object Detection
# Use this to track training metrics in real-time

import subprocess
import sys
import time

def monitor_training():
    """Monitor training with TensorBoard for small object metrics"""
    
    print("🚀 Starting YOLOv5 Small Object Detection Training...")
    print("📊 Monitor progress at: http://localhost:6006")
    print("🔍 Key metrics to watch:")
    print("   - val/box_loss (should decrease steadily)")
    print("   - val/obj_loss (should decrease)")
    print("   - val/cls_loss (should decrease)")
    print("   - metrics/mAP_0.5 (should increase)")
    print("   - metrics/precision (should stabilize >0.8)")
    print("   - metrics/recall (should stabilize >0.7)")
    print("\n" + "="*50)
    
    # Start TensorBoard in background
    subprocess.Popen([
        sys.executable, "-m", "tensorboard.main", 
        "--logdir", "runs/train",
        "--port", "6006",
        "--reload_interval", "30"
    ])
    
    print("✅ TensorBoard started on http://localhost:6006")
    print("📈 Training metrics will update every 30 seconds")
    print("🛑 Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            time.sleep(60)  # Check every minute
            print("📊 Training in progress... Check TensorBoard for real-time metrics")
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped. Training continues in background.")

if __name__ == "__main__":
    monitor_training()
