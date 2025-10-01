# Enhanced Validation Script for Small Object Detection
# Use this for final validation with optimized thresholds

import subprocess
import os

def validate_small_objects():
    """Validate model with optimized settings for small objects"""
    
    # Best weights path
    best_weights = "runs/train/km-tiling-optimized/weights/best.pt"
    
    # Validation with optimized NMS thresholds for small objects
    subprocess.run([
        "python", "val.py",
        f"--weights", best_weights,
        f"--data", "/home/muhammad_ardiansyah/yolov5-fork-ca/Kelenjar-Minyak-Yolo-Standard-8/data.yaml",
        "--img", "640",
        "--conf-thres", "0.15",  # Lower confidence threshold for small objects
        "--iou-thres", "0.3",    # Lower IoU threshold to prevent over-suppression
        "--device", "0",
        "--verbose",
        "--save-txt",
        "--save-conf",
        "--name", "small-object-validation"
    ])

if __name__ == "__main__":
    validate_small_objects()
