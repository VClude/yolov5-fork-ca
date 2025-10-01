# Alternative Training Script - Proven Configuration for Small Objects
# Use this if the current approach doesn't work well

dataset = "/home/muhammad_ardiansyah/yolov5-fork-ca/Kelenjar-Minyak-Yolo-Standard-8"

import subprocess

# Option 1: Standard YOLOv5s with larger image size (RECOMMENDED)
print("🚀 Starting OPTION 1: Standard YOLOv5s with proven settings...")
subprocess.run([
    "python", "train.py",
    "--img", "960",                     # Large image size for small objects
    "--batch", "8",                     # Smaller batch for larger images
    "--epochs", "200",                  # Standard epoch count
    f"--data={dataset}/data.yaml",
    "--weights", "yolov5s.pt",          # Standard YOLOv5s weights
    "--hyp", "data/hyps/hyp.scratch-low.yaml",  # Use standard hyperparameters
    "--cfg", "yolov5s.yaml",            # Use standard YOLOv5s model
    "--name", "standard-yolov5s-960",
    "--device", "0",
    "--optimizer", "SGD",               # Standard SGD optimizer
    "--workers", "8",
    "--cache", "disk",
    "--rect",                          # Rectangular training
    "--cos-lr",                        # Cosine LR
    "--patience", "30",
    "--save-period", "20",
])

# Uncomment to try Option 2: Your custom model with conservative settings
"""
print("🚀 Starting OPTION 2: Custom model with conservative settings...")
subprocess.run([
    "python", "train.py",
    "--img", "832",                     # Good balance for small objects
    "--batch", "12",                    
    "--epochs", "250",                  
    f"--data={dataset}/data.yaml",
    "--weights", "yolov5s.pt",          
    "--hyp", "data/hyps/hyp.scratch-low.yaml",  # Use proven hyperparameters
    "--cfg", "models/yolov5s-km-ca-neck.yaml",  # Your custom model
    "--name", "custom-conservative-832",
    "--device", "0",
    "--optimizer", "SGD",               # Standard optimizer
    "--workers", "8",
    "--cache", "disk",
    "--rect",
    "--cos-lr",
    "--patience", "40",
    "--save-period", "25",
])
"""
