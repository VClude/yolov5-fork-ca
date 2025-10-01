# import roboflow
# rf = roboflow.Roboflow(api_key="AkZNNZBdY0QrIN62pzp0")
# project = rf.workspace("kelenjar-minyak").project("kelenjar-minyak-yolo-standard-ke3j5")
# dataset = project.version(4).download("yolov5")

# import subprocess
# subprocess.run([
#     "python", "train.py",
#     "--img", "1280",
#     "--batch", "8",
#     "--epochs", "200",
#     f"--data={dataset.location}/data.yaml",
#     "--weights", "yolov5s.pt",
#     "--hyp", "hyp.dcn.precision.yaml",
#     "--cfg", "yolov5s-cav2-dcn.yaml",
#     "--name", "km-v5s-ca-dcn-1280-single-class",
#     "--device", "0",
#     "--optimizer", "AdamW",
#     "--workers", "8",
#     "--cache", "disk"
# ])

dataset = "/home/muhammad_ardiansyah/yolov5-fork-ca/Kelenjar-Minyak-Yolo-Standard-8"

import subprocess
subprocess.run([
    "python", "train.py",
    "--img", "640",                     # Optimal for 1920x1080 tiling
    "--batch", "16",                    # Conservative batch size for tiling
    "--epochs", "300",                  # Increased for small object learning
    f"--data={dataset}/data.yaml",
    "--weights", "yolov5s.pt",          # Use your 4-head model
    "--hyp", "data/hyps/hyp.km.yaml",   # Your optimized small object hyperparameters
    "--cfg", "models/yolov5s-km-ca-neck.yaml",  # Your P2-P5 model
    "--name", "km-v5s-km-cav2-neck-tiling",
    "--device", "0",
    "--optimizer", "AdamW",             # Better for small objects
    "--workers", "8",                   # Reduced to avoid bottleneck
    "--cache", "disk",
    "--rect",                          # Rectangular training for 16:9 images
    "--tiling",                        # Enable tiling for small objects
    "--cos-lr",                        # Cosine learning rate scheduler
    "--label-smoothing", "0.1",        # Label smoothing for better generalization
    "--patience", "50",                # Early stopping patience
    "--save-period", "25",             # Save checkpoint every 25 epochs
])