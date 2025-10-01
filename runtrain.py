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
    "--img", "832",                     # Larger image size for better small object detection
    "--batch", "12",                    # Adjusted for larger image size
    "--epochs", "300",                  # Keep extended training
    f"--data={dataset}/data.yaml",
    "--weights", "yolov5s.pt",          # Use standard weights for stability
    "--hyp", "data/hyps/hyp.km.yaml",   # Your optimized hyperparameters
    "--cfg", "models/yolov5s-km-ca-neck.yaml",  # Your P2-P5 model
    "--name", "km-no-tiling-832",
    "--device", "0",
    "--optimizer", "AdamW",             # Keep AdamW
    "--workers", "8",                   
    "--cache", "disk",
    "--rect",                          # Keep rectangular training
    # "--tiling",                      # REMOVED TILING
    "--cos-lr",                        # Cosine learning rate scheduler
    "--label-smoothing", "0.1",        # Label smoothing
    "--patience", "50",                # Early stopping patience
    "--save-period", "25",             # Save checkpoint every 25 epochs
    "--multi-scale",                   # Add multi-scale training for robustness
])