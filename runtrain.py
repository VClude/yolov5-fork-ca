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

import roboflow
rf = roboflow.Roboflow(api_key="AkZNNZBdY0QrIN62pzp0")
project = rf.workspace("kelenjar-minyak").project("kelenjar-minyak-yolo-standard-ke3j5")
dataset = project.version(3).download("yolov5")

import subprocess
subprocess.run([
    "python", "train.py",
    "--img", "640",
    "--batch", "64",
    "--epochs", "200",
    f"--data={dataset.location}/data.yaml",
    "--weights", "yolov5s.pt",
    "--hyp", "hyp.dcn.precision640.yaml",
    "--cfg", "yolov5s-cav2-eca.yaml",
    "--name", "km-v5s-ca-eca-640-single-class",
    "--device", "0",
    "--optimizer", "AdamW", 
    "--workers", "32",
    "--cache", "disk"
])