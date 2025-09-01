@echo off
setlocal enabledelayedexpansion

REM Set dataset path
set DATASET=V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml

REM Loop through each model folder in runs\train\yolo
for /d %%D in (runs\train\yolo\*) do (
    if exist "%%D\weights\best.pt" (
        set "MODEL_DIR=%%D"
        set "RESULT_DIR=%%D\eval_results"
        if not exist "!RESULT_DIR!" mkdir "!RESULT_DIR!"

        REM Run visualize.py and save output
        python visualize.py --weights "%%D\weights\best.pt" --imgsz 640 --batch-size 1 > "!RESULT_DIR!\visualize_output.txt"
        if exist model_graph.png move /Y model_graph.png "!RESULT_DIR!\model_graph.png"
    )
)

echo All evaluations and visualizations are complete.