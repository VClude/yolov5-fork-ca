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

        REM Run val.py and save output
        python val.py --weights "%%D\weights\best.pt" --data "%DATASET%" --img 640 --batch-size 8 --task val --name eval --project "!RESULT_DIR!" > "!RESULT_DIR!\val_output.txt"
    )
)

echo All evaluations and visualizations are complete.