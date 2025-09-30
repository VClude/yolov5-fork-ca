@REM python train.py --epochs 1000 --data V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml --weights V:\Work\yv5-ca-stm\app\runs\train_ca\yolov5_stomata_v3_CA3x\weights\best.pt --cfg yolov5s-cav2.yaml --cache disk --device 0 --batch-size 8 --workers 4 --name=yolov5s-cav2-km-transfer-learning-st --hyp=data\hyps\hyp.optimize-single-class.yaml --resume V:\Work\yolov5\runs\train\yolov5s-cav2-km-transfer-learning-st2\weights\last.pt
@REM python train.py --epochs 200 --data V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml --weights models/pretrained/yolov5m.pt --cfg yolov5m-c3k2-ca-c2psa.yaml --cache disk --device 0 --batch-size 8 --workers 4 --name=yolov5m-c3k2-ca-c2psa --resume runs/train/yolov5m-c3k2-ca-c2psa5/weights/last.pt
@REM python train.py --epochs 200 --data V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml --rect --imgsz 960 --weights models/pretrained/yolov5s6.pt --cfg yolov5s6-ca.yaml --cache disk --device 0 --batch-size 16 --workers 4 --name=yolov5s-c3x-c3ca-rect-960 --hyp=data\hyps\hyp.optimize-single-class.yaml


@echo off
REM Run YOLOv5 training on Windows

REM KM
@REM python train.py ^
@REM   --data "V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml" ^
@REM   --weights "V:\Work\yv5-ca-stm\app\runs\train_ca\yolov5_stomata_v3_CA3x\weights\best.pt" ^
@REM   --cfg "yolov5s.yaml" ^
@REM   --img 384 ^
@REM   --batch-size 4 ^
@REM   --epochs 200 ^
@REM   --hyp "data\hyps\hyp.scratch-low.yaml" ^
@REM   --name "km-v5s-base" ^
@REM   --cache disk ^
@REM   --workers 32 ^
@REM   --patience 30 ^
@REM   --device 0 

@REM STOMATA
python train.py ^
  --data "V:\Work\yv5-ca-stm\yolo_format_dataset\stomata\dataset.yaml" ^
  --weights "models\pretrained\yolov5s.pt" ^
  --cfg "yolov5s-ca1x.yaml" ^
  --img 640 ^
  --batch-size 16 ^
  --epochs 200 ^
  --hyp "data\hyps\hyp.ca.yaml" ^
  --name "sto-v5s-ca1x-640" ^
  --cache disk ^
  --workers 4 ^
  --patience 30 ^
  --device 0 

pause


