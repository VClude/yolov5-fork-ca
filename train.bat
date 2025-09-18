@REM python train.py --epochs 1000 --data V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml --weights V:\Work\yv5-ca-stm\app\runs\train_ca\yolov5_stomata_v3_CA3x\weights\best.pt --cfg yolov5s-cav2.yaml --cache disk --device 0 --batch-size 8 --workers 4 --name=yolov5s-cav2-km-transfer-learning-st --hyp=data\hyps\hyp.optimize-single-class.yaml --resume V:\Work\yolov5\runs\train\yolov5s-cav2-km-transfer-learning-st2\weights\last.pt
@REM python train.py --epochs 200 --data V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml --weights models/pretrained/yolov5m.pt --cfg yolov5m-c3k2-ca-c2psa.yaml --cache disk --device 0 --batch-size 8 --workers 4 --name=yolov5m-c3k2-ca-c2psa --resume runs/train/yolov5m-c3k2-ca-c2psa5/weights/last.pt
@REM python train.py --epochs 200 --data V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml --rect --imgsz 960 --weights models/pretrained/yolov5s6.pt --cfg yolov5s6-ca.yaml --cache disk --device 0 --batch-size 16 --workers 4 --name=yolov5s-c3x-c3ca-rect-960 --hyp=data\hyps\hyp.optimize-single-class.yaml


@echo off
REM Run YOLOv5 training on Windows

python train.py ^
  --data "V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml" ^
  --weights "V:\Work\yv5-ca-stm\app\runs\train_ca\yolov5_stomata_v3_CA3x\weights\best.pt" ^
  --cfg "yolov5s-cav2.yaml" ^
  --img 1280 ^
  --batch-size 8 ^
  --epochs 150 ^
  --hyp "data\hyps\hyp.micro.yaml" ^
  --project "micro-km" ^
  --name "s6-1280" ^
  --cache disk ^
  --workers 4 ^
  --patience 30 ^
  --device 0

pause


