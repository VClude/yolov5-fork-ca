python train.py --epochs 200 --data V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml --weights models/pretrained/resnet18.pt --cfg yolov5s-cav2.yaml --cache disk --device 0 --batch-size 32 --workers 4 --name=resnet-18-cav2 --hyp=data\hyps\hyp.ca.yaml
@REM python train.py --epochs 200 --data V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml --weights models/pretrained/yolov5m.pt --cfg yolov5m-c3k2-ca-c2psa.yaml --cache disk --device 0 --batch-size 8 --workers 4 --name=yolov5m-c3k2-ca-c2psa --resume runs/train/yolov5m-c3k2-ca-c2psa5/weights/last.pt
@REM python train.py --epochs 200 --data V:\AIModel\KelenjarMinyak\Dataset\YOLODataset\dataset.yaml --rect --imgsz 960 --weights models/pretrained/yolov5s6.pt --cfg yolov5s6-ca.yaml --cache disk --device 0 --batch-size 16 --workers 4 --name=yolov5s-c3x-c3ca-rect-960 --hyp=data\hyps\hyp.optimize-single-class.yaml


