from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8m.yaml')  # build a new model from YAML
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training) (loads a model pretrained on coco)
# current_best = 'path_to_current_best.pt'
# model = YOLO('yolov8m.yaml').load(f'{current_best}')  # build from YAML and transfer weights

## Load a model
# model = YOLO('path/to/last.pt')  # load a partially trained model

# Train the model
path_to_data = "D:\\omer\\human_detection\\data.yaml"
model.train(data=path_to_data, epochs=200, imgsz=640, batch=24, device='cuda:0', workers=0, project='head_count', name='head_detection_v8m')
## Resume training
# model.train(resume=True)