from ultralytics import YOLO

model = YOLO("yolov5s.pt")

def get_model():
    return model

def get_internal_model():
    return model.model
