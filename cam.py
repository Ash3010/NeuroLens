import torch
import numpy as np
import cv2
import torch.nn as nn
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import get_internal_model


class YOLOWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs


yolo_model = YOLOWrapper(get_internal_model())
target_layers = [yolo_model.model.model[-2]]
cam = EigenCAM(model=yolo_model, target_layers=target_layers)


def preprocess_image(img):
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def run_cam(img):
    rgb_img = cv2.resize(img, (640, 640))
    input_tensor = preprocess_image(img)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(
        rgb_img.astype(np.float32) / 255.0,
        grayscale_cam,
        use_rgb=True
    )
    return visualization
