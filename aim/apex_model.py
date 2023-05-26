import torch
from models.experimental import attempt_load

device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'

# weights = r'D:\0---Program\Projects\aimbot\yolov5-6.0\aim\weights\apex.pt'
weights = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\aim\weights\face_phone_detection.pt'
imgsz = 640


def load_model():
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16

    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    return model