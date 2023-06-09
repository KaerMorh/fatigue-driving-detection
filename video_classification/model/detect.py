import torch

import numpy as np

from utils1.datasets import letterbox
from utils1.general import non_max_suppression, scale_coords, xyxy2xywh


def detect(model, frame, stride, imgsz):

    result = []
    # dataset = LoadImages(frame, img_size=imgsz, stride=stride)
    img = letterbox(frame, imgsz, stride=stride)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)

    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]

    pred = non_max_suppression(pred, 0.5, 0.45,  agnostic=False)

    for i, det in enumerate(pred):
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                result.append([int(cls), xywh])

    return result
