#from aimtools.config import *
import numpy as np

from models.experimental import attempt_load
from utils.torch_utils import load_classifier, select_device, time_sync
from screen_inf import grab_screen_win32
# import cupy
import cv2
import torch
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox



weights = 'weights/apex1.pt'
imgsz = 640
x, y = (1920, 1080)
re_x, re_y = (1920, 1080)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'
#non_max_suppression
conf_thres = 0.4
iou_thres = 0.5
aims = []


def get_model():
    # select GPU as device
    # device = select_device('')
    # half = device.type != 'cpu' # half = True as we use cuda
    model = attempt_load(weights, map_location=device)
    # stride = int(model.stride.max())  # model stride
    # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half: # half is true anyway
        model.half()  # to FP16 半精度处理

    if device != 'cpu' :#cuda initialize
        model(torch.zeros(1, 3, imgsz).to(device).type_as(next(model.parameters())))[0]

    return model



model = get_model()
stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

while True:
    img0 = grab_screen_win32(region=(0, 0, x, y))
    img0 = cv2.resize(img0, (re_x, re_y))

    img = letterbox(img0, imgsz, stride=stride)[0]

    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img = img / 255.
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    pred = model(img, augment=False, visualize=False)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic = False)
    # print(pred)

    for i, det in enumerate(pred):  # per image
        s=''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh)  # label format
            aim = ('%g ' * len(line)).rstrip() % line
            aim = aim.split(' ')
            aims.append(aim)

    if len(aims):
        for i,det in enumerate(aims):
            _,x_center, y_center, width, hight = det
            x_center, width= re_x * float(x_center), re_x * float(width)
            y_center, hight = re_y * float(y_center), re_y *float(hight)
            top_left = (int(x_center - width/2.), int(y_center - hight /2.))
            bottom_right = int(int(x_center + width/2.), int(y_center + hight /2.))
            color = (0, 255, 0)  #bbox color
            cv2.rectangle(img0, top_left, bottom_right, color, thickness = 3)




    cv2.namedWindow('Apex Detection', cv2.WINDOW_NORMAL)  # 新建一个窗口，方便之后resize，并根据窗口大小调节图片大小（NORRMAL）
    cv2.resizeWindow('Apex Detection', re_x//3, re_y//3)
    # img = np.array(img)
    # cv2.imshow('Apex Detection', img)

