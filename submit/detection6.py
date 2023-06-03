from ultralytics import YOLO
from spiga_model_processing import spig_process_frame
import cv2

import torch
import numpy as np
from datetime import datetime
import time

# yolo_model = YOLO('best.pt')
# device = 'cpu'
# half = device != 'cpu'
# from utils.augmentations import letterbox

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xyxy2xywh(bbox):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return x0, y0, w, h


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # calculate the overlap coordinates
    xx1, yy1 = max(x1, x3), max(y1, y3)
    xx2, yy2 = min(x2, x4), min(y2, y4)

    # compute the width and height of the overlap area
    overlap_width, overlap_height = max(0, xx2 - xx1), max(0, yy2 - yy1)
    overlap_area = overlap_width * overlap_height

    # compute the area of both bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    # compute IoU
    iou = overlap_area / float(box1_area + box2_area - overlap_area)
    return iou







def run_video(video_path,save_path):
    cap = cv2.VideoCapture(video_path)

    yolo_model = YOLO('best.pt')
    device = 'cpu'
    half = device != 'cpu'

    cnt = 0  # 开始处理的帧数
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 待处理的总帧数

    # 获取当前视频的帧率与宽高，设置同样的格式，以确保相同帧率与宽高的视频输出
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # save_path += os.path.basename(video_path)
    # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    ######################################################################################################
    result = {"result": {"category": 0, "duration": 6000}}
    eyes_closed_frame = 0
    mouth_open_frame = 0
    use_phone_frame = 0
    max_phone = 0
    max_eyes = 0
    max_mouth = 0
    max_wandering = 0
    look_around_frame = 0

    sensitivity = 0.001

    now = time.time()  # 读取视频与加载模型的时间不被计时（？）

    while cap.isOpened():
        cnt += 1
        if cnt % 10 != 0:
            continue

        ret, frame = cap.read()
        print(f'video {frame}/{frames} {save_path}')  #delete
        # process the image with the yolo and get the person list
        results = yolo_model(frame)


        img1 = frame.copy()

        for result in results:
            boxes = result.boxes

        # process the boxes and get the person_list
        person_list = []
        for bbox in boxes.data:
            if int(bbox[-1]) == 0 and bbox[-2] > 0.5:
                bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                person_list.append(xyxy2xywh(bbox))
            else:
                continue

        # process the image with the spig_model
        for bbox in person_list:
            frame = spig_process_frame(frame, bbox)
        continue_loop = output_module(frame)

        # continue_loop = output_module(img1)
        if not continue_loop:
            break

if __name__ == '__main__':
    import os
    video_path = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\vedio\day_man_001_30_2.mp4'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # yolo_detection()
    # yolo_spig_cap_processing()
    run_video(video_path)