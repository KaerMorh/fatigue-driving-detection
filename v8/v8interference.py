from ultralytics import YOLO
from spiga_model_processing import spig_process_frame
import cv2
from grabscreen import grab_screen
import torch
import numpy as np


yolo_model = YOLO('best.pt')
side_model = YOLO(r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\submit\side.pt')
device = 'cpu'
half = device != 'cpu'
# from utils.augmentations import letterbox
import time
# from submit.toolkits.utils import face_analysis
# from ultralytics import YOLO
from submit.toolkits.analysis import face_analysis
from submit.toolkits.inference import spiga_frame_inference
from submit.toolkits.arithmetic_3d import Expert_3D
import cv2
import os

import numpy as np
from datetime import datetime
import time
import json

expert_3d = Expert_3D()

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


def yolo_inference(frame):
    results = yolo_model(frame, stream=True)

    for result in results:
        boxes = result.boxes

    # process the boxes and get the person_list
    person_list = []
    for bbox in boxes.data:
        if bbox[-2] > 0.5:
            bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            person_list.append(xyxy2xywh(bbox))
        else:
            continue

    return person_list

def input_module():
    img0 = grab_screen(region=(0, 0, 1920, 1080))
    img0 = cv2.resize(img0, (960, 540))
    if img0 is None:
        print('Resize failed!')
        return

    img = letterbox(img0, 640)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]
    return img, img0




def output_module(im0):
    print(im0.shape)
    # if torch.is_tensor(im0):
    #     # Convert to numpy array and transpose dimensions
    #     im0 = im0.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    #     im0 = (im0 * 255).astype(np.uint8)
    cv2.namedWindow('apex')
    # cv2.resizeWindow('apex', 1920 // 2, 1080 // 2)
    cv2.imshow('apex', im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False
    return True

def yolo_spig_cap_processing():


    while True:
        img, frame = input_module()
        # ret, frame = cap.read()

        # process the image with the yolo and get the person list
        results = yolo_model(frame)

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

def process_results(results, img0, sensitivity):
    # 创建img0的副本
    img1 = img0.copy()

    # 获取图像的宽度
    img_height = img0.shape[0]

    img_width = img0.shape[1]

    # 获取所有的边界框
    boxes = results[0].boxes

    # 获取所有类别
    classes = boxes.cls

    # 获取所有的置信度
    confidences = boxes.conf

    # 初始化最靠右的框和最靠右的手机
    rightmost_box = None
    rightmost_phone = None

    # 遍历所有的边界框
    for box, cls, conf in zip(boxes.xyxy, classes, confidences):
        # 如果类别为1（手机）且在图片的右2/3区域内
        if cls == 1 and box[0] > img_width * 1/ 3:
            # 如果还没有找到最靠右的手机或者这个手机更靠右
            if rightmost_phone is None or box[0] > rightmost_phone[0]:
                rightmost_phone = box

        # 如果类别为0（驾驶员）且在图片的右2/3区域内
        if cls == 0 and box[0] > img_width * 1/ 3:
            # 如果还没有找到最靠右的框或者这个框更靠右
            if rightmost_box is None or box[0] > rightmost_box[0]:
                rightmost_box = box

    # 如果没有找到有效的检测框，返回img1为img0的右3/5区域
    if rightmost_box is None:
        img1 = img1
        m1 = int(img_width * 2 / 5)
        n1 = 0
        # 右下角的坐标
        m2 = img_width
        n2 = img_height

    # 否则，返回img1仅拥有最靠右的框内的图片
    else:
        x1, y1, x2, y2 = rightmost_box
        x1 = max(0, int(x1 - 0.1 * (x2 - x1)))
        y1 = max(0, int(y1 - 0.1 * (y2 - y1)))
        x2 = min(img_width, int(x2 + 0.1 * (x2 - x1)))
        y2 = min(img_width, int(y2 + 0.1 * (y2 - y1)))
        img1 = img1[y1:y2, x1:x2]
        m1, n1, m2, n2 = rightmost_box

    # 如果找到了手机，检查它是否与驾驶员的框有重叠
    if rightmost_phone is not None and rightmost_box is not None:
        # 计算两个框的交集
        x1 = max(rightmost_box[0], rightmost_phone[0])
        y1 = max(rightmost_box[1], rightmost_phone[1])
        x2 = min(rightmost_box[2], rightmost_phone[2])
        y2 = min(rightmost_box[3], rightmost_phone[3])

        # 计算交集的面积
        if rightmost_phone is not None and rightmost_box is not None:
            # 计算两个框的IoU
            overlap = iou(rightmost_box, rightmost_phone)

            cv2.putText(img1, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 如果IoU大于阈值，打印警告
            if overlap > sensitivity:
                print("Warning: phone and driver overlap!")



    # 计算边界框的宽度和高度
    w = m2 - m1
    h = n2 - n1

    # 创建 bbox
    bbox = [m1, n1, w, h]


    return img1,bbox

# def process_results(results, img0, sensitivity):
#     img1 = img0.copy()
#     w, h = img1.shape[1], img1.shape[0]
#
#     right_region = int(w / 3)
#     det_cls1 = [box for *box, conf, cls in det if cls == 0]
#     det_cls1_right = [box for box in det_cls1 if box[2] > right_region]
#
#     det_cls2_right = [box for *box, conf, cls in det if cls == 1]
#     # det_cls2_right = [box for box in det_cls2 if box[2] > right_region]
#
#     if det_cls1_right:
#         # Find the box most to the right
#         box_most_right = max(det_cls1_right, key=lambda box: box[2])
#         x1, y1, x2, y2 = map(int, box_most_right)
#
#         # check if a phone is also detected
#         if det_cls2_right:
#             # Find the phone box most to the right bottom
#             box_most_right_bottom_phone = max(det_cls2_right, key=lambda box: (box[2], box[3]))
#
#             # check overlap
#             print(iou(box_most_right, box_most_right_bottom_phone))  # TODO:在一切完工后删除测试项
#             if iou(box_most_right, box_most_right_bottom_phone) > sensitivity:
#                 phone_around_face = True  ##判断手机
#
#         # Enlarge the box by 20%
#         dw, dh = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
#         x1, y1, x2, y2 = max(0, x1 - dw), max(0, y1 - dh), min(w, x2 + dw), min(h, y2 + dh)
#
#         img1 = img1[y1:y2, x1:x2]
#     else:
#         # If no valid box, return the right 3/5 region of img0
#         right_3_5_region = int(2 * w / 5)
#         img1 = img1[:, right_3_5_region:]
#
#     return img1


def run_video(video_path,save_path,mode,save_mode = 0):

    overlap = None
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
    if mode == 1:
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cnt < frames -1:
        phone_around_face = False
        overlap = 0
        if mode == 1:
            cnt += 1
            ret, frame = cap.read()
            if cnt % 21 != 0 and cnt != 1:
                continue

        if mode == 0:
            img, frame = input_module()

        # process the image with the yolo and get the person list
        results = yolo_model(frame)
        side_results = side_model(frame)

        # img1,bbox = process_results(results, frame, sensitivity)

        # 创建img0的副本
        img1 = frame.copy()
        print(f'video {cnt}/{frames} {save_path}')

        # 获取图像的宽度
        img_height = frame.shape[0]

        img_width = frame.shape[1]

        # 获取所有的边界框
        boxes = results[0].boxes

        # 获取所有类别
        classes = boxes.cls

        # 获取所有的置信度
        confidences = boxes.conf

        # 初始化最靠右的框和最靠右的手机
        rightmost_box = None
        rightmost_phone = None

        # 遍历所有的边界框
        for box, cls, conf in zip(boxes.xyxy, classes, confidences):
            # 如果类别为1（手机）且在图片的右2/3区域内
            if cls == 1 and box[0] > img_width * 1 / 3:
                # 如果还没有找到最靠右的手机或者这个手机更靠右
                if rightmost_phone is None or box[0] > rightmost_phone[0]:
                    rightmost_phone = box

            # 如果类别为0（驾驶员）且在图片的右2/3区域内
            if cls == 0 and box[0] > img_width * 1 / 3:
                # 如果还没有找到最靠右的框或者这个框更靠右
                if rightmost_box is None or box[0] > rightmost_box[0]:
                    rightmost_box = box

        # 如果没有找到有效的检测框，返回img1为img0的右3/5区域
        if rightmost_box is None:
            img1 = img1
            m1 = int(img_width * 2 / 5)
            n1 = 0
            # 右下角的坐标
            m2 = img_width
            n2 = img_height

        # 否则，返回img1仅拥有最靠右的框内的图片
        else:
            x1, y1, x2, y2 = rightmost_box
            x1 = max(0, int(x1 - 0.1 * (x2 - x1)))
            y1 = max(0, int(y1 - 0.1 * (y2 - y1)))
            x2 = min(img_width, int(x2 + 0.1 * (x2 - x1)))
            y2 = min(img_width, int(y2 + 0.1 * (y2 - y1)))
            img1 = img1[y1:y2, x1:x2]
            m1, n1, m2, n2 = x1, y1, x2, y2


            # 计算交集的面积
            if rightmost_phone is not None and rightmost_box is not None:
                # 计算两个框的IoU
                overlap = iou(rightmost_box, rightmost_phone)

                cv2.putText(img1, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 如果IoU大于阈值，打印警告
                if overlap > sensitivity:
                    print("Warning: phone and driver overlap!")



        # 计算边界框的宽度和高度
        w = m2 - m1
        h = n2 - n1

        # 创建 bbox
        bbox = [m1, n1, w, h]




        ANGLE_THRESHOLD = 30
        EAR_THRESHOLD = 0.2
        YAWN_THRESHOLD = 0.4

        pose, mar, ear = 0, 0,0
        landmark_entropy , trl_entropy,aar = 0,0,0
        landmark_entropy, trl_entropy, mar, ear, aar = face_analysis(frame, bbox, get_ratio=True)  # TODO: 添加功能
        # 指标对应的 threshold
        threshold = [i[0] for i in expert_3d.judger]
        landmarks_ts, trl_ts, YAWN_THRESHOLD, EAR_THRESHOLD, aar_ts = threshold
        mar > YAWN_THRESHOLD
        ear < EAR_THRESHOLD

        # frame = spig_process_frame(frame, bbox)
        # cv2.putText(frame, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #
        # cv2.putText(frame, f"Pose max: {np.abs(pose[[0, 2]]).max():.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(frame, f"Yaw: {np.abs(pose[0]):.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255, 0), 2)
        cv2.putText(frame, f"AAR: {aar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if         mar > YAWN_THRESHOLD :
            cv2.putText(frame, f"MAR: {mar}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"MAR: {mar}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if ear < EAR_THRESHOLD:
            cv2.putText(frame, f"EAR: {ear}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"EAR: {ear}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if landmark_entropy > landmarks_ts:
            cv2.putText(frame, f"landmark_entropy: {landmark_entropy}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"landmark_entropy: {landmark_entropy}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if trl_entropy > trl_ts:
            cv2.putText(frame, f"trl_entropy: {trl_entropy}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"trl_entropy: {trl_entropy}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # if overlap:
            cv2.putText(frame, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # continue_loop = output_module(frame)
        res_plotted = side_results[0].plot()
        if mode == 0:
            continue_loop = output_module(res_plotted)

            # continue_loop = output_module(img1)
            if not continue_loop:
                break

        else:



            print(f'video {cnt}/{frames} {save_path}')
            if save_mode == 1:
                cv2.imwrite(os.path.join(save_path, f'frame_{cnt}.jpg'), res_plotted)
            else:
                vid_writer.write(res_plotted)
    if mode == 1:
        vid_writer.release()



import os
import shutil

def process_videos(file_list, video_path, output_path):
    with open(file_list, 'r') as file:
        videos = file.read().splitlines()

    # Create a check directory under output path
    check_path = os.path.join(output_path, 'check')
    if not os.path.exists(check_path):
        os.makedirs(check_path)

    for video in videos:
        current_video_path = os.path.join(video_path, video)
        current_output_path = os.path.join(output_path, os.path.splitext(video)[0])

        # Create output directory if it does not exist
        if not os.path.exists(current_output_path):
            os.makedirs(current_output_path)

        # Copy the video to the check directory
        shutil.copy(current_video_path, check_path)

        run_video(current_video_path, current_output_path, mode=1, save_mode=1)


if __name__ == '__main__':
    # import os
    # # video_path = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\vedio\night_man_001_40_3.mp4'
    # video_path = r'F:\ccp1\la\test\night_woman_005_41_7.mp4'
    # save_path = r'F:\ccp1\la\test'
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # # yolo_detection()
    # # yolo_spig_cap_processing()
    # run_video(video_path,save_path,0,save_mode=0)
    file_list = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\file_list.txt'
    video_path = r'F:\ChallengeCup'
    output_path = r'F:\ccp1\interference'

    process_videos(file_list, video_path, output_path)

