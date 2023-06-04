from ultralytics import YOLO
from toolkits.model_loader import load_spiga_framework
from toolkits.utils import face_analysis
import cv2

# import torch
import numpy as np
from datetime import datetime
import time
import json

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


def run_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)

    # 初始化所有模型
    yolo_model = YOLO('best.pt')
    device = 'cpu'
    half = device != 'cpu'
    results = yolo_model(cv2.imread('bus.jpg'))


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
    phase0 = 0
    phase1 = 0
    phase2 = 0
    phase3 = 0
    phase4 = 0
    phase5 = 0
    m0 = 0
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    m5 = 0


    ANGLE_THRESHOLD = 35
    EAR_THRESHOLD = 0.2
    YAWN_THRESHOLD = 0.4

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
        phase0 = time.time()

        if cnt >= frames:
            break
        phone_around_face = False
        is_eyes_closed = False
        is_turning_head = False
        is_yawning = False
        overlap = 0
        cnt += 1
        ret, frame = cap.read()
        if cnt % 10 != 0:
            continue
        if cnt + 80 > frames:  # 最后三秒不判断了
            break

        phase1 = time.time()
        m0 += (phase1 - phase0)

        print(f'video {cnt}/{frames} {save_path}')  # delete
        # process the image with the yolo and get the person list
        results = yolo_model(frame)


        phase2 = time.time()
        m1 += (phase2 -phase1)

        # img1,bbox = process_results(results, frame, sensitivity)

        # 创建img0的副本
        img1 = frame.copy()

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
            # img1 = img1[y1:y2, x1:x2]
            m1, n1, m2, n2 = x1, y1, x2, y2

            # 计算交集的面积
            if rightmost_phone is not None and rightmost_box is not None:
                # 计算两个框的IoU
                overlap = iou(rightmost_box, rightmost_phone)

                # cv2.putText(img1, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(overlap)
                # 如果IoU大于阈值，打印警告
                if overlap > sensitivity:
                    phone_around_face = True  ##判断手机

        # 计算边界框的宽度和高度
        w = m2 - m1
        h = n2 - n1

        # 创建 bbox
        bbox = [m1, n1, w, h]

        phase3 = time.time()
        m2 += (phase3 - phase2)

        # frame = spig_process_frame(frame, bbox)  # TODO: 删除
        if phone_around_face == False:
            pose, mar, ear = face_analysis(frame, bbox,test=1)  # TODO: 添加功能


            is_turning_head = True if np.abs(pose[[0, 2]]).max() > ANGLE_THRESHOLD else False
            is_yawning = True if mar > YAWN_THRESHOLD else False
            is_eyes_closed = True if ear < EAR_THRESHOLD else False
            # is_eyes_closed, is_turning_head, is_yawning = face_analysis(frame, bbox) # TODO: 添加功能
        # is_eyes_closed, is_turning_head, is_yawning = False, False, False

        phase4 = time.time()
        m3 += (phase4 -phase3)


        if is_eyes_closed:
            eyes_closed_frame += 1
            if eyes_closed_frame > max_eyes:
                max_eyes = eyes_closed_frame
        else:
            eyes_closed_frame = 0

        if is_turning_head:
            look_around_frame += 1
            if look_around_frame > max_wandering:
                max_wandering = look_around_frame
        else:
            look_around_frame = 0

        if is_yawning:
            mouth_open_frame += 1
            eyes_closed_frame = 0    #有打哈欠则把闭眼和转头置零
            look_around_frame = 0
            if mouth_open_frame > max_mouth:
                max_mouth = mouth_open_frame
        else:
            mouth_open_frame = 0

        if phone_around_face:
            use_phone_frame += 1
            mouth_open_frame = 0
            look_around_frame = 0
            eyes_closed_frame = 0  #有手机则把其他都置零
            if use_phone_frame > max_phone:
                max_phone = use_phone_frame
        else:
            use_phone_frame = 0

            ###################################################
            # im0 = display_results(img0, det, names,is_eyes_closed, is_turning_head, is_yawning)
            # # write video
            # vid_writer.write(im0)

        if max_phone >= 7:
            result['result']['category'] = 3
            break

        elif max_wandering >= 9:
            result['result']['category'] = 4
            break

        elif max_mouth >= 7:
            result['result']['category'] = 2
            break

        elif max_eyes >= 7:
            result['result']['category'] = 1
            break
        phase5 = time.time()
        m4 += (phase5 -phase4)
        # continue_loop = output_module(img1)
        # vid_writer.release()
    final_time = time.time()
    duration = int(np.round((final_time - now) * 1000))

    cap.release()
    print(f'{video_path} finish, save to {save_path}')  # delete
    print(f'phase0-1={m0}')
    print(f'phase1-2={m1}')
    print(f'phase2-3={m2}')
    print(f'phase3-4={m3}')
    print(f'phase4-0={m4}')


    result['result']['duration'] = duration
    # print(result) #delete
    return result


def main():
    # video_dir = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\vedio'
    video_dir = r'F:\ccp1\closeandlawn'
    save_dir = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\output'

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]

    # Create a new log file with the current time
    log_file = os.path.join(save_dir, datetime.now().strftime("%Y%m%d%H%M%S") + '_log.json')
    log = {}

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        save_path = os.path.join(save_dir, video_file)

        result = run_video(video_path, save_path)
        json_save_path = save_path.rsplit('.', 1)[0] + '.json'

        with open(json_save_path, 'w') as json_file:
            json.dump(result, json_file)

        # Update the log and write it to the log file
        log[video_file] = result
        with open(log_file, 'w') as log_json:
            json.dump(log, log_json)


if __name__ == '__main__':
    import os

    # video_path = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\vedio\day_man_001_30_2.mp4'
    # video_path = './day_man_002_20_1.mp4'
    # save_path = './log'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # yolo_detection()
    # yolo_spig_cap_processing()
    # print(run_video(video_path,save_path))
    main()