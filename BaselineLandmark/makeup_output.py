import torch
import cv2
import numpy as np
import copy
from grabscreen import grab_screen
# yolo_model = YOLO('best.pt')
# side_model = YOLO(r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\submit\side.pt')
device = 'cpu'
half = device != 'cpu'
from tracker import Tracker
from EAR import eye_aspect_ratio
from ultralytics import YOLO
from MAR import mouth_aspect_ratio


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

def run_video(video_path,save_path ):
    cap = cv2.VideoCapture(video_path)
    eyes_model = YOLO('eyev8best.pt')
    # eyes_model = YOLO('eye_ce_facebest.pt')
    tracker = Tracker(1920, 1080, threshold=None, max_threads=4, max_faces=4,
                      discard_after=10, scan_every=3, silent=True, model_type=3,
                      model_dir=None, no_gaze=False, detection_threshold=0.6,
                      use_retinaface=0, max_feature_updates=900,
                      static_model=True, try_hard=False)

    yolo_model = YOLO('best.pt')
    sensitivity = 0.01
    side_model = YOLO('120best.pt')

    standard_pose = [180, 40, 80]
    lStart = 42
    lEnd = 48
    # (rStart, rEnd) = (36, 42)
    rStart = 36
    rEnd = 42
    # (mStart, mEnd) = (49, 66)
    mStart = 49
    mEnd = 66
    EAR_THRESHOLD = 0.1
    YAWN_THRESHOLD = 0.6
    face_num = 0
    pose1, pose2, pose3 = 0, 0, 0
    cnt = 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cnt < frames -1:
        overlap = 0
        cnt += 1
        ret, img0 = cap.read()
        if cnt % 10 != 0:
            continue
        # img, img0 = input_module()
        frame = img0.copy()
        print(f'video {cnt}/{frames} {save_path}')
        # img0 = img0[:, 600:1920, :]
        # width = img0.shape[1]
        # start_col = int(width * 8 / 30)
        # img0 = img0[:, start_col:, :]

        # frame = img0[:, 640:, :] # 获取右边的图像
        yolo1 = True

        # 获取图像的宽度
        results = yolo_model(frame)
        img_height = frame.shape[0]

        img_width = frame.shape[1]
        x1, y1, x2, y2 = 0, 0, img_width, img_height
        # 获取所有的边界框
        boxes = results[0].boxes

        # 获取所有类别
        classes = boxes.cls

        # 获取所有的置信度
        confidences = boxes.conf
        overlap = None
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

        img1 = frame.copy()

        # 如果没有找到有效的检测框，返回img1为img0的右3/5区域
        if rightmost_box is None:
            yolo1 = False
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
            y2 = min(img_width, int(y2 + 0.1 * (y2 - y1)))  # 把框扩大10%
            # img1 = img1[y1:y2, x1:x2]
            m1, n1, m2, n2 = x1, y1, x2, y2
            # x1, y1, x2, y2 = rightmost_box
            # #将图片的增大百分之20
            # dw, dh = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
            # x1, y1, x2, y2 = max(0, x1 - dw), max(0, y1 - dh), min(w, x2 + dw), min(h, y2 + dh)
            img1 = img1[y1:y2, x1:x2]
            # m1, n1, m2, n2 = x1, y1, x2, y2

            # 计算交集的面积
            if rightmost_phone is not None and rightmost_box is not None:
                # 计算两个框的IoU
                overlap = iou(rightmost_box, rightmost_phone)

                # cv2.putText(img1, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f'overlap:{overlap}')
                # 如果IoU大于阈值，打印警告
                if overlap > sensitivity:
                    phone_around_face = True  ##判断手机

        # 计算边界框的宽度和高度
        w = m2 - m1
        h = n2 - n1

        # 创建 bbox
        bbox = [m1, n1, w, h]

        # img0 = frame[y1:y2, x1:x2]

        side_results = side_model(frame)
        side_boxes = side_results[0].boxes
        side_classes = side_boxes.cls
        # 获取所有的置信度
        side_confidences = side_boxes.conf
        rightmost_box = None

        rightmost_cls = None

        # 遍历所有的边界框
        for box, cls, conf in zip(side_boxes.xyxy, side_classes, side_confidences):
            if box[0] > img_width * 3/9:
                if rightmost_box is None or box[0] > rightmost_box[0]:
                    rightmost_box = box
                    rightmost_cls = cls
            # if rightmost_box is None or box[0] > rightmost_box[0]:
            #     rightmost_box = box
            #     rightmost_cls = cls

        if rightmost_box is None:
            is_turning_head = None
        elif rightmost_cls != 0:
            is_turning_head = True
        else:
            is_turning_head = False

        # img1 = img0[:, 600:1920, :]
        faces = tracker.predict(frame)
        eye_results = eyes_model(img0)
        if len(faces) > 0:
            face_num = None
            max_x = 0
            for face_num_index, f in enumerate(faces):
                if max_x <= f.bbox[3] or face_num is None:
                    face_num = face_num_index
                    max_x = f.bbox[3]
            if face_num is not None:
                f = faces[face_num]
                f = copy.copy(f)

                # 检测是否转头
                # if np.abs(standard_pose[0] - f.euler[0]) >= 45 or np.abs(standard_pose[1] - f.euler[1]) >= 45 or \
                #         np.abs(standard_pose[2] - f.euler[2]) >= 45:
                #     is_turning_head = True
                # else:
                #     is_turning_head = False
                pose1 = np.abs(standard_pose[0] - f.euler[0])
                pose2 = np.abs(standard_pose[1] - f.euler[1])
                pose3 = np.abs(standard_pose[2] - f.euler[2])

                # 检测是否闭眼
                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = f.lms[lStart:lEnd]
                rightEye = f.lms[rStart:rEnd]
                L_E = eye_aspect_ratio(leftEye)
                R_E = eye_aspect_ratio(rightEye)
                # average the eye aspect ratio together for both eyes
                ear = (L_E + R_E) / 2.0

                if ear < EAR_THRESHOLD:
                    is_eyes_closed = True
                    print(f'ear:{ear}')
                    # print(EYE_AR_THRESH)
                else:
                    is_eyes_closed = False
                # print(ear, eyes_closed_frame)

                # 检测是否张嘴
                mar = mouth_aspect_ratio(f.lms)

                if mar > YAWN_THRESHOLD:
                    is_yawning = True
                    print(f'mar:{mar}')
                # print(MOUTH_AR_THRESH)
                #                         print(len(f.lms), f.euler)
                img0 = results[0].plot()
                # cv2.putText(frame, f"AAR: {aar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if mar > YAWN_THRESHOLD:
                    cv2.putText(img0, f"MAR: {mar}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(img0, f"MAR: {mar}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if ear < EAR_THRESHOLD:
                    cv2.putText(img0, f"EAR: {ear}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(img0, f"EAR: {ear}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if L_E < EAR_THRESHOLD:
                    cv2.putText(img0, f"L_E:{L_E}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(img0, f"L_E:{L_E}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if R_E < EAR_THRESHOLD:
                    cv2.putText(img0, f"R_E:{R_E}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(img0, f"R_E:{R_E}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if overlap:  # if overlap:
                    cv2.putText(img0, f"IoU: {overlap:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(img0, f"pose1: {pose1:.2f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img0, f"pose2: {pose2:.2f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img0, f"pose3: {pose3:.2f}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img0, f"cnt: {cnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 254, 37), 2)

                # img0 = eye_results[0].plot()
                # img0 = results[0].plot()

        cv2.putText(img0, f"yolo1: {yolo1}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 254, 37), 2)
        cv2.putText(img0, f"side: {is_turning_head}", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 254, 37), 2)
        cv2.imwrite(os.path.join(save_path, f'frame_{cnt}.jpg'), img0)

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
        # 如果最后三个字符不是mp4，就去掉最后一个字符
        if video[-3:] != 'mp4':
            video = video[:-1]
        current_video_path = os.path.join(video_path, video)
        current_output_path = os.path.join(output_path, os.path.splitext(video)[0])

        # Create output directory if it does not exist
        if not os.path.exists(current_output_path):
            os.makedirs(current_output_path)

        # Copy the video to the check directory
        shutil.copy(current_video_path, check_path)
        run_video(current_video_path, current_output_path)


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    file_list = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\file_list.txt'
    video_path = r'F:\ChallengeCup'
    output_path = r'F:\ccp1\interference'

    process_videos(file_list, video_path, output_path)