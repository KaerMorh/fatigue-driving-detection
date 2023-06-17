from detection5 import face_analysis,infer_image,iou
from main2 import grab_screen

import dlib

import torch
import numpy as np

from utils.augmentations import letterbox
from v5.models.experimental import attempt_load

from utils.general import (check_img_size, cv2)
from utils.plots import Annotator, colors


def display_results(img0, det, names,is_eyes_closed, is_turning_head, is_yawning):
    # plot label
    annotator = Annotator(img0.copy(), line_width=3, example=str(names))
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        # label = f'{names[c]} {conf:.2f}'
        label = f'{names[c]} {conf:.2f}\n'
        label += f'closed: {is_eyes_closed}\n'
        label += f'head: {is_turning_head}\n'
        label += f'awning: {is_yawning}'
        annotator.box_label(xyxy, label, color=colors(c, True))


    return annotator.result()

def output_module(im0):


    cv2.namedWindow('apex', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('apex', 1920 // 2, 1080 // 2)
    cv2.imshow('apex', im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False
    return True


def input_module():
    img0 = grab_screen(region=(0, 0, 1920, 1080))
    img0 = cv2.resize(img0, (960, 540))
    img = letterbox(img0, 640, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]
    return img, img0


weights = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\aim\weights\face_phone_detection.pt'
device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'
# 导入模型
imgsz = 640
model = attempt_load(weights)
if half:
    model.half()  # to FP16

if device != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))



sensitivity = 0.001

img_size=640
stride=32
augment=False
visualize=False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

img_size = check_img_size(img_size, s=stride)
names = model.names

while True:
    IOUU = -1
    img, img0 = input_module()
    det = infer_image(model, img, img0, augment, visualize)
##################################################
    '''
    输入det与img0，返回img1（），使得img1仅拥有det中图像框（要求是在图片的右2/3区域内，且cls为1，最靠右的一个）内的图片。
    若无有效检测框，则返回img1 = img0 
    img1仅拥有det中图像框（要求是在图片的右2/3区域内，且cls为1，最靠右的一个）内的图片（使img1包含的范围在有效范围内比det大20%）。
    若无有效检测框，则返回img1为img0的右3/5区域。要求全程使用img0的副本以保证img0不被修改
    '''
    img1 = img0.copy()
    w, h = img1.shape[1], img1.shape[0]
    # print('Warning: Phone usage detected!')

    right_region = int( w / 3)
    det_cls1 = [box for *box, conf, cls in det if cls == 0]
    det_cls1_right = [box for box in det_cls1 if box[2] > right_region]

    det_cls2 = [box for *box, conf, cls in det if cls == 1]
    det_cls2_right = [box for box in det_cls2 if box[2] > right_region]

    if det_cls1_right:
        # Find the box most to the right
        box_most_right = max(det_cls1_right, key=lambda box: box[2])
        x1, y1, x2, y2 = map(int, box_most_right)

        # check if a phone is also detected
        if det_cls2_right:
            # Find the phone box most to the right bottom
            box_most_right_bottom_phone = max(det_cls2_right, key=lambda box: (box[2], box[3]))

            # check overlap
            IOUU = iou(box_most_right, box_most_right_bottom_phone)
            if iou(box_most_right, box_most_right_bottom_phone) > sensitivity:

                print('Warning: Phone usage detected!')

        # Enlarge the box by 20%
        dw, dh = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
        x1, y1, x2, y2 = max(0, x1 - dw), max(0, y1 - dh), min(w, x2 + dw), min(h, y2 + dh)

        img1 = img1[y1:y2, x1:x2]
    else:
        # If no valid box, return the right 3/5 region of img0
        right_3_5_region = int(2 * w / 5)
        img1 = img1[:, right_3_5_region:]
    # img1 = img0.copy()
    # w, h = img1.shape[1], img1.shape[0]
    #
    # right_region = int(2 * w / 3)
    # det_cls1 = [box for *box, conf, cls in det if cls == 0]
    # det_cls1_right = [box for box in det_cls1 if box[2] > right_region]
    #
    # if det_cls1_right:
    #     # Find the box most to the right
    #     box_most_right = max(det_cls1_right, key=lambda box: box[2])
    #     x1, y1, x2, y2 = map(int, box_most_right)
    #
    #     # Enlarge the box by 20%
    #     dw, dh = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
    #     x1, y1, x2, y2 = max(0, x1 - dw), max(0, y1 - dh), min(w, x2 + dw), min(h, y2 + dh)
    #
    #     img1 = img1[y1:y2, x1:x2]
    # else:
    #     # If no valid box, return the right 3/5 region of img0
    #     right_3_5_region = int(2 * w / 5)
    #     img1 = img1[:, right_3_5_region:]

    # 进行脸特征检测，仅为测试用的单帧检测，需要做后期修改
    is_eyes_closed, is_turning_head, is_yawning = face_analysis(img0, detector, predictor)

    ###################################################
    im0 = display_results(img0, det, names, is_eyes_closed, is_turning_head, is_yawning)
    continue_loop = output_module(im0)
    # continue_loop = output_module(img1)
    if not continue_loop:
        break

