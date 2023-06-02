from ultralytics import YOLO
from spiga_model_processing import spig_process_frame
import cv2
from grabscreen import grab_screen
import torch
import numpy as np

yolo_model = YOLO('best.pt')
device = 'cpu'
half = device != 'cpu'
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



if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # yolo_detection()
    yolo_spig_cap_processing()
