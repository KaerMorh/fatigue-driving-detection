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

def main():
    tracker = Tracker(960, 540, threshold=None, max_threads=4, max_faces=4,
                      discard_after=10, scan_every=3, silent=True, model_type=3,
                      model_dir=None, no_gaze=False, detection_threshold=0.6,
                      use_retinaface=0, max_feature_updates=900,
                      static_model=True, try_hard=False)

    yolo_model = YOLO('best.pt')
    side_model = YOLO('120best.pt')

    while True:
        img, img0= input_module()
        frame = img0.copy()
        # frame = frame[:, 600:1920, :]




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


        faces = tracker.predict(frame)
        if len(faces) > 0:
            face_num = 0
            max_x = 0
            for face_num_index, f in enumerate(faces):
                if max_x <= f.bbox[3]:
                    face_num = face_num_index
                    max_x = f.bbox[3]
            # if face_num != 0:
                f = faces[face_num]
                f = copy.copy(f)

                # 检测是否转头
                if np.abs(standard_pose[0] - f.euler[0]) >= 45 or np.abs(standard_pose[1] - f.euler[1]) >= 45 or \
                        np.abs(standard_pose[2] - f.euler[2]) >= 45:
                    is_turning_head = True
                else:
                    is_turning_head = False

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
                    print(ear)
                    # print(EYE_AR_THRESH)
                else:
                    is_eyes_closed = False
                # print(ear, eyes_closed_frame)

                # 检测是否张嘴
                mar = mouth_aspect_ratio(f.lms)

                if mar > YAWN_THRESHOLD:
                    is_yawning = True
                print(mar)
                # print(MOUTH_AR_THRESH)
                #                         print(len(f.lms), f.euler)

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

                    # if overlap:
                    # cv2.putText(frame, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if not output_module(img0):
            break

    return

if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    main()