from v5.grabscreen import grab_screen
from v5.apex_model import load_model
import cv2
import win32gui
import torch
import numpy as np
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox

device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'
imgsz = 640

conf_thres = 0.4
iou_thres = 0.05

x, y = (1920, 1080)
re_x, re_y = (1920, 1080)

model = load_model()
stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names

lock_mode = False


while True:
    while True:
        img0 = grab_screen(region=(0, 0, x, y))
        img0 = cv2.resize(img0, (re_x, re_y))

        img = letterbox(img0, imgsz, stride=stride)[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.

        if len(img.shape) == 3:
            img = img[None] # img = img.unsqueeze(0)

        pred = model(img, augment=False, visualize=False)[0]

        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

        aims = []
        for i, det in enumerate(pred):
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    aim = ('%g ' * len(line)).rstrip() % line
                    aim = aim.split(' ')
                    # print(aim)
                    aims.append(aim)

            if len(aims):

                for i, det in enumerate(aims):
                    _, x_center, y_center, width, height = det
                    x_center, width = re_x * float(x_center), re_x * float(width)
                    y_center, height = re_y * float(y_center), re_y * float(height)
                    top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                    color = (0, 255, 0) # RGB
                    cv2.rectangle(img0, top_left, bottom_right, color, thickness=3)


        cv2.namedWindow('apex', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('apex', re_x // 2, re_y // 2)
        cv2.imshow('apex', img0)



        hwnd = win32gui.FindWindow(None, 'apex')
        CVRECT = cv2.getWindowImageRect('apex')



        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break