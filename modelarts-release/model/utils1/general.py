# YOLOR general utils

import glob
import logging
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml

from utils1.metrics import fitness
from utils1.torch_utils import init_torch_seeds

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
pd.options.display.max_columns = 10
cv2.setNumThreads(0)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def isdocker():
    return Path('/workspace').exists()


def emojis(str=''):
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def check_online():
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)
    except OSError:
        return False


def check_git_status():
    print(colorstr('github: '), end='')
    try:
        assert Path('.git').exists(), 'skipping check (not a git repository)'
        assert not isdocker(), 'skipping check (Docker image)'
        assert check_online(), 'skipping check (offline)'

        cmd = 'git fetch && git config --get remote.origin.url'
        url = subprocess.check_output(cmd, shell=True).decode().strip().rstrip('.git')
        branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()
        n = int(subprocess.check_output(f'git rev-list {branch}..origin/master --count', shell=True))
        if n > 0:
            s = f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. " \
                f"Use 'git pull' to update or 'git clone {url}' to download latest."
        else:
            s = f'up to date with {url} ✅'
        print(emojis(s))
    except Exception as e:
        print(e)


def check_requirements(requirements='requirements.txt', exclude=()):
    import pkg_resources as pkg
    prefix = colorstr('red', 'bold', 'requirements:')
    if isinstance(requirements, (str, Path)):
        file = Path(requirements)
        if not file.exists():
            print(f"{prefix} {file.resolve()} not found, check failed.")
            return
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:
        requirements = [x for x in requirements if x not in exclude]

    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:
            n += 1
            print(f"{prefix} {e.req} not found and is required by YOLOR, attempting auto-update...")
            print(subprocess.check_output(f"pip install '{e.req}'", shell=True).decode())

    if n:
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))


def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_imshow():
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_file(file):
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)
        assert len(files), f'File Not Found: {file}'
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"
        return files[0]


def check_dataset(dict):
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and len(s):
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):
                    f = Path(s).name
                    torch.hub.download_url_to_file(s, f)
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))
                else:
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))
            else:
                raise Exception('Dataset not found.')


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\033[30m',
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    if labels[0] is None:
        return torch.Tensor()

    labels = np.concatenate(labels, 0)
    classes = labels[:, 0].astype(np.int)
    weights = np.bincount(classes, minlength=nc)

    weights[weights == 0] = 1
    weights = 1 / weights
    weights /= weights.sum()
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    return image_weights


def coco80_to_coco91_class():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw
    y[:, 1] = h * x[:, 1] + padh
    return y


def segment2box(segment, width=640, height=640):
    x, y = segment.T
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))


def segments2boxes(segments):
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))


def resample_segments(segments, n=1000):
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    box2 = box2.T
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
    else:
        return iou


def bbox_alpha_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, alpha=2, eps=1e-9):
    box2 = box2.T
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = torch.pow(inter/union + eps, alpha)
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps
            rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
            rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
            rho2 = ((rho_x ** 2 + rho_y ** 2) / 4) ** alpha
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / ((1 + eps) - inter / union + v)
                return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))
        else:
            c_area = torch.max(cw * ch + eps, union)
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)
    else:
        return iou


def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def wh_iou(wh1, wh2):
    wh1 = wh1[:, None]
    wh2 = wh2[None]
    inter = torch.min(wh1, wh2).prod(2)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)


def box_giou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)
    areai = whi[:, :, 0] * whi[:, :, 1]

    return iou - (areai - union) / areai


def box_ciou(box1, box2, eps: float = 1e-7):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2

    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2

    w_pred = box1[:, None, 2] - box1[:, None, 0]
    h_pred = box1[:, None, 3] - box1[:, None, 1]

    w_gt = box2[:, 2] - box2[:, 0]
    h_gt = box2[:, 3] - box2[:, 1]

    v = (4 / (torch.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (centers_distance_squared / diagonal_distance_squared) - alpha * v


def box_diou(box1, box2, eps: float = 1e-7):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2
    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2

    return iou - (centers_distance_squared / diagonal_distance_squared)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):

    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    min_wh, max_wh = 2, 4096
    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        if nc == 1:
            x[:, 5:] = x[:, 4:5]
        else:
            x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def non_max_suppression_kpt(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), kpt_label=False, nc=None, nkpt=None):

    if nc is None:
        nc = prediction.shape[2] - 5 if not kpt_label else prediction.shape[2] - 56
    xc = prediction[..., 4] > conf_thres

    min_wh, max_wh = 2, 4096
    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0,6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:5+nc] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            if not kpt_label:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            else:
                kpts = x[:, 6:]
                conf, j = x[:, 5:6].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def strip_optimizer(f='best.pt', s=''):
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':
        x[k] = None
    x['epoch'] = -1
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    a = '%10s' * len(hyp) % tuple(hyp.keys())
    b = '%10.3g' * len(hyp) % tuple(hyp.values())
    c = '%10.4g' * len(results) % results
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    with open('evolve.txt', 'a') as f:
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)
    x = x[np.argsort(-fitness(x))]
    np.savetxt('evolve.txt', x, '%10.3g')

    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))


def apply_classifier(x, model, img, im0):
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):
        if d is not None and len(d):
            d = d.clone()

            b = xyxy2xywh(d[:, :4])
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
            b[:, 2:] = b[:, 2:] * 1.3 + 30
            d[:, :4] = xywh2xyxy(b).long()

            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))

                im = im[:, :, ::-1].transpose(2, 0, 1)
                im = np.ascontiguousarray(im, dtype=np.float32)
                im /= 255.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)
            x[i] = x[i][pred_cls1 == pred_cls2]

    return x


def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"
