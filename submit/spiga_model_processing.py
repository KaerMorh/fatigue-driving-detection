from spiga.demo.visualize.plotter import Plotter
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import copy
import cv2
import numpy as np

# Process image
dataset = 'wflw'
processor = SPIGAFramework(ModelConfig(dataset))

def draw_headpose_simple(canvas, bbox, headpose, hpose_axe_length=2, focal_ratio=1, thick=2,
                         colors=(255, 0, 0), euler=False):
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1
    focal_length_x = w * focal_ratio
    focal_length_y = h * focal_ratio
    face_center = (x1 + (w * 0.5)), (y1 + (h * 0.5))

    cam_matrix = np.array([[focal_length_x, 0, face_center[0]],
                           [0, focal_length_y, face_center[1]],
                           [0, 0, 1]], dtype=np.float32)

    rot = np.float32(headpose)
    K = cam_matrix

    if euler:
        rot = euler_to_rotation_matrix(rot)

    rotV, _ = cv2.Rodrigues(rot)
    points = np.float32([[hpose_axe_length, 0, 0], [0, -hpose_axe_length, 0], [0, 0, -hpose_axe_length], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, np.zeros(3), K, (0, 0, 0))
    canvas = cv2.line(canvas, tuple(axisPoints[3].ravel().astype(int)), tuple(axisPoints[2].ravel().astype(int)), colors, thick)
    canvas = cv2.line(canvas, tuple(axisPoints[3].ravel().astype(int)), tuple(axisPoints[1].ravel().astype(int)), colors, thick)
    canvas = cv2.line(canvas, tuple(axisPoints[3].ravel().astype(int)), tuple(axisPoints[0].ravel().astype(int)), colors, thick)
    return canvas

def euler_to_rotation_matrix(headpose):
    euler = np.array([-(headpose[0] - 90), -headpose[1], -(headpose[2] + 90)])
    rad = euler * (np.pi / 180.0)
    cy = np.cos(rad[0])
    sy = np.sin(rad[0])
    cp = np.cos(rad[1])
    sp = np.sin(rad[1])
    cr = np.cos(rad[2])
    sr = np.sin(rad[2])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])  # yaw
    Rp = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])  # pitch
    Rr = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])  # roll
    return np.matmul(np.matmul(Ry, Rp), Rr)



def spig_process_frame(frame, bbox):
    features = processor.inference(frame, [bbox])

    x0, y0, w, h = bbox
    # canvas = copy.deepcopy(frame)
    landmarks = np.array(features['landmarks'][0])
    headpose = np.array(features['headpose'][0])

    # Plot features
    # plotter = Plotter()
    # canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
    # canvas = plotter.hpose.draw_headpose(canvas, [x0,y0,x0+w,y0+h], headpose[:3], headpose[3:], euler=True)
    # canvas = draw_headpose_simple(canvas, [x0, y0, x0 + w, y0 + h], headpose[:3], headpose[3:], euler=True)
    # (h, w) = canvas.shape[:2]
    # canvas = cv2.resize(canvas, (512, int(h * 512 / w)))

    return 1


if __name__ == '__main__':
    pass
