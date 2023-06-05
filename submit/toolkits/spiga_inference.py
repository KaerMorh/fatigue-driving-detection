
import copy
from collections import OrderedDict
import numpy as np
from spiga.demo.visualize.plotter import Plotter
import cv2

# in: model + frame + bbox tuple([int, int, int, int]) (x, y, w, h)
def spiga_process_frame(processor, frame, bbox,plot = 0):

    features = processor.inference(frame, [bbox])
    landmarks = np.array(features['landmarks'][0])
    headpose = np.array(features['headpose'][0])

    if plot == 1:
        plotter = Plotter()
        frame = plotter.landmarks.draw_landmarks(frame, landmarks)
        x0, y0, w, h = bbox
        frame = plotter.hpose.draw_headpose(frame, [x0,y0,x0+w,y0+h], headpose[:3], headpose[3:], euler=True)

    return landmarks, headpose, frame

if __name__ == '__main__':
    print()
