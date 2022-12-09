import numpy as np
from PIL import Image

def xywh_to_xyxy(bbox):
    """
    Script to convert bbox coordinates (in dictionary format) into a list.
    Args:
    :bbox (dictionary): {'x':x1, 'y':y1, 'w': w, 'h': h}
    Returns:
    : bbox (list): [x1, y1, x2, y2]
    """
    x1 = bbox['x']
    y1 = bbox['y']
    x2 = x1 + bbox['w']
    y2 = y1 + bbox['h']

    return [x1, y1, x2, y2]

def xyxy_to_xywh(bbox):
    """
    Script to convert bbox coordinates (in list format) into a dictionary.
    Args:
    : bbox (list): [x1, y1, x2, y2]
    Returns:
    :bbox (dictionary): {'x':x1, 'y':y1, 'w': w, 'h': h}
    """
    x1,y1,x2,y2 = bbox
    w = x2 -x1
    h = y2 - y1

    return {'x' : x1, 'y' : y1, 'w' : w, 'h' : h}

def getScale(width, height, target_size):

    x_, y_ = width, height

    xtargetSize, ytargetSize = target_size

    x_scale = float(xtargetSize) / x_
    y_scale = float(ytargetSize) / y_

    return x_scale, y_scale

def ResizeBox(box, x_scale, y_scale):
    """
    Rescale the bounding-box according to the x-scale and y-scale.
    It is used to resize the box which is drawn on the original image and now you have resized the original image according to the x-scale and y-scale.
    """
    (origLeft, origTop, origRight, origBottom) = box

    x = int(np.round(origLeft * x_scale))
    y = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale))

    return [x,y,xmax,ymax]
