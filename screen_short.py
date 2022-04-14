# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 07:30:59 2022

@author: 11510
"""

import cv2
import numpy as np
from PIL import ImageGrab


def capture(left, top, right, bottom):
    img = ImageGrab.grab(bbox=(left, top, right, bottom))
    img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
    r, g, b = cv2.split(img)
    cv2.merge([b, g, r], img)
    return img


cv2.imshow("screen", capture(752, 332, 1168, 784))
cv2.waitKey(0)
