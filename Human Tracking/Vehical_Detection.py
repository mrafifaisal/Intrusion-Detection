# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:25:59 2022

@author: sohaib
"""
# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('video.mp4')
input_size = 320


print("Load")
