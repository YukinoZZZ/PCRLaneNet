# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:29:35 2020

@author: 20150
"""

import cv2
import numpy as np
from scipy import interpolate


image_path = '/home/dxjforyou/Desktop/ApolloScapes/road02_ins/CULane/test/driver_100_30frame/05251548_0439.MP4/01200.jpg'
line_path = '/home/dxjforyou/Desktop/ApolloScapes/road02_ins/CULane/test/driver_100_30frame/05251548_0439.MP4/01200.lines.txt'

image = cv2.imread(image_path)
binaryimage = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)

f = open(line_path)
data = []
len_lane = []
for line in f.readlines():
    index = 0
    for point in line.split(' '):
        if point != '\n':
            data.append(float(point))
            index = index + 1
    len_lane.append(index/2)

col = 2
row = int(len(data) / col)
line_data = np.reshape(np.array(data), (row, col))

target_x = []
target_y = []
tmp_x = []
tmp_y = []
length = len(line_data)
if length>0:
    index = 0
    for m in range(len(len_lane)):
        for n in range(int(len_lane[m])):
            tmp_x.append(line_data[index][0])
            tmp_y.append(line_data[index][1])
            index = index + 1

        start = 0.0
        end = 0.0
        for tmp_index in range(len(tmp_y)):
            if tmp_y[tmp_index]>=0 and tmp_y[tmp_index]<= 800 and tmp_x[tmp_index]>=0 and tmp_x[tmp_index]<=1640:
                start = tmp_y[tmp_index]
                break

        for tmp_index in range(len(tmp_y)):
            if tmp_y[len(tmp_y)-tmp_index-1]>=0 and tmp_y[len(tmp_y)-tmp_index-1]<= 800 and tmp_x[len(tmp_y)-tmp_index-1]>=0 and tmp_x[len(tmp_y)-tmp_index-1]<=1640:
                end = tmp_y[len(tmp_y)-tmp_index-1]
                break




        yn = np.linspace(start, end, num=12)

        if len(tmp_y) >= 3:
            f = interpolate.interp1d(tmp_y, tmp_x, kind="quadratic")
        else:
            f = interpolate.interp1d(tmp_y, tmp_x, kind="slinear")

        xn = f(yn)
        target_x.append(xn)
        target_y.append(yn)
        tmp_x = []
        tmp_y = []

    target = []
    tmp_target_x = target_x.copy()
    tmp_target_y = target_y.copy()
    target_x = []
    target_y = []
    flag = []
    for j in range(len(tmp_target_x)):
        flag.append(tmp_target_x[j][0])
    flag = np.array(flag)
    sort_index = np.argsort(flag)
    for j in range(len(tmp_target_x)):
        target_x.append(tmp_target_x[sort_index[j]])
        target_y.append(tmp_target_y[sort_index[j]])

    for j in range(len(target_x)):
        tmp_target = np.stack((target_x[j], target_y[j]), axis=-1)
        target.append(tmp_target)

    else:
        target = []
