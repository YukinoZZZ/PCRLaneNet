#########################################################################
##
##  Data loader source code for CULane dataset
##
#########################################################################


import math
import numpy as np
import cv2
import random
from copy import deepcopy
from Parameter import Parameters

from scipy import interpolate

from random import shuffle

def func(x,a,b,c):
    return a*np.sqrt(x)*(b*np.square(x)+c)

#########################################################################
## some iamge transform utils
#########################################################################
def Translate_Points(point, translation):
    point = point + translation

    return point


def Rotate_Points(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


#########################################################################
## Data loader class
#########################################################################
class Generator(object):
    ################################################################################
    ## initialize (load data set from url)
    ################################################################################
    def __init__(self):
        self.p = Parameters()

        # load training set
        self.root_url = '/home/doujian/Desktop/CULane/list/'
        self.train_image_data = []
        self.train_instance_data = []
        self.train_exist = []

        with open(self.root_url + 'train_gt.txt') as f:
            for _info in f:
                info_tmp = _info.strip(' ').split()
                self.train_image_data.append(self.p.train_root_url + info_tmp[0])
                self.train_instance_data.append(self.p.train_root_url + info_tmp[1])
                exist_label = np.zeros(4)
                exist_label[0] = int(info_tmp[2])
                exist_label[1] = int(info_tmp[3])
                exist_label[2] = int(info_tmp[4])
                exist_label[3] = int(info_tmp[5])
                self.train_exist.append(exist_label)



        self.size_train = len(self.train_image_data)


        # load test set
        self.test_data = []
        with open(self.root_url + 'test0_normal.txt') as f:
            for _info in f:
                info_tmp = _info.strip(' ').split()
                self.test_data.append(self.p.test_root_url + info_tmp[0])

        self.size_test = len(self.test_data)

        assert len(self.train_exist) == len(self.train_instance_data) == len(self.train_image_data)
        index = [i for i in range(self.size_train)]
        shuffle(index)
        tmp_train_exist = self.train_exist[:]
        self.train_exist = []
        self.train_exist = [tmp_train_exist[index[i]] for i in range(len(index))]
        tmp_train_instance_data  = self.train_instance_data[:]
        self.train_instance_data = []
        self.train_instance_data = [tmp_train_instance_data[index[i]] for i in range(len(index))]
        tmp_train_image_data = self.train_image_data[:]
        self.train_image_data = []
        self.train_image_data = [tmp_train_image_data[index[i]] for i in range(len(index))]



    #################################################################################################################
    ## Generate data as much as batchsize and augment data (filp, translation, rotation, gaussian noise, scaling)
    #################################################################################################################
    def Generate(self):
        cuts = [(b, min(b + self.p.batch_size, self.size_train)) for b in range(0, self.size_train, self.p.batch_size)]
        for start, end in cuts:
            self.start = start
            # resize original image to 800*288
            self.inputs, self.lines_data, self.exist_label, self.test_image, self.len_lanes = self.Resize_data(start, end)

            self.raw_linedata = self.lines_data

            self.actual_batchsize = self.inputs.shape[0]
            self.Flip()
            self.Translation()
            self.Rotate()
            self.Gaussian()
            self.Change_intensity()
            self.Shadow()

            yield self.inputs / 255.0, self.R2_Points, self.R1_Points, self.L2_Points, self.L1_Points, self.exist_label, self.test_image / 255.0  # generate normalized image

    #################################################################################################################
    ## Generate test data
    #################################################################################################################
    def Generate_Test(self):
        for i in range(self.size_test):
            #print(i)
            raw_image = cv2.imread(self.test_data[i])
            test_image = cv2.imread(self.test_data[i])
            # print(self.test_data[i]['raw_file'])
            ratio_w = self.p.x_size * 1.0 / test_image.shape[1]
            ratio_h = self.p.y_size * 1.0 / test_image.shape[0]
            test_image = cv2.resize(test_image, (self.p.x_size, self.p.y_size))
            test_line_path = self.test_data[i][0:-3] + 'lines.txt'
            f = open(test_line_path)
            data = []
            len_lane = []
            for line in f.readlines():
                index = 0
                for point in line.split(' '):
                    if point != '\n':
                        data.append(float(point))
                        index = index + 1
                len_lane.append(index / 2)

            col = 2
            row = int(len(data) / col)

            line_data = np.reshape(np.array(data), (row, col))
            for j in range(row):
                line_data[j][0] = line_data[j][0] * ratio_w
                line_data[j][1] = line_data[j][1] * ratio_h

            len_lanes=[]
            lines_data=[]
            lines_data.append(line_data)
            len_lanes.append(len_lane)

            yield np.rollaxis(test_image, axis=2, start=0) / 255.0, lines_data, len_lanes, np.rollaxis(raw_image, axis=2, start=0) / 255.0

    #################################################################################################################
    ## Generate batch test data
    ###########################################################################################################
    def Generate_batch_test(self):
        cuts = [(b, min(b + self.p.batch_size, self.size_test)) for b in range(0, self.size_test, self.p.batch_size)]
        for start, end in cuts:
            inputs = []
            raw_input = []
            lines_data = []
            len_lanes = []
            for i in range(start,end):
                raw_image = cv2.imread(self.test_data[i])
                # ratio_w = self.p.x_size * 1.0 / temp_image.shape[1]
                # ratio_h = self.p.y_size * 1.0 / temp_image.shape[0]
                temp_image = cv2.resize(raw_image, (self.p.x_size, self.p.y_size))
                raw_input.append(raw_image)
                inputs.append(np.rollaxis(temp_image, axis=2, start=0))
                line_path = self.test_data[i][0:-3] + 'lines.txt'
                f = open(line_path)
                data = []
                len_lane = []
                for line in f.readlines():
                    index = 0
                    for point in line.split(' '):
                        if point != '\n':
                            data.append(float(point))
                            index = index + 1
                    len_lane.append(index / 2)

                col = 2
                row = int(len(data) / col)

                line_data = np.reshape(np.array(data), (row, col))
                # for j in range(row):
                #     line_data[j][0] = line_data[j][0] * ratio_w
                #     line_data[j][1] = line_data[j][1] * ratio_h
                lines_data.append(line_data)
                len_lanes.append(len_lane)
            yield np.array(inputs)/255.0,lines_data,len_lanes,np.array(raw_input)

    #################################################################################################################
    ## resize original image to 512*256 and matching correspond points
    #################################################################################################################
    def Resize_data(self, start, end):
        inputs = []
        lines_data = []
        len_lanes = []
        exist_label = []

        # choose data from each number of lanes
        for i in range(start, end):


            # train set image
            exist_label.append(self.train_exist[i])
            temp_image = cv2.imread(self.train_image_data[i])
            ratio_w = self.p.x_size * 1.0 / temp_image.shape[1]
            ratio_h = self.p.y_size * 1.0 / temp_image.shape[0]
            temp_image = cv2.resize(temp_image, (self.p.x_size, self.p.y_size))
            inputs.append(np.rollaxis(temp_image, axis=2, start=0))

            line_path = self.train_image_data[i][0:-3] + 'lines.txt'
            f = open(line_path)
            data = []
            len_lane = []
            for line in f.readlines():
                index = 0
                for point in line.split(' '):
                    if point!='\n':
                        data.append(float(point))
                        index = index + 1
                len_lane.append(index/2)

            col = 2
            row = int(len(data)/col)

            line_data = np.reshape(np.array(data),(row,col))
            for j in range(row):
                line_data[j][0] = line_data[j][0] * ratio_w
                line_data[j][1] = line_data[j][1] * ratio_h
            lines_data.append(line_data)
            len_lanes.append(len_lane)



        # test set image
        test_index = random.randrange(0, self.size_test - 1)
        test_image = cv2.imread(self.test_data[test_index])
        test_image = cv2.resize(test_image, (self.p.x_size, self.p.y_size))

        ##change lane point to point ground truth number
        processed_data = []

        for k in range(len(lines_data)):
            line_data = lines_data[k]
            len_lane = len_lanes[k]
            length = len(line_data)
            if length>0:
                index = 0
                tmp_xn = []
                tmp_yn = []
                for m in range(len(len_lane)):
                    tmp_x = []
                    tmp_y = []
                    for n in range(int(len_lane[m])):
                        tmp_x.append(line_data[index][0])
                        tmp_y.append(line_data[index][1])
                        index = index + 1
                    len_lanes[k][m] = self.p.ground_truth_num

                    for tmp_index in range(len(tmp_y)):
                        if tmp_y[tmp_index]>0 and tmp_y[tmp_index]<self.p.y_size and tmp_x[tmp_index]>0 and tmp_x[tmp_index]<self.p.x_size:
                            start = tmp_y[tmp_index]
                            break

                    for tmp_index in range(len(tmp_y)):
                        if tmp_y[len(tmp_y)-tmp_index-1]>0 and tmp_y[len(tmp_y)-tmp_index-1]<self.p.y_size and tmp_x[len(tmp_y)-tmp_index-1]>0 and tmp_x[len(tmp_y)-tmp_index-1]<self.p.x_size:
                            end = tmp_y[len(tmp_y)-tmp_index-1]
                            break
                    yn = np.linspace(start, end, num=self.p.ground_truth_num)
                    if len(tmp_y) >= 3:
                        f = interpolate.interp1d(tmp_y, tmp_x, kind="quadratic")
                    else:
                        f = interpolate.interp1d(tmp_y, tmp_x, kind="slinear")
                    xn = f(yn)
                    ##test iou
                    # binary_image_target = np.zeros((self.p.image_H, self.p.image_W, 1), np.uint8)
                    # binary_image_gt = np.zeros((self.p.image_H, self.p.image_W, 1), np.uint8)
                    #
                    # for kk in range(len(tmp_x) - 1):
                    #     cv2.line(binary_image_gt, (int(tmp_x[kk]/ratio_w), int(tmp_y[kk]/ratio_h)),
                    #              (int(tmp_x[kk+1]/ratio_w), int(tmp_y[kk+1]/ratio_h)), 255, self.p.line_width)
                    # for kk in range(len(xn) - 1):
                    #     cv2.line(binary_image_target, (int(xn[kk]/ratio_w), int(yn[kk]/ratio_h)),
                    #              (int(xn[kk+1]/ratio_w), int(yn[kk+1]/ratio_h)), 255, self.p.line_width)
                    # PR = binary_image_target.nonzero()[0].size
                    # GT = binary_image_gt.nonzero()[0].size
                    # TP_point = (binary_image_target * binary_image_gt).nonzero()[0].size
                    # union = PR + GT - TP_point
                    # iou = TP_point / union
                    # if iou<0.9:
                    #     print('error!!')


                    for point_index in range(len(xn)):
                        tmp_xn.append(xn[point_index])
                        tmp_yn.append(yn[point_index])
                lane_point = np.stack((tmp_xn,tmp_yn),axis=1)
            else:
                lane_point = np.array([])
            processed_data.append(lane_point)








        return np.array(inputs), processed_data, exist_label, np.rollaxis(test_image, axis=2, start=0), len_lanes

    #################################################################################################################
    ## Generate random unique indices according to ratio
    #################################################################################################################
    def Random_indices(self, ratio):
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)

    #################################################################################################################
    ## Add Gaussian noise
    #################################################################################################################
    def Gaussian(self):
        indices = self.Random_indices(self.p.noise_ratio)
        img = np.zeros((self.p.y_size, self.p.x_size, 3), np.uint8)
        m = (0, 0, 0)
        s = (20, 20, 20)

        for i in indices:
            test_image = deepcopy(self.inputs[i])
            test_image = np.rollaxis(test_image, axis=2, start=0)
            test_image = np.rollaxis(test_image, axis=2, start=0)
            cv2.randn(img, m, s)
            test_image = test_image + img
            test_image = np.rollaxis(test_image, axis=2, start=0)
            self.inputs[i] = test_image

    #################################################################################################################
    ## Change intensity
    #################################################################################################################
    def Change_intensity(self):
        indices = self.Random_indices(self.p.intensity_ratio)
        for i in indices:
            test_image = deepcopy(self.inputs[i])
            test_image = np.rollaxis(test_image, axis=2, start=0)
            test_image = np.rollaxis(test_image, axis=2, start=0)

            hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            value = int(random.uniform(-60.0, 60.0))
            if value > 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = -1 * value
                v[v < lim] = 0
                v[v >= lim] -= lim
            final_hsv = cv2.merge((h, s, v))
            test_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            test_image = np.rollaxis(test_image, axis=2, start=0)
            self.inputs[i] = test_image

    #################################################################################################################
    ## Generate random shadow in random region
    #################################################################################################################
    def Shadow(self, min_alpha=0.5, max_alpha=0.75):
        indices = self.Random_indices(self.p.shadow_ratio)
        for i in indices:
            test_image = deepcopy(self.inputs[i])
            test_image = np.rollaxis(test_image, axis=2, start=0)
            test_image = np.rollaxis(test_image, axis=2, start=0)

            top_x, bottom_x = np.random.randint(0, 512, 2)
            coin = np.random.randint(2)
            rows, cols, _ = test_image.shape
            shadow_img = test_image.copy()
            if coin == 0:
                rand = np.random.randint(2)
                vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
                if rand == 0:
                    vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
                elif rand == 1:
                    vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
                mask = test_image.copy()
                channel_count = test_image.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (0,) * channel_count
                cv2.fillPoly(mask, [vertices], ignore_mask_color)
                rand_alpha = np.random.uniform(min_alpha, max_alpha)
                cv2.addWeighted(mask, rand_alpha, test_image, 1 - rand_alpha, 0., shadow_img)
                shadow_img = np.rollaxis(shadow_img, axis=2, start=0)
                self.inputs[i] = shadow_img

    #################################################################################################################
    ## Flip
    #################################################################################################################
    def Flip(self):
        indices = self.Random_indices(self.p.flip_ratio)
        self.flipflag = np.zeros(self.actual_batchsize)
        for i in indices:
            self.flipflag[i] = 1
            temp_image = deepcopy(self.inputs[i])
            temp_image = np.rollaxis(temp_image, axis=2, start=0)
            temp_image = np.rollaxis(temp_image, axis=2, start=0)

            temp_image = cv2.flip(temp_image, 1)
            temp_image = np.rollaxis(temp_image, axis=2, start=0)
            self.inputs[i] = temp_image


            line_data = self.lines_data[i]
            for j in range(len(line_data)):
                line_data[j][0] = self.p.x_size - line_data[j][0]

            self.lines_data[i] = line_data
            exist = self.exist_label[i].copy()
            for j in range(len(exist)):
                self.exist_label[i][len(exist)-j-1] = exist[j]


    #################################################################################################################
    ## Translation
    #################################################################################################################
    def Translation(self):
        indices = self.Random_indices(self.p.translation_ratio)
        self.tran_indices = indices
        for i in indices:
            temp_image = deepcopy(self.inputs[i])
            temp_image = np.rollaxis(temp_image, axis=2, start=0)
            temp_image = np.rollaxis(temp_image, axis=2, start=0)

            self.tx = np.random.randint(-50, 50)
            self.ty = np.random.randint(-30, 30)

            temp_image = cv2.warpAffine(temp_image, np.float32([[1, 0, self.tx], [0, 1, self.ty]]),
                                        (self.p.x_size, self.p.y_size))
            temp_image = np.rollaxis(temp_image, axis=2, start=0)
            self.inputs[i] = temp_image


            line_data = self.lines_data[i]
            len_lane = self.len_lanes[i]
            length = len(line_data)
            if length > 0:
                index = 0
                tmp_xn = []
                tmp_yn = []
                for m in range(len(len_lane)):
                    tmp_x = []
                    tmp_y = []
                    for n in range(int(len_lane[m])):
                        tmp_x.append(line_data[index][0])
                        tmp_y.append(line_data[index][1])
                        index = index + 1
                    tran_tmp_x = [xx + self.tx for xx in tmp_x]
                    tran_tmp_y = [yy + self.ty for yy in tmp_y]
                    flag = False
                    out_flag = []
                    for k in range(len(tmp_x)):
                        if tran_tmp_x[k] < 0 or tran_tmp_x[k] > self.p.x_size or tran_tmp_y[k] < 0 or tran_tmp_y[k] > self.p.y_size:
                            flag = True
                        else:
                            out_flag.append(k)
                    if len(out_flag)<=1:
                        flag = False
                        tran_tmp_y = [-1 for _ in range(self.p.ground_truth_num)]
                        tran_tmp_x = [-1 for _ in range(self.p.ground_truth_num)]
                    if flag:
                        normal_tmp_y = [tmp_y[k] for k in out_flag]
                        start = max(normal_tmp_y)
                        end = min(normal_tmp_y)
                        yn = np.linspace(start, end, num=self.p.ground_truth_num)
                        if len(tmp_y) >= 3:
                            f = interpolate.interp1d(tmp_y, tmp_x, kind="quadratic")
                        else:
                            f = interpolate.interp1d(tmp_y, tmp_x, kind="slinear")
                        xn = f(yn)
                        xn = xn + self.tx
                        yn = yn + self.ty
                    else:
                        xn = np.array(tran_tmp_x)
                        yn = np.array(tran_tmp_y)
                    for point_index in range(len(xn)):
                        tmp_xn.append(xn[point_index])
                        tmp_yn.append(yn[point_index])
                lane_point = np.stack((tmp_xn, tmp_yn), axis=1)
            else:
                lane_point = np.array([])

            self.lines_data[i] = lane_point



    #################################################################################################################
    ## Rotate
    #################################################################################################################
    def Rotate(self):
        indices = self.Random_indices(self.p.rotate_ratio)
        self.rotate_indices = indices
        tmp_target_points = []
        R2_target_points = np.zeros([self.actual_batchsize,self.p.ground_truth_num,2])
        R1_target_points = np.zeros([self.actual_batchsize,self.p.ground_truth_num,2])
        L2_target_points = np.zeros([self.actual_batchsize,self.p.ground_truth_num,2])
        L1_target_points = np.zeros([self.actual_batchsize,self.p.ground_truth_num,2])

        for i in indices:
            temp_image = deepcopy(self.inputs[i])
            temp_image = np.rollaxis(temp_image, axis=2, start=0)
            temp_image = np.rollaxis(temp_image, axis=2, start=0)

            angle = np.random.randint(-10, 10)
            self.angle = angle

            M = cv2.getRotationMatrix2D((self.p.x_size / 2, self.p.y_size / 2), angle, 1)

            temp_image = cv2.warpAffine(temp_image, M, (self.p.x_size, self.p.y_size))
            temp_image = np.rollaxis(temp_image, axis=2, start=0)
            self.inputs[i] = temp_image


            line_data = self.lines_data[i]

            len_lane = self.len_lanes[i]
            length = len(line_data)
            if length > 0:
                index = 0
                tmp_xn = []
                tmp_yn = []
                for m in range(len(len_lane)):
                    tmp_x = []
                    tmp_y = []
                    for n in range(int(len_lane[m])):
                        tmp_x.append(line_data[index][0])
                        tmp_y.append(line_data[index][1])
                        index = index + 1
                    tran_tmp_x = []
                    tran_tmp_y = []
                    for j in range(len(tmp_x)):
                        x, y = Rotate_Points((self.p.x_size / 2, self.p.y_size / 2),
                                             (tmp_x[j], tmp_y[j]),
                                             (-angle * 2 * np.pi) / 360)
                        tran_tmp_x.append(x)
                        tran_tmp_y.append(y)
                    flag = False
                    out_flag = []
                    for k in range(len(tmp_x)):
                        if tran_tmp_x[k] < 0 or tran_tmp_x[k] > self.p.x_size or tran_tmp_y[
                            k] < 0 or tran_tmp_y[k] > self.p.y_size:
                            flag = True
                        else:
                            out_flag.append(k)

                    if len(out_flag)<=1:
                        flag = False
                        tran_tmp_y = [-1 for _ in range(self.p.ground_truth_num)]
                        tran_tmp_x = [-1 for _ in range(self.p.ground_truth_num)]
                    if flag:
                        normal_tmp_y = [tmp_y[k] for k in out_flag]
                        start = max(normal_tmp_y)
                        end = min(normal_tmp_y)
                        yn = np.linspace(start, end, num=self.p.ground_truth_num)
                        if len(tmp_y) >= 3:
                            f = interpolate.interp1d(tmp_y, tmp_x, kind="quadratic")
                        else:
                            f = interpolate.interp1d(tmp_y, tmp_x, kind="slinear")
                        xn = f(yn)
                        for j in range(len(xn)):
                            x, y = Rotate_Points((self.p.x_size / 2, self.p.y_size / 2),
                                                 (xn[j], yn[j]),
                                                 (-angle * 2 * np.pi) / 360)
                            xn[j] = x
                            yn[j] = y
                    else:
                        xn = np.array(tran_tmp_x)
                        yn = np.array(tran_tmp_y)
                    for point_index in range(len(xn)):
                        tmp_xn.append(xn[point_index])
                        tmp_yn.append(yn[point_index])
                lane_point = np.stack((tmp_xn, tmp_yn), axis=1)
            else:
                lane_point = np.array([])

            self.lines_data[i] = lane_point



        for k in range(len(self.lines_data)):
            line_data = self.lines_data[k]
            len_lane = self.len_lanes[k]
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

                    # sorted_tmp_y = sorted(tmp_y,reverse=True)
                    # re_sorted_tmp_x = sorted(tmp_x,reverse=True)
                    # sorted_tmp_x = sorted(tmp_x)
                    # start = 0
                    # end = 0
                    # if sorted_tmp_y==tmp_y:
                    #     for tmp_index in range(len(tmp_y)):
                    #         if tmp_y[tmp_index]>0 and tmp_y[tmp_index]<self.p.y_size and tmp_x[tmp_index]>0 and tmp_x[tmp_index]<self.p.x_size:
                    #             start = tmp_y[tmp_index]
                    #             break
                    #
                    #     for tmp_index in range(len(tmp_y)):
                    #         if tmp_y[len(tmp_y)-tmp_index-1]>0 and tmp_y[len(tmp_y)-tmp_index-1]<self.p.y_size and tmp_x[len(tmp_y)-tmp_index-1]>0 and tmp_x[len(tmp_y)-tmp_index-1]<self.p.x_size:
                    #             end = tmp_y[len(tmp_y)-tmp_index-1]
                    #             break
                    #     yn = np.linspace(start, end, self.p.point_num)
                    #     if len(tmp_y) >= 3:
                    #         f = interpolate.interp1d(tmp_y, tmp_x, kind="quadratic")
                    #     else:
                    #         f = interpolate.interp1d(tmp_y, tmp_x, kind="slinear")
                    #     if start==0 and end==0:
                    #         xn = yn
                    #     else:
                    #         xn = f(yn)
                    #
                    # elif re_sorted_tmp_x==tmp_x or sorted_tmp_x==tmp_x:
                    #     for tmp_index in range(len(tmp_y)):
                    #         if tmp_y[tmp_index] > 0 and tmp_y[tmp_index] < self.p.y_size and tmp_x[tmp_index] > 0 and \
                    #                 tmp_x[tmp_index] < self.p.x_size:
                    #             start = tmp_x[tmp_index]
                    #             break
                    #
                    #     for tmp_index in range(len(tmp_y)):
                    #         if tmp_y[len(tmp_y) - tmp_index - 1] > 0 and tmp_y[
                    #             len(tmp_y) - tmp_index - 1] < self.p.y_size and tmp_x[
                    #             len(tmp_y) - tmp_index - 1] > 0 and tmp_x[len(tmp_y) - tmp_index - 1] < self.p.x_size:
                    #             end = tmp_x[len(tmp_y) - tmp_index - 1]
                    #             break
                    #     xn = np.linspace(start, end, self.p.point_num)
                    #     if len(tmp_y) >= 3:
                    #         f = interpolate.interp1d(tmp_x, tmp_y, kind="quadratic")
                    #     else:
                    #         f = interpolate.interp1d(tmp_x, tmp_y, kind="slinear")
                    #     if start == 0 and end == 0:
                    #         yn = xn
                    #     else:
                    #         yn = f(xn)
                    # else:
                    #     xn, yn = fit_point(tmp_x,tmp_y,self.p)
                    #
                    #
                    # #test iou
                    # ratio_w = self.p.x_size * 1.0 / 1640
                    # ratio_h = self.p.y_size * 1.0 / 590
                    # binary_image_target = np.zeros((self.p.image_H, self.p.image_W, 1), np.uint8)
                    # binary_image_gt = np.zeros((self.p.image_H, self.p.image_W, 1), np.uint8)
                    #
                    # for kk in range(len(tmp_x) - 1):
                    #     cv2.line(binary_image_gt, (int(tmp_x[kk]/ratio_w), int(tmp_y[kk]/ratio_h)),
                    #              (int(tmp_x[kk+1]/ratio_w), int(tmp_y[kk+1]/ratio_h)), 255, self.p.line_width)
                    # for kk in range(len(xn) - 1):
                    #     cv2.line(binary_image_target, (int(xn[kk]/ratio_w), int(yn[kk]/ratio_h)),
                    #              (int(xn[kk+1]/ratio_w), int(yn[kk+1]/ratio_h)), 255, self.p.line_width)
                    # PR = binary_image_target.nonzero()[0].size
                    # GT = binary_image_gt.nonzero()[0].size
                    # TP_point = (binary_image_target * binary_image_gt).nonzero()[0].size
                    # union = PR + GT - TP_point
                    # iou = TP_point / union
                    # # binary_image_target = binary_image_target.reshape(self.p.image_H, self.p.image_W)
                    # # cv2.imwrite('test_result/result2.png', binary_image_target)
                    # if iou<0.9 and iou!=0:
                    #     binary_image_gt = binary_image_gt.reshape(self.p.image_H, self.p.image_W)
                    #     cv2.imwrite('test_result/result1.png', binary_image_gt)
                    #     binary_image_target = binary_image_target.reshape(self.p.image_H, self.p.image_W)
                    #     cv2.imwrite('test_result/result2.png', binary_image_target)
                    #     print('error!!')
                    # xn = []
                    # yn = []
                    # for tmp_index in range(len(tmp_y)):
                    #     if tmp_y[tmp_index]>0 and tmp_y[tmp_index]<self.p.y_size and tmp_x[tmp_index]>0 and tmp_x[tmp_index]<self.p.x_size:
                    #         xn.append(tmp_x[tmp_index])
                    #         yn.append(tmp_y[tmp_index])
                    # if len(xn)<self.p.ground_truth_num and len(xn)>=2:
                    #     add_nums = self.p.ground_truth_num - len(xn)
                    #     add_points_yn = np.linspace(yn[len(xn)-2], yn[len(xn)-1], add_nums+2)
                    #     f = interpolate.interp1d(yn[len(xn)-2:], xn[len(xn)-2:], kind="slinear")
                    #     add_points_xn = f(add_points_yn)
                    #     correct_x = [xn[zzz] for zzz in range(len(xn) - 2)]
                    #     correct_y = [yn[zzz] for zzz in range(len(yn) - 2)]
                    #     for zzz in range(len(add_points_yn)):
                    #         correct_x.append(add_points_xn[zzz])
                    #         correct_y.append(add_points_yn[zzz])
                    #     target_x.append(correct_x)
                    #     target_y.append(correct_y)
                    # elif len(xn)<2:
                    #     correct_x = [-1 for _ in range(self.p.ground_truth_num)]
                    #     correct_y = [-1 for _ in range(self.p.ground_truth_num)]
                    #     target_x.append(correct_x)
                    #     target_y.append(correct_y)
                    # else:
                    #     target_x.append(xn)
                    #     target_y.append(yn)
                    target_x.append(tmp_x)
                    target_y.append(tmp_y)

                    tmp_x = []
                    tmp_y = []

                target = []

                for j in range(len(target_x)):
                    tmp_target = np.stack((target_x[j], target_y[j]), axis=-1)
                    target.append(tmp_target)
            else:
                target = []

            tmp_target_points.append(target)

        for k in range(self.actual_batchsize):
            tmp_label = self.exist_label[k]
            index = 0
            if tmp_label[0]==1:
                for zz in range(self.p.ground_truth_num):
                    if tmp_target_points[k][index][zz,0]>self.p.x_size or tmp_target_points[k][index][zz,0]<0 or tmp_target_points[k][index][zz,1]>self.p.y_size or tmp_target_points[k][index][zz,1]<0:
                        self.exist_label[k][0] = 0
                        break
                    else:
                        L2_target_points[k] = tmp_target_points[k][index]
                index = index + 1
            if tmp_label[1]==1:
                for zz in range(self.p.ground_truth_num):
                    if tmp_target_points[k][index][zz,0]>self.p.x_size or tmp_target_points[k][index][zz,0]<0 or tmp_target_points[k][index][zz,1]>self.p.y_size or tmp_target_points[k][index][zz,1]<0:
                        self.exist_label[k][1] = 0
                        break
                    else:
                        L1_target_points[k] = tmp_target_points[k][index]
                index = index + 1
            if tmp_label[2]==1:
                for zz in range(self.p.ground_truth_num):
                    if tmp_target_points[k][index][zz,0]>self.p.x_size or tmp_target_points[k][index][zz,0]<0 or tmp_target_points[k][index][zz,1]>self.p.y_size or tmp_target_points[k][index][zz,1]<0:
                        self.exist_label[k][2] = 0
                        break
                    else:
                        R1_target_points[k] = tmp_target_points[k][index]
                index = index + 1
            if tmp_label[3]==1:
                for zz in range(self.p.ground_truth_num):
                    if tmp_target_points[k][index][zz,0]>self.p.x_size or tmp_target_points[k][index][zz,0]<0 or tmp_target_points[k][index][zz,1]>self.p.y_size or tmp_target_points[k][index][zz,1]<0:
                        self.exist_label[k][3] = 0
                        break
                    else:
                        R2_target_points[k] = tmp_target_points[k][index]


            # test_image = self.inputs[k]
            # test_image = np.rollaxis(test_image, axis=2, start=0)
            # test_image = np.rollaxis(test_image, axis=2, start=0)
            # test_image = test_image.astype(np.uint8).copy()
            # lane_data = L2_target_points[k]
            # for m in range(len(lane_data)):
            #     test_image = cv2.circle(test_image, (int(lane_data[m, 0]), int(lane_data[m, 1])), 5, (0, 255, 0), -1)
            # lane_data = L1_target_points[k]
            # for m in range(len(lane_data)):
            #     test_image = cv2.circle(test_image, (int(lane_data[m, 0]), int(lane_data[m, 1])), 5, (0, 255, 0), -1)
            # lane_data = R2_target_points[k]
            # for m in range(len(lane_data)):
            #     test_image = cv2.circle(test_image, (int(lane_data[m, 0]), int(lane_data[m, 1])), 5, (0, 255, 0), -1)
            # lane_data = R1_target_points[k]
            # for m in range(len(lane_data)):
            #     test_image = cv2.circle(test_image, (int(lane_data[m, 0]), int(lane_data[m, 1])), 5, (0, 255, 0), -1)
            # cv2.imwrite('test_result/result'+str(k)+'.png', test_image)


        self.R2_Points = R2_target_points
        self.R1_Points = R1_target_points
        self.L2_Points = L2_target_points
        self.L1_Points = L1_target_points

        '''
        import matplotlib.pyplot as plt
import numpy as np
test_image = self.inputs[k]
test_image = np.rollaxis(test_image, axis=2, start=0)
test_image = np.rollaxis(test_image, axis=2, start=0)
test_image=test_image.astype(np.uint8).copy()
lane_data=L2_target_points[k]
for m in range(len(lane_data)):
    test_image = cv2.circle(test_image, (int(lane_data[m, 0]), int(lane_data[m, 1])), 5, (0,255,0), -1)
lane_data=L1_target_points[k]
for m in range(len(lane_data)):
    test_image = cv2.circle(test_image, (int(lane_data[m, 0]), int(lane_data[m, 1])), 5, (0,255,0), -1)
lane_data=R2_target_points[k]
for m in range(len(lane_data)):
    test_image = cv2.circle(test_image, (int(lane_data[m, 0]), int(lane_data[m, 1])), 5, (0,255,0), -1)
lane_data=R1_target_points[k]
for m in range(len(lane_data)):
    test_image = cv2.circle(test_image, (int(lane_data[m, 0]), int(lane_data[m, 1])), 5, (0,255,0), -1)
cv2.imwrite('test_result/result.png',test_image)



z = np.polyfit(R2_target_points[k][:,1],R2_target_points[k][:,0], 3)
f = np.poly1d(z)
        '''






