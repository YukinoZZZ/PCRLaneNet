#############################################################################################################
##
##  Parameters
##
#############################################################################################################
import numpy as np


class Parameters():
    alpha = 1
    n_epoch = 1000
    l_rate = 1e-2
    flood_level = 0.0
    weight_decay = 1e-4
    steps = [5,10,20,30]
    curve_threshold = 1e-2
    ground_truth_num = 15
    old_ground_truth_num = 15
    save_path = "/home/doujian/Desktop/model/culane_down/"
    save_path2 = "/data1/alpha-doujian/resnet101_models/"
    save_path3 = "/data1/alpha-doujian/10_points/"
    lr_decay_every = 4000
    model_path = ""
    # model_path = "savefile/"
    batch_size = 12
    point_feature_channel = 16
    x_size = 800
    y_size = 288
    resize_ratio = 8
    point_num = 15
    threshold_iou = 0.5
    line_width = 30

    image_H = 590
    image_W = 1640

    # loss function parameter

    # data loader parameter
    flip_ratio = 0.0
    translation_ratio = 0.6#0.6
    rotate_ratio = 0.6#0.6
    noise_ratio = 0.4#0.4
    intensity_ratio = 0.4#0.4
    shadow_ratio = 0.6#0.6
    scaling_ratio = 0.2#0.2

    train_root_url = "/home/doujian/Desktop/CULane/train_validate"
    test_root_url = "/home/doujian/Desktop/CULane/test/"

    # test parameter
    color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]




