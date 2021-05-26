import Model.DSNT as dsntnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.resnet import _resnet34
from Parameter import Parameters
import Model.DSNT as DSTN
from Model.drn import drn_c_26, drn_c_42, drn_d_54, drn_d_105
from Model.Erfnet import Encoder
import time


p = Parameters()


class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super(CoordRegressionNetwork, self).__init__()
        self.hm_conv = nn.Conv2d(512, n_locations, kernel_size=1, bias=False)

    def forward(self, fcn_out):
        # 1. Run the images through our FCN
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps


class Lane_exist(nn.Module):
    def __init__(self, num_output, Channel=4500):#4500
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(512, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()

        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(Channel, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.maxpool(output)
        # print(output.shape)
        b,c,h,w = output.shape
        output = output.view(-1, c*h*w)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.sigmoid(output)

        return output


class Lane_Detection_CPLUS(nn.Module):
    def __init__(self,encoder = 'drn_c_42'):
        super(Lane_Detection_CPLUS, self).__init__()
        # self.resnet34 = models.resnet34(pretrained=False)
        if encoder == 'drn_c_26':
            self.encoder = nn.Sequential(drn_c_26(pretrained=True).conv1, drn_c_26(pretrained=True).bn1, drn_c_26(pretrained=True).relu, drn_c_26(pretrained=True).layer1,
                                     drn_c_26(pretrained=True).layer2,drn_c_26(pretrained=True).layer3, drn_c_26(pretrained=True).layer4,drn_c_26(pretrained=True).layer5,
                                     drn_c_26(pretrained=True).layer6,drn_c_26(pretrained=True).layer7,drn_c_26(pretrained=True).layer8)
        if encoder == 'drn_c_42':
            self.encoder = nn.Sequential(drn_c_42(pretrained=True).conv1, drn_c_42(pretrained=True).bn1,
                                         drn_c_42(pretrained=True).relu, drn_c_42(pretrained=True).layer1,
                                         drn_c_42(pretrained=True).layer2, drn_c_42(pretrained=True).layer3,
                                         drn_c_42(pretrained=True).layer4, drn_c_42(pretrained=True).layer5,
                                         drn_c_42(pretrained=True).layer6, drn_c_42(pretrained=True).layer7,
                                         drn_c_42(pretrained=True).layer8
                                         )
        if encoder == 'drn_d_54':
            self.encoder = nn.Sequential(drn_d_54(pretrained=True).layer0, drn_d_54(pretrained=True).layer1,
                                         drn_d_54(pretrained=True).layer2, drn_d_54(pretrained=True).layer3,
                                         drn_d_54(pretrained=True).layer4, drn_d_54(pretrained=True).layer5,
                                         drn_d_54(pretrained=True).layer6, drn_d_54(pretrained=True).layer7,
                                         drn_d_54(pretrained=True).layer8
                                         )
        if encoder == 'resnet':
            self.encoder = nn.Sequential(_resnet34().conv1, _resnet34().bn1,
                                         _resnet34().relu, _resnet34().maxpool,
                                         _resnet34().layer1, _resnet34().layer2,
                                         _resnet34().layer3, _resnet34().layer4)
        self.lane_exist = Lane_exist(4)
        self.lane_point = CoordRegressionNetwork(n_locations=p.point_num * 4)

    def forward(self, input):
        encoder = self.encoder(input)
        pred_exist = self.lane_exist(encoder)
        Lane_Coor, Lane_HeapMaps = self.lane_point(encoder)
        R2Coors, R1Coors, L1Coors, L2Coors = torch.split(Lane_Coor, p.point_num, dim=1)
        R2HeapMaps, R1HeapMaps, L1HeapMaps, L2HeapMaps = torch.split(Lane_HeapMaps, p.point_num, dim=1)

        return pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps


class Fushion_Lane_detection(nn.Module):
    def __init__(self,in_channels, point_feature_channels, point_number,lane_fushion=True, encoder = 'resnet'):
        super(Fushion_Lane_detection, self).__init__()
        self.point_feature_channels = point_feature_channels
        self.point_number = point_number
        self.lane_fushion = lane_fushion
        if encoder == 'drn_c_26':
            self.encoder = nn.Sequential(drn_c_26(pretrained=True).conv1, drn_c_26(pretrained=True).bn1, drn_c_26(pretrained=True).relu, drn_c_26(pretrained=True).layer1,
                                     drn_c_26(pretrained=True).layer2,drn_c_26(pretrained=True).layer3, drn_c_26(pretrained=True).layer4,drn_c_26(pretrained=True).layer5,
                                     drn_c_26(pretrained=True).layer6,drn_c_26(pretrained=True).layer7,drn_c_26(pretrained=True).layer8)
        if encoder == 'drn_c_42':
            self.encoder = nn.Sequential(drn_c_42(pretrained=True).layer0, drn_c_42(pretrained=True).layer1,
                                         drn_c_42(pretrained=True).layer2, drn_c_42(pretrained=True).layer3,
                                         drn_c_42(pretrained=True).layer4, drn_c_42(pretrained=True).layer5,
                                         drn_c_42(pretrained=True).layer6, drn_c_42(pretrained=True).layer7,
                                         drn_c_42(pretrained=True).layer8
                                         )
        if encoder == 'resnet':
            self.encoder = nn.Sequential(_resnet34().conv1, _resnet34().bn1,
                                         _resnet34().relu, _resnet34().maxpool,
                                         _resnet34().layer1, _resnet34().layer2,
                                         _resnet34().layer3, _resnet34().layer4)
        self.index = []
        for i in range(point_feature_channels):
            for j in range(point_number):
                self.index.append(j * point_feature_channels + i)
        self.lane_exist = Lane_exist(4)
        self.lane_fushion_L2 = nn.Sequential(nn.Conv2d(point_feature_channels, point_feature_channels, kernel_size=5, stride=1,
                                         padding=2))
        self.lane_fushion_L1 = nn.Sequential(nn.Conv2d(point_feature_channels, point_feature_channels, kernel_size=5, stride=1,
                                         padding=2))
        self.lane_fushion_R1 = nn.Sequential(nn.Conv2d(point_feature_channels, point_feature_channels, kernel_size=5, stride=1,
                                         padding=2))
        self.lane_fushion_R2 = nn.Sequential(nn.Conv2d(point_feature_channels, point_feature_channels, kernel_size=5, stride=1,
                                         padding=2))
        self.L2_features = nn.Sequential(nn.Conv2d(in_channels, point_number*point_feature_channels, kernel_size=3, stride=1, padding=1),
                                         nn.GroupNorm(point_number, point_number*point_feature_channels),nn.ReLU())
        self.L1_features = nn.Sequential(nn.Conv2d(in_channels, point_number*point_feature_channels, kernel_size=3, stride=1, padding=1),
                                         nn.GroupNorm(point_number, point_number*point_feature_channels),nn.ReLU())
        self.R1_features = nn.Sequential(nn.Conv2d(in_channels, point_number*point_feature_channels, kernel_size=3, stride=1, padding=1),
                                         nn.GroupNorm(point_number, point_number*point_feature_channels),nn.ReLU())
        self.R2_features = nn.Sequential(nn.Conv2d(in_channels, point_number*point_feature_channels, kernel_size=3, stride=1, padding=1),
                                         nn.GroupNorm(point_number, point_number*point_feature_channels),nn.ReLU())
        self.final_conv_hm_L2 = nn.Conv2d(2*point_number * point_feature_channels, point_number, 1, groups=point_number)
        self.final_conv_hm_L1 = nn.Conv2d(2*point_number * point_feature_channels, point_number, 1, groups=point_number)
        self.final_conv_hm_R1 = nn.Conv2d(2*point_number * point_feature_channels, point_number, 1, groups=point_number)
        self.final_conv_hm_R2 = nn.Conv2d(2*point_number * point_feature_channels, point_number, 1, groups=point_number)


    def forward(self, input):
        global_feature = self.encoder(input)
        pred_exist = self.lane_exist(global_feature)
        L2_feature = self.L2_features(global_feature)
        L1_feature = self.L1_features(global_feature)
        R1_feature = self.R1_features(global_feature)
        R2_feature = self.R2_features(global_feature)

        ##lane fushion module
        permute_L2_feature = L2_feature[:, self.index, :, :]
        lane_feature_l2 = [None for _ in range(self.point_feature_channels)]
        for i in range(self.point_feature_channels):
            lane_feature_l2[i], _ = torch.max(
                permute_L2_feature[:, i * self.point_number:(i + 1) * self.point_number, :, :], dim=1, keepdim=True)
        L2_global_feature = torch.cat(lane_feature_l2, dim=1)

        permute_L1_feature = L1_feature[:, self.index, :, :]
        lane_feature_l1 = [None for _ in range(self.point_feature_channels)]
        for i in range(self.point_feature_channels):
            lane_feature_l1[i], _ = torch.max(
                permute_L1_feature[:, i * self.point_number:(i + 1) * self.point_number, :, :],
                dim=1, keepdim=True)
        L1_global_feature = torch.cat(lane_feature_l1, dim=1)

        permute_R1_feature = R1_feature[:, self.index, :, :]
        lane_feature_r1 = [None for _ in range(self.point_feature_channels)]
        for i in range(self.point_feature_channels):
            lane_feature_r1[i], _ = torch.max(
                permute_R1_feature[:, i * self.point_number:(i + 1) * self.point_number, :, :],
                dim=1, keepdim=True)
        R1_global_feature = torch.cat(lane_feature_r1, dim=1)

        permute_R2_feature = R2_feature[:, self.index, :, :]
        lane_feature_r2 = [None for _ in range(self.point_feature_channels)]
        for i in range(self.point_feature_channels):
            lane_feature_r2[i], _ = torch.max(
                permute_R2_feature[:, i * self.point_number:(i + 1) * self.point_number, :, :],
                dim=1, keepdim=True)
        R2_global_feature = torch.cat(lane_feature_r2, dim=1)

        if self.lane_fushion:
            # l2 lane fushion
            l2_lane_fushion = L2_global_feature + self.lane_fushion_L2(L1_global_feature)
            # l1 lane fushion
            l1_lane_fushion = L1_global_feature + self.lane_fushion_L1(L2_global_feature) + self.lane_fushion_L1(
                R1_global_feature)
            # r1 lane fushion
            r1_lane_fushion = R1_global_feature + self.lane_fushion_R1(R2_global_feature) + self.lane_fushion_R1(
                L1_global_feature)
            # r2 lane fushion
            r2_lane_fushion = R2_global_feature + self.lane_fushion_R2(R1_global_feature)
        else:
            l2_lane_fushion = L2_global_feature
            l1_lane_fushion = L1_global_feature
            r1_lane_fushion = R1_global_feature
            r2_lane_fushion = R2_global_feature

        ##lane point fushion
        lane_fushion_l2 = [None for _ in range(self.point_number)]
        lane_fushion_l1 = [None for _ in range(self.point_number)]
        lane_fushion_r1 = [None for _ in range(self.point_number)]
        lane_fushion_r2 = [None for _ in range(self.point_number)]

        for i in range(self.point_number):
            lane_fushion_l2[i] = torch.cat((L2_feature[:, i * self.point_feature_channels:(i + 1) * self.point_feature_channels],l2_lane_fushion),dim=1)
            lane_fushion_l1[i] = torch.cat((L1_feature[:, i * self.point_feature_channels:(i + 1) * self.point_feature_channels],l1_lane_fushion),dim=1)
            lane_fushion_r1[i] = torch.cat((R1_feature[:, i * self.point_feature_channels:(i + 1) * self.point_feature_channels],r1_lane_fushion),dim=1)
            lane_fushion_r2[i] = torch.cat((R2_feature[:, i * self.point_feature_channels:(i + 1) * self.point_feature_channels],r2_lane_fushion),dim=1)

        fushion_l2 = torch.cat(lane_fushion_l2,dim=1)
        fushion_l1 = torch.cat(lane_fushion_l1, dim=1)
        fushion_r1 = torch.cat(lane_fushion_r1, dim=1)
        fushion_r2 = torch.cat(lane_fushion_r2, dim=1)

        final_unnormalized_heatmaps_l2 = self.final_conv_hm_L2(fushion_l2)
        final_normalized_heatmaps_l2 = DSTN.flat_softmax(final_unnormalized_heatmaps_l2)
        final_coors_l2 = DSTN.dsnt(final_normalized_heatmaps_l2)

        final_unnormalized_heatmaps_l1 = self.final_conv_hm_L1(fushion_l1)
        final_normalized_heatmaps_l1 = DSTN.flat_softmax(final_unnormalized_heatmaps_l1)
        final_coors_l1 = DSTN.dsnt(final_normalized_heatmaps_l1)

        final_unnormalized_heatmaps_r1 = self.final_conv_hm_R1(fushion_r1)
        final_normalized_heatmaps_r1 = DSTN.flat_softmax(final_unnormalized_heatmaps_r1)
        final_coors_r1 = DSTN.dsnt(final_normalized_heatmaps_r1)

        final_unnormalized_heatmaps_r2 = self.final_conv_hm_R2(fushion_r2)
        final_normalized_heatmaps_r2 = DSTN.flat_softmax(final_unnormalized_heatmaps_r2)
        final_coors_r2 = DSTN.dsnt(final_normalized_heatmaps_r2)

        return pred_exist, final_coors_r2, final_normalized_heatmaps_r2, final_coors_r1, final_normalized_heatmaps_r1, \
               final_coors_l1, final_normalized_heatmaps_l1, final_coors_l2, final_normalized_heatmaps_l2

class Fushion_model(nn.Module):
    def __init__(self,in_channels, point_feature_channels, point_number, encoder = 'drn_d_105'):
        super(Fushion_model, self).__init__()
        self.point_num = point_number
        self.point_feature = point_feature_channels
        if encoder == 'ERFNet':
            self.encoder = Encoder()
        if encoder == 'drn_c_26':
            self.encoder = nn.Sequential(drn_c_26(pretrained=True).conv1, drn_c_26(pretrained=True).bn1,
                                         drn_c_26(pretrained=True).relu, drn_c_26(pretrained=True).layer1,
                                         drn_c_26(pretrained=True).layer2, drn_c_26(pretrained=True).layer3,
                                         drn_c_26(pretrained=True).layer4, drn_c_26(pretrained=True).layer5,
                                         drn_c_26(pretrained=True).layer6, drn_c_26(pretrained=True).layer7,
                                         drn_c_26(pretrained=True).layer8)
        if encoder == 'drn_c_42':
            self.encoder = nn.Sequential(drn_c_42(pretrained=True).conv1, drn_c_42(pretrained=True).bn1,
                                         drn_c_42(pretrained=True).relu, drn_c_42(pretrained=True).layer1,
                                         drn_c_42(pretrained=True).layer2, drn_c_42(pretrained=True).layer3,
                                         drn_c_42(pretrained=True).layer4, drn_c_42(pretrained=True).layer5,
                                         drn_c_42(pretrained=True).layer6, drn_c_42(pretrained=True).layer7,
                                         drn_c_42(pretrained=True).layer8
                                         )
        if encoder == 'resnet':
            self.encoder = nn.Sequential(_resnet34().conv1, _resnet34().bn1,
                                         _resnet34().relu, _resnet34().maxpool,
                                         _resnet34().layer1, _resnet34().layer2,
                                         _resnet34().layer3, _resnet34().layer4)

        if encoder == 'drn_d_54':
            self.encoder = nn.Sequential(drn_d_54(pretrained=True).layer0, drn_d_54(pretrained=True).layer1,
                                         drn_d_54(pretrained=True).layer2, drn_d_54(pretrained=True).layer3,
                                         drn_d_54(pretrained=True).layer4, drn_d_54(pretrained=True).layer5,
                                         drn_d_54(pretrained=True).layer6, drn_d_54(pretrained=True).layer7,
                                         drn_d_54(pretrained=True).layer8
                                         )

        if encoder == 'drn_d_105':
            self.encoder = nn.Sequential(drn_d_105(pretrained=True).layer0, drn_d_105(pretrained=True).layer1,
                                         drn_d_105(pretrained=True).layer2, drn_d_105(pretrained=True).layer3,
                                         drn_d_105(pretrained=True).layer4, drn_d_105(pretrained=True).layer5,
                                         drn_d_105(pretrained=True).layer6, drn_d_105(pretrained=True).layer7,
                                         drn_d_105(pretrained=True).layer8
                                         )

        self.L2_features = nn.Sequential(
            nn.Conv2d(in_channels, point_number * point_feature_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(point_number, point_number * point_feature_channels), nn.ReLU())
        self.L1_features = nn.Sequential(
            nn.Conv2d(in_channels, point_number * point_feature_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(point_number, point_number * point_feature_channels), nn.ReLU())
        self.R1_features = nn.Sequential(
            nn.Conv2d(in_channels, point_number * point_feature_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(point_number, point_number * point_feature_channels), nn.ReLU())
        self.R2_features = nn.Sequential(
            nn.Conv2d(in_channels, point_number * point_feature_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(point_number, point_number * point_feature_channels), nn.ReLU())

        self.conv_u = nn.Conv2d(point_feature_channels, point_feature_channels, kernel_size=5, stride=1,
                                         padding=2)

        self.conv_d = nn.Conv2d(point_feature_channels, point_feature_channels, kernel_size=5, stride=1,
                                              padding=2)
        #middle point feature supervision
        self.middle_conv_hm_L2 = nn.Conv2d(point_number * point_feature_channels, point_number, 1,
                                          groups=point_number)
        self.middle_conv_hm_L1 = nn.Conv2d(point_number * point_feature_channels, point_number, 1,
                                          groups=point_number)
        self.middle_conv_hm_R1 = nn.Conv2d(point_number * point_feature_channels, point_number, 1,
                                          groups=point_number)
        self.middle_conv_hm_R2 = nn.Conv2d(point_number * point_feature_channels, point_number, 1,
                                          groups=point_number)

        ##final point pred
        self.final_conv_hm_L2 = nn.Conv2d(point_number * point_feature_channels, point_number, 1,
                                          groups=point_number)
        self.final_conv_hm_L1 = nn.Conv2d(point_number * point_feature_channels, point_number, 1,
                                          groups=point_number)
        self.final_conv_hm_R1 = nn.Conv2d(point_number * point_feature_channels, point_number, 1,
                                          groups=point_number)
        self.final_conv_hm_R2 = nn.Conv2d(point_number * point_feature_channels, point_number, 1,
                                          groups=point_number)
        self.lane_exist = Lane_exist(4)

    def forward(self, input):
        global_feature = self.encoder(input)
        pred_exist = self.lane_exist(global_feature)
        L2_feature = self.L2_features(global_feature)
        L1_feature = self.L1_features(global_feature)
        R1_feature = self.R1_features(global_feature)
        R2_feature = self.R2_features(global_feature)

        middle_R2_feature = R2_feature.clone()
        middle_R1_feature = R1_feature.clone()
        middle_L1_feature = L1_feature.clone()
        middle_L2_feature = L2_feature.clone()
        #middle point feature supervision
        middle_unnormalized_heatmaps_l2 = self.middle_conv_hm_L2(middle_L2_feature)
        middle_normalized_heatmaps_l2 = DSTN.flat_softmax(middle_unnormalized_heatmaps_l2)
        middle_coors_l2 = DSTN.dsnt(middle_normalized_heatmaps_l2)

        middle_unnormalized_heatmaps_l1 = self.middle_conv_hm_L1(middle_L1_feature)
        middle_normalized_heatmaps_l1 = DSTN.flat_softmax(middle_unnormalized_heatmaps_l1)
        middle_coors_l1 = DSTN.dsnt(middle_normalized_heatmaps_l1)

        middle_unnormalized_heatmaps_r1 = self.middle_conv_hm_R1(middle_R1_feature)
        middle_normalized_heatmaps_r1 = DSTN.flat_softmax(middle_unnormalized_heatmaps_r1)
        middle_coors_r1 = DSTN.dsnt(middle_normalized_heatmaps_r1)

        middle_unnormalized_heatmaps_r2 = self.middle_conv_hm_R2(middle_R2_feature)
        middle_normalized_heatmaps_r2 = DSTN.flat_softmax(middle_unnormalized_heatmaps_r2)
        middle_coors_r2 = DSTN.dsnt(middle_normalized_heatmaps_r2)
        ## point feature fusion
        for i in range(1,self.point_num):
            L2_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :] = \
                self.conv_d(L2_feature[:, (i - 1) * self.point_feature:i * self.point_feature, :, :]) +\
                            L2_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :]
            L1_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :] = \
                self.conv_d(L1_feature[:, (i - 1) * self.point_feature:i * self.point_feature, :, :]) + \
                L1_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :]
            R1_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :] = \
                self.conv_d(R1_feature[:, (i - 1) * self.point_feature:i * self.point_feature, :, :]) + \
                R1_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :]
            R2_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :] = \
                self.conv_d(R2_feature[:, (i - 1) * self.point_feature:i * self.point_feature, :, :]) + \
                R2_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :]

        for i in range(self.point_num-2,0,-1):
            L2_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :] = \
            self.conv_u(L2_feature[:, (i + 1) * self.point_feature:(i + 2) * self.point_feature, :, :]) + \
                        L2_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :]
            L1_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :] = \
                self.conv_u(L1_feature[:, (i + 1) * self.point_feature:(i + 2) * self.point_feature, :, :]) + \
                L1_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :]
            R1_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :] = \
                self.conv_u(R1_feature[:, (i + 1) * self.point_feature:(i + 2) * self.point_feature, :, :]) + \
                R1_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :]
            R2_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :] = \
                self.conv_u(R2_feature[:, (i + 1) * self.point_feature:(i + 2) * self.point_feature, :, :]) + \
                R2_feature[:, i * self.point_feature:(i + 1) * self.point_feature, :, :]

        final_unnormalized_heatmaps_l2 = self.final_conv_hm_L2(L2_feature)
        final_normalized_heatmaps_l2 = DSTN.flat_softmax(final_unnormalized_heatmaps_l2)
        final_coors_l2 = DSTN.dsnt(final_normalized_heatmaps_l2)

        final_unnormalized_heatmaps_l1 = self.final_conv_hm_L1(L1_feature)
        final_normalized_heatmaps_l1 = DSTN.flat_softmax(final_unnormalized_heatmaps_l1)
        final_coors_l1 = DSTN.dsnt(final_normalized_heatmaps_l1)

        final_unnormalized_heatmaps_r1 = self.final_conv_hm_R1(R1_feature)
        final_normalized_heatmaps_r1 = DSTN.flat_softmax(final_unnormalized_heatmaps_r1)
        final_coors_r1 = DSTN.dsnt(final_normalized_heatmaps_r1)

        final_unnormalized_heatmaps_r2 = self.final_conv_hm_R2(R2_feature)
        final_normalized_heatmaps_r2 = DSTN.flat_softmax(final_unnormalized_heatmaps_r2)
        final_coors_r2 = DSTN.dsnt(final_normalized_heatmaps_r2)

        return pred_exist, final_coors_r2, final_normalized_heatmaps_r2, final_coors_r1, final_normalized_heatmaps_r1, \
               final_coors_l1, final_normalized_heatmaps_l1, final_coors_l2, final_normalized_heatmaps_l2, middle_coors_r2, \
               middle_normalized_heatmaps_r2, middle_coors_r1, middle_normalized_heatmaps_r1, \
               middle_coors_l1, middle_normalized_heatmaps_l1, middle_coors_l2, middle_normalized_heatmaps_l2
        # return pred_exist, final_coors_r2, final_normalized_heatmaps_r2, final_coors_r1, final_normalized_heatmaps_r1, \
        #        final_coors_l1, final_normalized_heatmaps_l1, final_coors_l2, final_normalized_heatmaps_l2



def feature_display(lane_feature):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    lane_feature = torch.max(lane_feature,dim=1)[0]
    lane_feature = DSTN.flat_softmax(lane_feature)
    lane_feature = lane_feature.cpu().detach().numpy()
    ax = sns.heatmap(lane_feature[0],cmap='rainbow')
    plt.savefig('test_result/final_l1.png')


def feature_display2(lane_feature):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    lane_feature = torch.sum(lane_feature.pow(2), dim=1)
    lane_feature = DSTN.flat_softmax(lane_feature)
    lane_feature = lane_feature.cpu().detach().numpy()
    ax = sns.heatmap(lane_feature[0],cmap='rainbow')
    plt.savefig('test_result/final_l1.png')






if __name__ == "__main__":
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True
    model = Fushion_model(in_channels=128,point_feature_channels=32,point_number=15).cuda()
    model.eval()
    image = torch.randn(1, 3, 288, 800).cuda()

    t_all = 0
    for i in range(5):
        pred_exist, final_coors_r2, final_normalized_heatmaps_r2, final_coors_r1, final_normalized_heatmaps_r1, \
        final_coors_l1, final_normalized_heatmaps_l1, final_coors_l2, final_normalized_heatmaps_l2, middle_coors_r2, \
        middle_normalized_heatmaps_r2, middle_coors_r1, middle_normalized_heatmaps_r1, \
        middle_coors_l1, middle_normalized_heatmaps_l1, middle_coors_l2, middle_normalized_heatmaps_l2 = model(
            image)

    for i in range(100):
        t1 = time.time()
        pred_exist, final_coors_r2, final_normalized_heatmaps_r2, final_coors_r1, final_normalized_heatmaps_r1, \
        final_coors_l1, final_normalized_heatmaps_l1, final_coors_l2, final_normalized_heatmaps_l2, middle_coors_r2, \
        middle_normalized_heatmaps_r2, middle_coors_r1, middle_normalized_heatmaps_r1, \
        middle_coors_l1, middle_normalized_heatmaps_l1, middle_coors_l2, middle_normalized_heatmaps_l2 = model(
            image)
        t2 = time.time()
        t_all += t2-t1

    print('avg_time:',t_all/100)
    print('avg_fps:',100/t_all)



