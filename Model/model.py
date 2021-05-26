from Model.deeplabv2 import DeepLabV2
import Model.DSNT as dsntnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet
import torchvision.models as models
from Parameter import Parameters
from Model.se_resnet import SEBasicBlock
import time

p = Parameters()


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """


    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

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

class CoordRegressionNetwork_V2(nn.Module):
    def __init__(self, n_locations):
        super(CoordRegressionNetwork_V2, self).__init__()
        self.hm_conv = nn.Conv2d(512, n_locations, kernel_size=1, bias=False)

    def forward(self, fcn_out):
        # 1. Run the images through our FCN
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps, unnormalized_heatmaps

class CoordRegressionNetwork_Fushion(nn.Module):
    def __init__(self, n_locations):
        super(CoordRegressionNetwork_Fushion, self).__init__()
        self.hm_conv = nn.Conv2d(512, n_locations, kernel_size=1, bias=False)
        self.heatmap_feature = nn.Conv2d(n_locations,n_locations,kernel_size=5,padding=2,bias=False,groups=n_locations)
        self.fushion_conv = nn.Conv2d(n_locations,n_locations,kernel_size=1,bias=False,groups=n_locations)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fcn_out):
        # 1. Run the images through our FCN
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the intermedi-coordinates
        interme_coords = dsntnn.dsnt(heatmaps)
        #5. Fushion Point Feature
        heatmaps = self.heatmap_feature(heatmaps)
        B, C, H, W = heatmaps.size()

        feat_a = heatmaps.view(B,C,H*W)
        feat_a_transpose = heatmaps.view(B, C, H * W).permute(0, 2, 1)
        relation = torch.bmm(feat_a, feat_a_transpose)
        relation_new = torch.max(relation, dim=-1, keepdim=True)[0].expand_as(relation) - relation
        relation = self.softmax(relation_new)


        feat_e = torch.bmm(relation, feat_a).view(B, C, H, W)
        feature_fushion = feat_e + heatmaps

        heatmaps = self.fushion_conv(feature_fushion)

        heatmaps = dsntnn.flat_softmax(heatmaps)

        final_coords = dsntnn.dsnt(heatmaps)

        return final_coords, interme_coords, heatmaps


class Fushion(nn.Module):
    def __init__(self, nlocations):
        super(Fushion, self).__init__()
        self.L2_feature = _ConvBnReLU(512, p.point_num * p.point_feature_channel, 3, 1, 1, 1)
        self.L2_golbal_fushion = nn.Conv2d(2,1,3,1,1,bias=False)
        self.L2_fushion = nn.ModuleList()
        self.L1_feature = _ConvBnReLU(512, p.point_num * p.point_feature_channel, 3, 1, 1, 1)
        self.L1_golbal_fushion = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.L1_fushion = nn.ModuleList()
        self.R1_feature = _ConvBnReLU(512, p.point_num * p.point_feature_channel, 3, 1, 1, 1)
        self.R1_golbal_fushion = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.R1_fushion = nn.ModuleList()
        self.R2_feature = _ConvBnReLU(512, p.point_num * p.point_feature_channel, 3, 1, 1, 1)
        self.R2_golbal_fushion = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.R2_fushion = nn.ModuleList()
        self.neighbors = []
        self.softmax = nn.Softmax(dim=-1)
        for i in range(p.point_num):
            neighbor = []
            if i-1>0:
                neighbor.append(i-2)
            if i>0:
                neighbor.append(i-1)
            if i<p.point_num-1:
                neighbor.append(i+1)
            if i<p.point_num-2:
                neighbor.append(i + 2)
            self.neighbors.append(neighbor)
        for i in range(p.point_num):
            neighbor = self.neighbors[i]
            fushion = nn.ModuleList()
            for _ in range(len(neighbor)):
                fushion.append(nn.Sequential(nn.Conv2d(p.point_feature_channel,p.point_feature_channel,5,1,2,groups=p.point_feature_channel)))
            self.L2_fushion.append(fushion)
            self.L1_fushion.append(fushion)
            self.R1_fushion.append(fushion)
            self.R2_fushion.append(fushion)
        self.heatmap_conv_L2 = nn.Conv2d((p.point_feature_channel+1)*p.point_num,p.point_num,1,groups=p.point_num)
        self.heatmap_conv_L1 = nn.Conv2d((p.point_feature_channel+1) * p.point_num, p.point_num, 1, groups=p.point_num)
        self.heatmap_conv_R1 = nn.Conv2d((p.point_feature_channel+1) * p.point_num, p.point_num, 1, groups=p.point_num)
        self.heatmap_conv_R2 = nn.Conv2d((p.point_feature_channel+1) * p.point_num, p.point_num, 1, groups=p.point_num)

        self.orgin_heatmap_conv_L2 = nn.Conv2d(p.point_feature_channel * p.point_num, p.point_num, 1, groups=p.point_num)
        self.orgin_heatmap_conv_L1 = nn.Conv2d(p.point_feature_channel * p.point_num, p.point_num, 1, groups=p.point_num)
        self.orgin_heatmap_conv_R1 = nn.Conv2d(p.point_feature_channel * p.point_num, p.point_num, 1, groups=p.point_num)
        self.orgin_heatmap_conv_R2 = nn.Conv2d(p.point_feature_channel * p.point_num, p.point_num, 1, groups=p.point_num)

    def forward(self, encoder):
        L2_feature = self.L2_feature(encoder)
        L1_feature = self.L1_feature(encoder)
        R1_feature = self.R1_feature(encoder)
        R2_feature = self.R2_feature(encoder)

        # POINT FUSHION
        t1 = time.time()
        tmp_L2_fushion = [None for _ in range(p.point_num)]
        for i in range(p.point_num):
            tmp_L2_fushion[i] = L2_feature[:, i * p.point_feature_channel:(i + 1) * p.point_feature_channel]
            for j, point_index in enumerate(self.neighbors[i]):
                tmp_L2_fushion[i] = tmp_L2_fushion[i] + self.L2_fushion[i][j](
                    L2_feature[:, point_index * p.point_feature_channel:(point_index + 1) * p.point_feature_channel])
        L2_fushion = torch.cat(tmp_L2_fushion, dim=1)

        tmp_L1_fushion = [None for _ in range(p.point_num)]
        for i in range(p.point_num):
            tmp_L1_fushion[i] = L1_feature[:, i * p.point_feature_channel:(i + 1) * p.point_feature_channel]
            for j, point_index in enumerate(self.neighbors[i]):
                tmp_L1_fushion[i] = tmp_L1_fushion[i] + self.L1_fushion[i][j](
                    L1_feature[:, point_index * p.point_feature_channel:(point_index + 1) * p.point_feature_channel])
        L1_fushion = torch.cat(tmp_L1_fushion, dim=1)

        tmp_R1_fushion = [None for _ in range(p.point_num)]
        for i in range(p.point_num):
            tmp_R1_fushion[i] = R1_feature[:, i * p.point_feature_channel:(i + 1) * p.point_feature_channel]
            for j, point_index in enumerate(self.neighbors[i]):
                tmp_R1_fushion[i] = tmp_R1_fushion[i] + self.R1_fushion[i][j](
                    R1_feature[:, point_index * p.point_feature_channel:(point_index + 1) * p.point_feature_channel])
        R1_fushion = torch.cat(tmp_R1_fushion, dim=1)

        tmp_R2_fushion = [None for _ in range(p.point_num)]
        for i in range(p.point_num):
            tmp_R2_fushion[i] = R2_feature[:, i * p.point_feature_channel:(i + 1) * p.point_feature_channel]
            for j, point_index in enumerate(self.neighbors[i]):
                tmp_R2_fushion[i] = tmp_R2_fushion[i] + self.R2_fushion[i][j](
                    R2_feature[:, point_index * p.point_feature_channel:(point_index + 1) * p.point_feature_channel])
        R2_fushion = torch.cat(tmp_R2_fushion, dim=1)
        t2 = time.time()
        print(t2-t1)

        #LANE FUSHION
        L2_avg = torch.mean(L2_fushion,dim=1,keepdim=True)
        L2_max,_ = torch.max(L2_fushion,dim=1,keepdim=True)
        L2_global_feature = self.L2_golbal_fushion(torch.cat([L2_avg,L2_max],dim=1))
        L1_avg = torch.mean(L1_fushion, dim=1,keepdim=True)
        L1_max,_ = torch.max(L1_fushion, dim=1,keepdim=True)
        L1_global_feature = self.L1_golbal_fushion(torch.cat([L1_avg, L1_max], dim=1))
        R1_avg = torch.mean(R1_fushion, dim=1,keepdim=True)
        R1_max,_ = torch.max(R1_fushion, dim=1,keepdim=True)
        R1_global_feature = self.R1_golbal_fushion(torch.cat([R1_avg, R1_max], dim=1))
        R2_avg = torch.mean(R2_fushion, dim=1,keepdim=True)
        R2_max,_ = torch.max(R2_fushion, dim=1,keepdim=True)
        R2_global_feature = self.R2_golbal_fushion(torch.cat([R2_avg, R2_max], dim=1))
        All_feature = torch.cat([L2_global_feature,L1_global_feature,R1_global_feature,R2_global_feature],dim=1)
        B,C,H,W = All_feature.size()
        feat_a = All_feature.view(B,C,H*W)
        feat_a_transpose = All_feature.view(B,C,H*W).permute(0,2,1)
        attention = torch.bmm(feat_a,feat_a_transpose)
        attention_new = torch.max(attention,dim=-1,keepdim=True)[0].expand_as(attention)-attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention,feat_a).view(B,C,H,W)
        fushion_global = feat_e + All_feature

        fushion_global_L2 = [None for _ in range(p.point_num)]
        for i in range(p.point_num):
            fushion_global_L2[i] = torch.cat([fushion_global[:,0:1,:,:],L2_fushion[:,i*p.point_feature_channel:(i+1)*p.point_feature_channel,:,:]],dim=1)
        L2_fushion_global_feature = torch.cat(fushion_global_L2,dim=1)
        fushion_global_L1 = [None for _ in range(p.point_num)]
        for i in range(p.point_num):
            fushion_global_L1[i] = torch.cat([fushion_global[:, 1:2, :, :], L1_fushion[:, i * p.point_feature_channel:(i + 1) * p.point_feature_channel, :, :]],
                                             dim=1)
        L1_fushion_global_feature = torch.cat(fushion_global_L1, dim=1)
        fushion_global_R1 = [None for _ in range(p.point_num)]
        for i in range(p.point_num):
            fushion_global_R1[i] = torch.cat([fushion_global[:, 2:3, :, :], R1_fushion[:, i * p.point_feature_channel:(i + 1) * p.point_feature_channel, :, :]],
                                             dim=1)
        R1_fushion_global_feature = torch.cat(fushion_global_R1, dim=1)
        fushion_global_R2 = [None for _ in range(p.point_num)]
        for i in range(p.point_num):
            fushion_global_R2[i] = torch.cat([fushion_global[:, 3:4, :, :], R2_fushion[:, i * p.point_feature_channel:(i + 1) * p.point_feature_channel, :, :]],
                                             dim=1)
        R2_fushion_global_feature = torch.cat(fushion_global_R2, dim=1)

        L2_unnormal_heatmap = self.heatmap_conv_L2(L2_fushion_global_feature)
        fushion_heatmaps_L2 = dsntnn.flat_softmax(L2_unnormal_heatmap)
        final_Coor_L2 = dsntnn.dsnt(fushion_heatmaps_L2)

        L1_unnormal_heatmap = self.heatmap_conv_L1(L1_fushion_global_feature)
        fushion_heatmaps_L1 = dsntnn.flat_softmax(L1_unnormal_heatmap)
        final_Coor_L1 = dsntnn.dsnt(fushion_heatmaps_L1)

        R1_unnormal_heatmap = self.heatmap_conv_R1(R1_fushion_global_feature)
        fushion_heatmaps_R1 = dsntnn.flat_softmax(R1_unnormal_heatmap)
        final_Coor_R1 = dsntnn.dsnt(fushion_heatmaps_R1)

        R2_unnormal_heatmap = self.heatmap_conv_R2(R2_fushion_global_feature)
        fushion_heatmaps_R2 = dsntnn.flat_softmax(R2_unnormal_heatmap)
        final_Coor_R2 = dsntnn.dsnt(fushion_heatmaps_R2)

        orgin_unnormal_heatmap_L2 = self.orgin_heatmap_conv_L2(L2_feature)
        L2HeatMaps = dsntnn.flat_softmax(orgin_unnormal_heatmap_L2)
        L2Coors = dsntnn.dsnt(L2HeatMaps)

        orgin_unnormal_heatmap_L1 = self.orgin_heatmap_conv_L1(L1_feature)
        L1HeatMaps = dsntnn.flat_softmax(orgin_unnormal_heatmap_L1)
        L1Coors = dsntnn.dsnt(L1HeatMaps)

        orgin_unnormal_heatmap_R1 = self.orgin_heatmap_conv_R1(R1_feature)
        R1HeatMaps = dsntnn.flat_softmax(orgin_unnormal_heatmap_R1)
        R1Coors = dsntnn.dsnt(R1HeatMaps)

        orgin_unnormal_heatmap_R2 = self.orgin_heatmap_conv_R2(R2_feature)
        R2HeatMaps = dsntnn.flat_softmax(orgin_unnormal_heatmap_R2)
        R2Coors = dsntnn.dsnt(R2HeatMaps)





        return fushion_heatmaps_R2, final_Coor_R2, R2Coors, R2HeatMaps, fushion_heatmaps_R1, final_Coor_R1, R1Coors, R1HeatMaps,\
               fushion_heatmaps_L1, final_Coor_L1, L1Coors, L1HeatMaps, fushion_heatmaps_L2, final_Coor_L2, L2Coors, L2HeatMaps














class Lane_exist(nn.Module):
    def __init__(self, num_output, Channel=240):
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


class Lane_Detection(nn.Module):
    def __init__(self):
        super(Lane_Detection, self).__init__()
        self.resnet34 = models.resnet34(pretrained=False)
        # self.encoder = nn.Sequential(models.resnet34(pretrained=True).conv1, models.resnet34(pretrained=True).bn1, models.resnet34(pretrained=True).relu, models.resnet34(pretrained=True).maxpool,
        #                              models.resnet34(pretrained=True).layer1, models.resnet34(pretrained=True).layer2, models.resnet34(pretrained=True).layer3,
        #                              models.resnet34(pretrained=True).layer4)
        self.encoder = nn.Sequential(self.resnet34.conv1, self.resnet34.bn1,
                                     self.resnet34.relu, self.resnet34.maxpool,
                                     self.resnet34.layer1, self.resnet34.layer2,
                                     self.resnet34.layer3, self.resnet34.layer4)
        self.lane_exist = Lane_exist(4)
        self.lane_point = CoordRegressionNetwork(n_locations=p.point_num * 4)

    def forward(self, input):
        encoder = self.encoder(input)
        pred_exist = self.lane_exist(encoder)
        Lane_Coor, Lane_HeapMaps = self.lane_point(encoder)
        R2Coors, R1Coors, L1Coors, L2Coors = torch.split(Lane_Coor, p.point_num, dim=1)
        R2HeapMaps, R1HeapMaps, L1HeapMaps, L2HeapMaps = torch.split(Lane_HeapMaps, p.point_num, dim=1)
        '''
        feature visual
        
import numpy as np
import matplotlib.pyplot as plt
feature = L2HeapMaps + L1HeapMaps + R1HeapMaps + R2HeapMaps
feature = feature.cpu().detach().numpy()
feature = feature[0,:,:,:]
feature_max = np.max(feature,axis=0)
feature_max = feature_max**1
feature_max = np.round(feature_max*255)
plt.imshow(feature_max), time_spent

import numpy as np
import matplotlib.pyplot as plt
feature = (L2HeapMaps + L1HeapMaps + R1HeapMaps + R2HeapMaps)
feature = feature.cpu().detach().numpy()
feature = feature[0,:,:,:]
feature = np.abs(feature)
feature = feature**1
feature_sum = np.sum(feature,axis=0)
plt.imshow(feature_sum,cmap=plt.get_cmap('hot'))

import seaborn as sns


        '''

        return pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps


class resnet_lane_detection(nn.Module):
    def __init__(self):
        super(resnet_lane_detection, self).__init__()
        self.resnet34 = models.resnet34(pretrained=False)
        self.encoder = nn.Sequential(self.resnet34.conv1,self.resnet34.bn1,self.resnet34.relu,self.resnet34.maxpool,self.resnet34.layer1,self.resnet34.layer2,self.resnet34.layer3,self.resnet34.layer4)
        self.lane_exist = Lane_exist(4)
        self.fushion_layer = Fushion(nlocations=p.point_num)
    def forward(self, input):
        encoder = self.encoder(input)
        pred_exist = self.lane_exist(encoder)

        fushion_heatmaps_R2, final_Coor_R2, R2Coors, R2HeatMaps, fushion_heatmaps_R1, final_Coor_R1, R1Coors, R1HeatMaps, \
        fushion_heatmaps_L1, final_Coor_L1, L1Coors, L1HeatMaps, fushion_heatmaps_L2, final_Coor_L2, L2Coors, L2HeatMaps = self.fushion_layer(encoder)

        return pred_exist, fushion_heatmaps_R2, final_Coor_R2, R2Coors, R2HeatMaps, fushion_heatmaps_R1, final_Coor_R1, R1Coors, R1HeatMaps,\
               fushion_heatmaps_L1, final_Coor_L1, L1Coors, L1HeatMaps, fushion_heatmaps_L2, final_Coor_L2, L2Coors, L2HeatMaps

class resnet_lane_detection_simple(nn.Module):
    def __init__(self):
        super(resnet_lane_detection_simple, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        self.encoder = nn.Sequential(self.resnet34.conv1, self.resnet34.bn1, self.resnet34.relu, self.resnet34.maxpool,
                                         self.resnet34.layer1, self.resnet34.layer2, self.resnet34.layer3,
                                         self.resnet34.layer4)
        self.lane_exist = Lane_exist(4)



        self.lane_point = CoordRegressionNetwork(n_locations=p.point_num*4)
    def forward(self, input):
        encoder = self.encoder(input)
        pred_exist = self.lane_exist(encoder)
        Lane_Coor, Lane_HeapMaps = self.lane_point(encoder)
        R2Coors, R1Coors, L1Coors, L2Coors = torch.split(Lane_Coor,p.point_num,dim=1)
        R2HeapMaps, R1HeapMaps, L1HeapMaps, L2HeapMaps = torch.split(Lane_HeapMaps,p.point_num,dim=1)
        return pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps


class SE_Resnet_Model(nn.Module):
    def __init__(self):
        super(SE_Resnet_Model, self).__init__()
        self.resnet34 = ResNet(SEBasicBlock, [3, 4, 6, 3])
        self.encoder = nn.Sequential(self.resnet34.conv1, self.resnet34.bn1, self.resnet34.relu, self.resnet34.maxpool,
                                     self.resnet34.layer1, self.resnet34.layer2, self.resnet34.layer3,
                                     self.resnet34.layer4)
        self.lane_exist = Lane_exist(4)
        self.lane_point = CoordRegressionNetwork(n_locations=p.point_num * 4)

    def forward(self, input):
        encoder = self.encoder(input)
        pred_exist = self.lane_exist(encoder)
        Lane_Coor, Lane_HeapMaps = self.lane_point(encoder)
        R2Coors, R1Coors, L1Coors, L2Coors = torch.split(Lane_Coor, p.point_num, dim=1)
        R2HeapMaps, R1HeapMaps, L1HeapMaps, L2HeapMaps = torch.split(Lane_HeapMaps, p.point_num, dim=1)

        return pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps

class Lane_Detection_CPLUS(nn.Module):
    def __init__(self):
        super(Lane_Detection_CPLUS, self).__init__()
        # self.resnet34 = models.resnet34(pretrained=False)
        self.encoder = nn.Sequential(models.resnet34(pretrained=True).conv1, models.resnet34(pretrained=True).bn1, models.resnet34(pretrained=True).relu, models.resnet34(pretrained=True).maxpool,
                                     models.resnet34(pretrained=True).layer1, models.resnet34(pretrained=True).layer2, models.resnet34(pretrained=True).layer3,
                                     models.resnet34(pretrained=True).layer4)
        # self.encoder = nn.Sequential(self.resnet34.conv1, self.resnet34.bn1,
        #                              self.resnet34.relu, self.resnet34.maxpool,
        #                              self.resnet34.layer1, self.resnet34.layer2,
        #                              self.resnet34.layer3, self.resnet34.layer4)
        self.lane_exist = Lane_exist(4)
        self.lane_point = CoordRegressionNetwork(n_locations=p.point_num * 4)

    def forward(self, input):
        encoder = self.encoder(input)
        pred_exist = self.lane_exist(encoder)
        Lane_Coor, Lane_HeapMaps = self.lane_point(encoder)
        R2Coors, R1Coors, L1Coors, L2Coors = torch.split(Lane_Coor, p.point_num, dim=1)
        R2HeapMaps, R1HeapMaps, L1HeapMaps, L2HeapMaps = torch.split(Lane_HeapMaps, p.point_num, dim=1)

        return pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps


class Lane_Detection_Mobile(nn.Module):
    def __init__(self):
        super(Lane_Detection_Mobile, self).__init__()
        # self.resnet34 = models.resnet34(pretrained=False)
        self.encoder = nn.Sequential(models.resnet18(pretrained=True).conv1, models.resnet18(pretrained=True).bn1, models.resnet18(pretrained=True).relu, models.resnet18(pretrained=True).maxpool,
                                     models.resnet18(pretrained=True).layer1, models.resnet18(pretrained=True).layer2, models.resnet18(pretrained=True).layer3,
                                     models.resnet18(pretrained=True).layer4)
        # self.encoder = nn.Sequential(self.resnet34.conv1, self.resnet34.bn1,
        #                              self.resnet34.relu, self.resnet34.maxpool,
        #                              self.resnet34.layer1, self.resnet34.layer2,
        #                              self.resnet34.layer3, self.resnet34.layer4)
        self.lane_exist = Lane_exist(4)
        self.lane_point = CoordRegressionNetwork(n_locations=p.point_num * 4)

    def forward(self, input):
        encoder = self.encoder(input)
        pred_exist = self.lane_exist(encoder)
        Lane_Coor, Lane_HeapMaps = self.lane_point(encoder)
        R2Coors, R1Coors, L1Coors, L2Coors = torch.split(Lane_Coor, p.point_num, dim=1)
        R2HeapMaps, R1HeapMaps, L1HeapMaps, L2HeapMaps = torch.split(Lane_HeapMaps, p.point_num, dim=1)

        return pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps




if __name__ == "__main__":
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True
    model = Lane_Detection_CPLUS().cuda()
    model.eval()
    image = torch.randn(1, 3, 288, 800).cuda()

    t_all = 0
    for i in range(20):
        pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps = model(
            image)

    for i in range(100):
        t1 = time.time()
        pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps = model(
            image)
        t2 = time.time()
        t_all += t2-t1

    print('avg_time:',t_all/100)
    print('avg_fps:',100/t_all)



