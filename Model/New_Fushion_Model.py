import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import Model.DSNT as DSTN
import time

class Lane_exist(nn.Module):
    def __init__(self, num_output, in_channel):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(in_channel, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()

        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(240, 128)
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
        output = output.view(-1, 240)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.sigmoid(output)

        return output

class Fushion_Lane_detection(nn.Module):
    def __init__(self,in_channels, point_feature_channels, point_number,encoder_model='resnet34',lane_fushion=True):
        super(Fushion_Lane_detection, self).__init__()
        self.point_feature_channels = point_feature_channels
        self.point_number = point_number
        self.lane_fushion = lane_fushion
        if encoder_model == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)
        elif encoder_model == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif encoder_model == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        elif encoder_model == 'MobileNet':
            self.MobileNet = models.mobilenet_v2(pretrained=True)
        if encoder_model != 'MobileNet':
            self.encoder = nn.Sequential(self.resnet.conv1,self.resnet.bn1,self.resnet.relu,self.resnet.maxpool,self.resnet.layer1,self.resnet.layer2,
                                     self.resnet.layer3,self.resnet.layer4)
        else:
            self.encoder = self.MobileNet.features
        self.index = []
        for i in range(point_feature_channels):
            for j in range(point_number):
                self.index.append(j * point_feature_channels + i)
        self.lane_exist = Lane_exist(num_output=4,in_channel=in_channels)
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

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.GroupNorm):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)

if __name__ == "__main__":
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True
    model = Fushion_Lane_detection(in_channels=512,point_feature_channels=32,point_number=15,lane_fushion=True,encoder_model='resnet34').cuda()
    model.eval()
    image = torch.randn(1, 3, 288, 800).cuda()

    t_all = 0
    for i in range(10):
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
    # a = torch.rand((1,480,25,9))
    #
    # for i in range(15):
    #     b = torch.index_select(a,dim=1,index=torch.arange(i*32,(i+1)*32))
    #
    # lane1,lane2,lane3,lane4,lane5,lane6,lane7,lane8,lane9,lane10,lane11,lane12,lane13,lane14,lane15 = torch.split(a,32,dim=1)
    # t_all = 0
    # for _ in range(100):
    #     t1 = time.time()
    #     lane = lane1 + lane2 + lane3 + lane1
    #     lane = lane1 + lane2 + lane3 + lane1
    #
    #     lane = lane1 + lane2 + lane3 + lane1
    #
    #     lane = lane1 + lane2 + lane3 + lane1
    #
    #     t2 = time.time()
    #     t_all = t_all + t2 - t1
    # print(t_all/100)