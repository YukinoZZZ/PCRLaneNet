import Model.DSNT as dsntnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import Model.DSNT as DSTN
from Model.drn import drn_c_26, drn_c_42, drn_d_54

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
        self.lane_exist = Lane_exist(4)
        self.lane_point = CoordRegressionNetwork(n_locations=15 * 4)

    def forward(self, input):
        encoder = self.encoder(input)
        pred_exist = self.lane_exist(encoder)
        Lane_Coor, Lane_HeapMaps = self.lane_point(encoder)
        R2Coors, R1Coors, L1Coors, L2Coors = torch.split(Lane_Coor, 15, dim=1)
        R2HeapMaps, R1HeapMaps, L1HeapMaps, L2HeapMaps = torch.split(Lane_HeapMaps, 15, dim=1)

        return pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps


def load_GPUS(model,model_path):
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def getValue(x):
    '''Convert Torch tensor/variable to numpy array/python scalar
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    elif isinstance(x, torch.autograd.Variable):
        x = x.data.cpu().detach().numpy()
    if x.size == 1:
        x = x.item()
    return x


if __name__ == "__main__":
    ##load model
    import os
    import cv2
    import numpy as np
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    torch.backends.cudnn.benchmark = True
    model = Lane_Detection_CPLUS().cuda()
    model_path = '/data1/alpha-doujian/save_model_wp/dilated_resnet34_nofusion/12_0.37931859493255615_SGD_lane_detection_network.pkl'
    model = load_GPUS(model,model_path)
    ##data load
    image_path = '/home/doujian/Desktop/CULane/train_validate/driver_23_30frame/05151643_0420.MP4/01620.jpg'
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (800, 288))
    raw_image = np.rollaxis(raw_image, axis=2, start=0) / 255.0
    image_size = np.array([800, 288]).reshape(1, 2)
    ratio_w = 800 * 1.0 / 1640
    ratio_h = 288 * 1.0 / 590
    raw_image = np.array(raw_image).reshape(1, 3, 288, 800)
    raw_image = torch.Tensor(raw_image).cuda()
    image_data = Variable(raw_image)
    ##model test
    model.eval()
    pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps = model(image_data)
    ##show the result
    tmp_image_data = getValue(raw_image[0])
    tmp_image_data = np.rollaxis(tmp_image_data, axis=2, start=0)
    tmp_image_data = np.rollaxis(tmp_image_data, axis=2, start=0)
    tmp_image_data = tmp_image_data * 255
    b, g, r = cv2.split(tmp_image_data)
    tmp_image_data = cv2.merge([r, g, b])
    image = tmp_image_data.astype(np.uint8).copy()
    pred = pred_exist[0]>0.5

    if pred[3] == 1:
        tmp_R2Coors = getValue(R2Coors[0])
        tmp_R2Coors = tmp_R2Coors.reshape(15, 2)
        tmp_R2Coors = 0.5 * ((tmp_R2Coors + 1) * image_size - 1)
        for index in range(15):
            image = cv2.circle(image, (int(tmp_R2Coors[index, 0]), int(tmp_R2Coors[index, 1])), 5, (255,0,0),
                               -1)
    if pred[2] == 1:
        tmp_R1Coors = getValue(R1Coors[0])
        tmp_R1Coors = tmp_R1Coors.reshape(15, 2)
        tmp_R1Coors = 0.5 * ((tmp_R1Coors + 1) * image_size - 1)
        for index in range(15):
            image = cv2.circle(image, (int(tmp_R1Coors[index, 0]), int(tmp_R1Coors[index, 1])), 5,
                               (0, 255, 0), -1)
    if pred[1] == 1:
        tmp_L1Coors = getValue(L1Coors[0])
        tmp_L1Coors = tmp_L1Coors.reshape(15, 2)
        tmp_L1Coors = 0.5 * ((tmp_L1Coors + 1) * image_size - 1)
        for index in range(15):
            image = cv2.circle(image, (int(tmp_L1Coors[index, 0]), int(tmp_L1Coors[index, 1])), 5,
                               (0, 0, 255), -1)
    if pred[0] == 1:
        tmp_L2Coors = getValue(L2Coors[0])
        tmp_L2Coors = tmp_L2Coors.reshape(15, 2)
        tmp_L2Coors = 0.5 * ((tmp_L2Coors + 1) * image_size - 1)
        for index in range(15):
            image = cv2.circle(image, (int(tmp_L2Coors[index, 0]), int(tmp_L2Coors[index, 1])), 5,
                               (255, 255, 0), -1)

    plt.imshow(image)