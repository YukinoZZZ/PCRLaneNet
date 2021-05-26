from Model.deeplabv2 import DeepLabV2
import Model.DSNT as dsntnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from Parameter import Parameters

p = Parameters()

class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super(CoordRegressionNetwork, self).__init__()
        self.hm_conv = nn.Conv2d(256, n_locations, kernel_size=1, bias=False)

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
    def __init__(self, num_output):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(256, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()

        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(4500, 128)
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
        output = output.view(-1, 4500)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.sigmoid(output)

        return output


class Lane_Detection(nn.Module):
    def __init__(self):
        super(Lane_Detection, self).__init__()
        self.Encode = DeepLabV2( n_blocks=[3, 4, 6, 3], atrous_rates=[6, 12, 18]
    )
        self.lane_exist = Lane_exist(4)
        self.RL2 = CoordRegressionNetwork(n_locations=p.point_num)
        self.RL1 = CoordRegressionNetwork(n_locations=p.point_num)
        self.LL1 = CoordRegressionNetwork(n_locations=p.point_num)
        self.LL2 = CoordRegressionNetwork(n_locations=p.point_num)

    def forward(self, input):
        encoder = self.Encode(input)
        pred_exist = self.lane_exist(encoder)
        R2Coors, R2HeapMaps = self.RL2(encoder)
        R1Coors, R1HeapMaps = self.RL1(encoder)
        L1Coors, L1HeapMaps = self.LL1(encoder)
        L2Coors, L2HeapMaps = self.LL2(encoder)

        return pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps


if __name__ == "__main__":
    model = Lane_Detection()
    model.eval()
    image = torch.randn(3, 3, 288, 800)

    print(model)
    print("input:", image.shape)
    pred_exist, R2Coors, R1Coors, L1Coors, L2Coors = model(image)
    print("output:", pred_exist.shape)


