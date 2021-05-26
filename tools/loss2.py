import torch
import torch.nn as nn
import Model.DSNT as dsntnn
from Parameter import Parameters
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
device_id = [0,1]
cuda_avaliable = torch.cuda.is_available()
p = Parameters()

def calculate_exist_loss(pred_exist,exist_label):
    '''
    :param pred_exist:
    :param exist_label:
    :return:
    '''
    criterion_exist = torch.nn.BCELoss().cuda()
    loss_exist = criterion_exist(pred_exist, exist_label)

    return loss_exist

def calculate_point_loss(actualR2, target_R2, R2HeapMaps, actualR1, target_R1, R1HeapMaps, actualL2, target_L2, L2HeapMaps, actualL1, target_L1, L1HeapMaps, exist_label):
    '''
    :param actualR2:
    :param target_R2:
    :param R2HeapMaps:
    :param actualR1:
    :param target_R1:
    :param R1HeapMaps:
    :param actualL2:
    :param target_L2:
    :param L2HeapMaps:
    :param actualL1:
    :param target_L1:
    :param L1HeapMaps:
    :param exist_label:
    :return:
    '''
    batch_size = exist_label.shape[0]
    maskR2 = torch.Tensor([1]).expand_as(actualR2[:,:,0].reshape(batch_size,p.point_num)).cuda()
    maskR1 = torch.Tensor([1]).expand_as(actualR1[:,:,0].reshape(batch_size,p.point_num)).cuda()
    maskL2 = torch.Tensor([1]).expand_as(actualL2[:,:,0].reshape(batch_size,p.point_num)).cuda()
    maskL1 = torch.Tensor([1]).expand_as(actualL1[:,:,0].reshape(batch_size,p.point_num)).cuda()
    for i in range(batch_size):
        tmp_label = exist_label[i]
        if tmp_label[0] == 0:
            maskL2[i] = 0
        if tmp_label[1] == 0:
            maskL1[i] = 0
        if tmp_label[2] == 0:
            maskR1[i] = 0
        if tmp_label[3] == 0:
            maskR2[i] = 0

    loss_pointR2 = dsntnn.euclidean_losses(actualR2, target_R2)
    loss_regR2 = dsntnn.js_reg_losses(R2HeapMaps, target_R2, sigma_t=1.0)
    lossR2 = dsntnn.average_loss(loss_pointR2 + loss_regR2, maskR2)
    loss_pointR1 = dsntnn.euclidean_losses(actualR1, target_R1)
    loss_regR1 = dsntnn.js_reg_losses(R1HeapMaps, target_R1, sigma_t=1.0)
    lossR1 = dsntnn.average_loss(loss_pointR1 + loss_regR1, maskR1)
    loss_pointL1 = dsntnn.euclidean_losses(actualL1, target_L1)
    loss_regL1 = dsntnn.js_reg_losses(L1HeapMaps, target_L1, sigma_t=1.0)
    lossL1 = dsntnn.average_loss(loss_pointL1 + loss_regL1, maskL1)
    loss_pointL2 = dsntnn.euclidean_losses(actualL2, target_L2)
    loss_regL2 = dsntnn.js_reg_losses(L2HeapMaps, target_L2, sigma_t=1.0)
    lossL2 = dsntnn.average_loss(loss_pointL2 + loss_regL2, maskL2)

    loss = lossR2 + lossR1 + lossL1 + lossL2

    denom = maskR2.sum() + maskR1.sum() + maskL1.sum() + maskL2.sum()

    loss = loss/denom

    return loss


def calculate_line_loss(actualR2, actualR1, actualL1, actualL2, exist_label):
    loss = 0
    for i in range(actualR2.shape[0]):
        if exist_label[i][3] == 1:
            for j in range(actualR2.shape[1]-2):
                loss = (actualR2[i,j,0]-actualR2[i,j+1,0])/(actualR2[i,j+1,1]-actualR2[i,j+2,1]) + loss

    for i in range(actualR1.shape[0]):
        if exist_label[i][2] == 1:
            for j in range(actualR1.shape[1] - 2):
                loss = (actualR1[i, j, 0] - actualR1[i, j + 1, 0]) / (
                            actualR1[i, j + 1, 1] - actualR1[i, j + 2, 1]) + loss


    for i in range(actualL1.shape[0]):
        if exist_label[i][1] == 1:
            for j in range(actualL1.shape[1] - 2):
                loss = (actualL1[i, j, 0] - actualL1[i, j + 1, 0]) / (
                            actualL1[i, j + 1, 1] - actualL1[i, j + 2, 1]) + loss



    for i in range(actualL2.shape[0]):
        if exist_label[i][0] == 1:
            for j in range(actualL2.shape[1] - 2):
                loss = (actualL2[i, j, 0] - actualL2[i, j + 1, 0]) / (
                        actualL2[i, j + 1, 1] - actualL2[i, j + 2, 1]) + loss

    denom = exist_label.sum()

    loss = loss/denom

    return loss



class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()


    def forward(self, target_points, actual_points):
        '''
        :param target_points:  B*M*2 in GPU
        :param actual_points:  B*N*2 in GPU
        :return:
        '''

        B,M = target_points.shape[0],target_points.shape[1]
        N = actual_points.shape[1]

        target_points_expand = target_points.unsqueeze(2).expand(B,M,N,2)
        actual_points_expand = actual_points.unsqueeze(1).expand(B,M,N,2)

        diff = torch.norm(target_points_expand - actual_points_expand, dim=3, keepdim=False)

        target_actual_min_dist, _ = torch.min(diff,dim=2,keepdim=False)
        forward_loss = target_actual_min_dist.mean()

        actual_target_min_dist, _ = torch.min(diff,dim=1,keepdim=False)
        backward_loss = actual_target_min_dist.mean()

        chamfer_pure = forward_loss + backward_loss

        return chamfer_pure










