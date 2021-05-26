import torch
import torch.nn as nn
import Model.DSNT as dsntnn
from Parameter import Parameters
#from tools.EMD_Loss import earth_mover_distance
import torch.nn.functional as F
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
device_id = [0,3]
cuda_avaliable = torch.cuda.is_available()
p = Parameters()

def calculate_exist_loss(pred_exist,exist_label):
    '''
    :param pred_exist:
    :param exist_label:
    :return: exist loss
    '''
    criterion_exist = nn.BCELoss().cuda()
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

    if maskR2.sum() == 0:
        loss_R2 = lossR2 * maskR2.sum()
    else:
        loss_R2 = lossR2 / maskR2.sum()

    if maskR1.sum() == 0:
        loss_R1 = lossR1 * maskR1.sum()
    else:
        loss_R1 = lossR1 / maskR1.sum()

    if maskL1.sum() == 0:
        loss_L1 = lossL1 * maskL1.sum()
    else:
        loss_L1 = lossL1 / maskL1.sum()

    if maskL2.sum() == 0:
        loss_L2 = lossL2 * maskL2.sum()
    else:
        loss_L2 = lossL2 / maskL2.sum()

    loss = loss_L2 + loss_L1 + loss_R1 + loss_R2


    return loss

def calculate_point_l1_loss(actualR2, target_R2, R2HeapMaps, actualR1, target_R1, R1HeapMaps, actualL2, target_L2, L2HeapMaps, actualL1, target_L1, L1HeapMaps, exist_label):
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

    loss_pointR2 = dsntnn.smooth_l1_loss(actualR2, target_R2)
    #loss_regR2 = dsntnn.js_reg_losses(R2HeapMaps, target_R2, sigma_t=1.0)
    lossR2 = dsntnn.average_loss(loss_pointR2, maskR2)
    loss_pointR1 = dsntnn.smooth_l1_loss(actualR1, target_R1)
    #loss_regR1 = dsntnn.js_reg_losses(R1HeapMaps, target_R1, sigma_t=1.0)
    lossR1 = dsntnn.average_loss(loss_pointR1, maskR1)
    loss_pointL1 = dsntnn.smooth_l1_loss(actualL1, target_L1)
    #loss_regL1 = dsntnn.js_reg_losses(L1HeapMaps, target_L1, sigma_t=1.0)
    lossL1 = dsntnn.average_loss(loss_pointL1, maskL1)
    loss_pointL2 = dsntnn.smooth_l1_loss(actualL2, target_L2)
    #loss_regL2 = dsntnn.js_reg_losses(L2HeapMaps, target_L2, sigma_t=1.0)
    lossL2 = dsntnn.average_loss(loss_pointL2, maskL2)

    loss = lossR2/maskR2.sum() + lossR1/maskR1.sum() + lossL1/maskL1.sum() + lossL2/maskL2.sum()


    return loss



class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()


    def forward(self, target_points, actual_points):
        '''

        :param target_points: B*M*2 in GPU
        :param actual_points: B*N*2 in GPU
        :return:
        '''

        B,M = target_points.shape[0],target_points.shape[1]
        N = actual_points.shape[1]

        target_points_expand = target_points.unsqueeze(2).expand(B,M,N,2)
        actual_points_expand = actual_points.unsqueeze(1).expand(B,M,N,2)

        diff = torch.norm(target_points_expand - actual_points_expand, dim=3, keepdim=False)

        target_actual_min_dist, _ = torch.min(diff,dim=2,keepdim=False)
        forward_loss = target_actual_min_dist

        actual_target_min_dist, _ = torch.min(diff,dim=1,keepdim=False)
        backward_loss = actual_target_min_dist


        return forward_loss, backward_loss


def calculate_point_chamfer_loss(actualR2, target_R2, R2HeapMaps, actualR1, target_R1, R1HeapMaps, actualL2, target_L2, L2HeapMaps, actualL1, target_L1, L1HeapMaps, exist_label):
    '''
    :param actualR2: the lane point that the network pred
    :param target_R2: the ground lane point
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
    forward_maskR2 = torch.Tensor([1]).expand_as(target_R2[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    forward_maskR1 = torch.Tensor([1]).expand_as(target_R1[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    forward_maskL2 = torch.Tensor([1]).expand_as(target_L2[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    forward_maskL1 = torch.Tensor([1]).expand_as(target_L1[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    for i in range(batch_size):
        tmp_label = exist_label[i]
        if tmp_label[0] == 0:
            maskL2[i] = 0
            forward_maskL2[i] = 0
        if tmp_label[1] == 0:
            maskL1[i] = 0
            forward_maskL1[i] = 0
        if tmp_label[2] == 0:
            maskR1[i] = 0
            forward_maskR1[i] = 0
        if tmp_label[3] == 0:
            maskR2[i] = 0
            forward_maskR2[i] = 0
    chamfer_losses = ChamferLoss()
    forward_lossR2,backward_lossR2 = chamfer_losses(target_R2,actualR2)
    forward_lossR1, backward_lossR1 = chamfer_losses(target_R1, actualR1)
    forward_lossL1, backward_lossL1 = chamfer_losses(target_L1, actualL1)
    forward_lossL2, backward_lossL2 = chamfer_losses(target_L2, actualL2)

    if maskR1.sum() == 0:
        R1_chamfer_loss = (forward_lossR1 * forward_maskR1).sum() * forward_maskR1.sum() + (
                    backward_lossR1 * maskR1).sum() * maskR1.sum()
    else:
        R1_chamfer_loss = (forward_lossR1 * forward_maskR1).sum()/forward_maskR1.sum() + (backward_lossR1 * maskR1).sum()/maskR1.sum()
    if maskR2.sum() == 0:
        R2_chamfer_loss = (forward_lossR2 * forward_maskR2).sum() * forward_maskR2.sum() + (
                backward_lossR2 * maskR2).sum() * maskR2.sum()
    else:
        R2_chamfer_loss = (forward_lossR2 * forward_maskR2).sum() / forward_maskR2.sum() + (
                    backward_lossR2 * maskR2).sum() / maskR2.sum()
    if maskL1.sum() == 0:
        L1_chamfer_loss = (forward_lossL1 * forward_maskL1).sum() * forward_maskL1.sum() + (
                backward_lossL1 * maskL1).sum() * maskL1.sum()
    else:
        L1_chamfer_loss = (forward_lossL1 * forward_maskL1).sum() / forward_maskL1.sum() + (
                    backward_lossL1 * maskL1).sum() / maskL1.sum()
    if maskL2.sum() == 0:
        L2_chamfer_loss = (forward_lossL2 * forward_maskL2).sum() * forward_maskL2.sum() + (
                backward_lossL2 * maskL2).sum() * maskL2.sum()
    else:
        L2_chamfer_loss = (forward_lossL2 * forward_maskL2).sum() / forward_maskL2.sum() + (
                    backward_lossL2 * maskL2).sum() / maskL2.sum()

    loss = R1_chamfer_loss + R2_chamfer_loss + L1_chamfer_loss + L2_chamfer_loss

    return loss

def lane_emd_loss(actualR2, target_R2, R2HeapMaps, actualR1, target_R1, R1HeapMaps, actualL2, target_L2, L2HeapMaps, actualL1, target_L1, L1HeapMaps, exist_label):
    batch_size = exist_label.shape[0]
    maskR2 = torch.ones(batch_size).cuda()
    maskR1 = torch.ones(batch_size).cuda()
    maskL2 = torch.ones(batch_size).cuda()
    maskL1 = torch.ones(batch_size).cuda()

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

    emd_loss_L2 = earth_mover_distance(actualL2,target_L2,transpose=False)
    emd_loss_L1 = earth_mover_distance(actualL1, target_L1, transpose=False)
    emd_loss_R1 = earth_mover_distance(actualR1, target_R1, transpose=False)
    emd_loss_R2 = earth_mover_distance(actualR2, target_R2, transpose=False)

    emd_loss_mean = (emd_loss_L1*maskL1 + emd_loss_L2*maskL2 + emd_loss_R1*maskR1 + emd_loss_R2*maskR2).mean()

    return emd_loss_mean


class Least_square_loss(nn.Module):
    def __init__(self,point_num,batch_size,order=3,use_cholesky=False,no_cuda=False,reg_ls=0):
        super(Least_square_loss, self).__init__()
        self.order = order
        self.use_cholesky = use_cholesky
        self.tensor_ones = torch.ones(batch_size,point_num,1).float()
        self.reg_ls = reg_ls*torch.eye(order+1)
        self.no_cuda = no_cuda
        if not no_cuda:
            self.tensor_ones = self.tensor_ones.cuda()
            self.reg_ls = self.reg_ls.cuda()


    def forward(self,target_points,actual_points):
        pred_tensor_ones = torch.ones(target_points.shape[0],target_points.shape[1],1).float()
        actual_x = actual_points[:,:,0].reshape(actual_points.shape[0],actual_points.shape[1],1)
        actual_y = actual_points[:, :, 1].reshape(actual_points.shape[0], actual_points.shape[1], 1)
        target_x = target_points[:,:,0].reshape(target_points.shape[0],target_points.shape[1],1)
        target_y = target_points[:, :, 1].reshape(target_points.shape[0], target_points.shape[1], 1)
        if not self.no_cuda:
            pred_tensor_ones = pred_tensor_ones.cuda()
        if self.order == 0:
            Y = self.tensor_ones
            pred_Y = pred_tensor_ones
        elif self.order == 1:
            Y = torch.cat((actual_y,self.tensor_ones),2)
            pred_Y = torch.cat((target_y,pred_tensor_ones),2)
        elif self.order == 2:
            Y = torch.cat((actual_y**2,actual_y,self.tensor_ones),2)
            pred_Y = torch.cat((target_y**2,target_y, pred_tensor_ones), 2)
        elif self.order == 3:
            Y = torch.cat((actual_y**3,actual_y**2,actual_y, self.tensor_ones), 2)
            pred_Y = torch.cat((target_y ** 3,target_y ** 2, target_y, pred_tensor_ones), 2)
        else:
            raise NotImplementedError('Requested order for polynomial fit is not implemented')

        Z = torch.bmm(Y.transpose(1, 2), Y) + self.reg_ls
        if not self.use_cholesky:
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y.transpose(1,2),actual_x)
            beta = torch.bmm(Z_inv,X)
        else:
            beta = []
            X = torch.bmm(Y.transpose(1, 2), actual_x)
            for image,rhs in zip(torch.unbind(Z),torch.unbind(X)):
                R = torch.cholesky(image)
                opl = torch.triangular_solve(rhs,R,upper=False)
                beta.append(torch.triangular_solve(opl[0],R.transpose(0,1))[0])

            beta = torch.cat((beta),1).transpose(0,1).unsqueeze(2)

        pred_X = torch.bmm(pred_Y,beta)

        loss = torch.norm(pred_X-target_x,dim=2)

        return loss


class Least_square_loss_self(nn.Module):
    def __init__(self,point_num,batch_size,order=2,use_cholesky=False,no_cuda=False,reg_ls=0):
        super(Least_square_loss_self, self).__init__()
        self.order = order
        self.use_cholesky = use_cholesky
        self.tensor_ones = torch.ones(batch_size,point_num,1).float()
        self.reg_ls = reg_ls*torch.eye(order+1)
        self.no_cuda = no_cuda
        if not no_cuda:
            self.tensor_ones = self.tensor_ones.cuda()
            self.reg_ls = self.reg_ls.cuda()


    def forward(self,actual_points):

        actual_x = actual_points[:,:,0].reshape(actual_points.shape[0],actual_points.shape[1],1)
        actual_y = actual_points[:, :, 1].reshape(actual_points.shape[0], actual_points.shape[1], 1)


        if self.order == 0:
            Y = self.tensor_ones
        elif self.order == 1:
            Y = torch.cat((actual_y,self.tensor_ones),2)
        elif self.order == 2:
            Y = torch.cat((actual_y**2,actual_y,self.tensor_ones),2)
        elif self.order == 3:
            Y = torch.cat((actual_y**3,actual_y**2,actual_y, self.tensor_ones), 2)
        else:
            raise NotImplementedError('Requested order for polynomial fit is not implemented')

        Z = torch.bmm(Y.transpose(1, 2), Y) + self.reg_ls
        if not self.use_cholesky:
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y.transpose(1,2),actual_x)
            beta = torch.bmm(Z_inv,X)
        else:
            beta = []
            X = torch.bmm(Y.transpose(1, 2), actual_x)
            for image,rhs in zip(torch.unbind(Z),torch.unbind(X)):
                R = torch.cholesky(image)
                opl = torch.triangular_solve(rhs,R,upper=False)
                beta.append(torch.triangular_solve(opl[0],R.transpose(0,1))[0])

            beta = torch.cat((beta),1).transpose(0,1).unsqueeze(2)

        pred_X = torch.bmm(Y,beta)

        loss = torch.norm(pred_X-actual_x,dim=2)

        return loss

class Least_square_loss_v2(nn.Module):
    def __init__(self,point_num,batch_size,order=3,use_cholesky=False,no_cuda=False,reg_ls=0):
        super(Least_square_loss_v2, self).__init__()
        self.order = order
        self.use_cholesky = use_cholesky
        self.tensor_ones = torch.ones(batch_size,point_num,1).float()
        self.reg_ls = reg_ls*torch.eye(order+1)
        self.no_cuda = no_cuda
        if not no_cuda:
            self.tensor_ones = self.tensor_ones.cuda()
            self.reg_ls = self.reg_ls.cuda()


    def forward(self,target_points,actual_points):
        '''
        :param target_points: output of the network
        :param actual_points: ground truth
        :return: loss
        '''
        pred_tensor_ones = torch.ones(target_points.shape[0],target_points.shape[1],1).float()
        actual_x = actual_points[:,:,0].reshape(actual_points.shape[0],actual_points.shape[1],1)
        actual_y = actual_points[:, :, 1].reshape(actual_points.shape[0], actual_points.shape[1], 1)
        target_x = target_points[:,:,0].reshape(target_points.shape[0],target_points.shape[1],1)
        target_y = target_points[:, :, 1].reshape(target_points.shape[0], target_points.shape[1], 1)
        if not self.no_cuda:
            pred_tensor_ones = pred_tensor_ones.cuda()
        if self.order == 0:
            Y = self.tensor_ones
            pred_Y = pred_tensor_ones
        elif self.order == 1:
            Y = torch.cat((actual_y,self.tensor_ones),2)
            pred_Y = torch.cat((target_y,pred_tensor_ones),2)
        elif self.order == 2:
            Y = torch.cat((actual_y**2,actual_y,self.tensor_ones),2)
            pred_Y = torch.cat((target_y**2,target_y, pred_tensor_ones), 2)
        elif self.order == 3:
            Y = torch.cat((actual_y**3,actual_y**2,actual_y, self.tensor_ones), 2)
            pred_Y = torch.cat((target_y ** 3,target_y ** 2, target_y, pred_tensor_ones), 2)
        elif self.order == 4:
            Y = torch.cat((actual_y**4,actual_y**3,actual_y**2,actual_y, self.tensor_ones), 2)
            pred_Y = torch.cat((target_y ** 4,target_y ** 3,target_y ** 2, target_y, pred_tensor_ones), 2)
        else:
            raise NotImplementedError('Requested order for polynomial fit is not implemented')

        Z = torch.bmm(Y.transpose(1, 2), Y) + self.reg_ls
        if not self.use_cholesky:
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y.transpose(1,2),actual_x)
            beta = torch.bmm(Z_inv,X)
        else:
            beta = []
            X = torch.bmm(Y.transpose(1, 2), actual_x)
            for image,rhs in zip(torch.unbind(Z),torch.unbind(X)):
                R = torch.cholesky(image)
                opl = torch.triangular_solve(rhs,R,upper=False)
                beta.append(torch.triangular_solve(opl[0],R.transpose(0,1))[0])

            beta = torch.cat((beta),1).transpose(0,1).unsqueeze(2)

        pred_X = torch.bmm(pred_Y,beta)

        loss = torch.norm(pred_X-target_x,dim=2)

        return loss



class Least_square_loss_v3(nn.Module):
    def __init__(self,point_num,batch_size,order=3,use_cholesky=False,no_cuda=False,reg_ls=0):
        super(Least_square_loss_v3, self).__init__()
        self.order = order
        self.use_cholesky = use_cholesky
        self.tensor_ones = torch.ones(batch_size,point_num,1).float()
        self.reg_ls = reg_ls*torch.eye(order+1)
        self.no_cuda = no_cuda
        if not no_cuda:
            self.tensor_ones = self.tensor_ones.cuda()
            self.reg_ls = self.reg_ls.cuda()


    def forward(self,target_points,actual_points):
        '''
        :param target_points: output of the network
        :param actual_points: ground truth
        :return: loss, the ground truth of the lane kind
        '''
        pred_tensor_ones = torch.ones(target_points.shape[0],target_points.shape[1],1).float()
        actual_x = actual_points[:,:,0].reshape(actual_points.shape[0],actual_points.shape[1],1)
        actual_y = actual_points[:, :, 1].reshape(actual_points.shape[0], actual_points.shape[1], 1)
        target_x = target_points[:,:,0].reshape(target_points.shape[0],target_points.shape[1],1)
        target_y = target_points[:, :, 1].reshape(target_points.shape[0], target_points.shape[1], 1)
        if not self.no_cuda:
            pred_tensor_ones = pred_tensor_ones.cuda()
        if self.order == 0:
            Y = self.tensor_ones
            pred_Y = pred_tensor_ones
        elif self.order == 1:
            Y = torch.cat((actual_y,self.tensor_ones),2)
            pred_Y = torch.cat((target_y,pred_tensor_ones),2)
        elif self.order == 2:
            Y = torch.cat((actual_y**2,actual_y,self.tensor_ones),2)
            pred_Y = torch.cat((target_y**2,target_y, pred_tensor_ones), 2)
        elif self.order == 3:
            Y = torch.cat((actual_y**3,actual_y**2,actual_y, self.tensor_ones), 2)
            pred_Y = torch.cat((target_y ** 3,target_y ** 2, target_y, pred_tensor_ones), 2)
        else:
            raise NotImplementedError('Requested order for polynomial fit is not implemented')

        Z = torch.bmm(Y.transpose(1, 2), Y) + self.reg_ls
        if not self.use_cholesky:
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y.transpose(1,2),actual_x)
            beta = torch.bmm(Z_inv,X)
        else:
            beta = []
            X = torch.bmm(Y.transpose(1, 2), actual_x)
            for image,rhs in zip(torch.unbind(Z),torch.unbind(X)):
                R = torch.cholesky(image)
                opl = torch.triangular_solve(rhs,R,upper=False)
                beta.append(torch.triangular_solve(opl[0],R.transpose(0,1))[0])

            beta = torch.cat((beta),1).transpose(0,1).unsqueeze(2)

        pred_X = torch.bmm(pred_Y,beta)

        loss = torch.norm(pred_X-target_x,dim=2)

        return loss



def calculate_point_chamfer_loss_V2(actualR2, target_R2, R2HeapMaps, actualR1, target_R1, R1HeapMaps, actualL2, target_L2, L2HeapMaps, actualL1, target_L1, L1HeapMaps, exist_label):
    '''
    loss function design by wangpan 2020.07.28
    make the network care more about the curve
    '''
    batch_size = exist_label.shape[0]
    cal_lsq = True
    maskR2 = torch.Tensor([1]).expand_as(actualR2[:,:,0].reshape(batch_size,p.point_num)).cuda()
    maskR1 = torch.Tensor([1]).expand_as(actualR1[:,:,0].reshape(batch_size,p.point_num)).cuda()
    maskL2 = torch.Tensor([1]).expand_as(actualL2[:,:,0].reshape(batch_size,p.point_num)).cuda()
    maskL1 = torch.Tensor([1]).expand_as(actualL1[:,:,0].reshape(batch_size,p.point_num)).cuda()
    forward_maskR2 = torch.Tensor([1]).expand_as(target_R2[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    forward_maskR1 = torch.Tensor([1]).expand_as(target_R1[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    forward_maskL2 = torch.Tensor([1]).expand_as(target_L2[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    forward_maskL1 = torch.Tensor([1]).expand_as(target_L1[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    for i in range(batch_size):
        tmp_label = exist_label[i]
        if tmp_label[0] == 0:
            maskL2[i] = 0
            forward_maskL2[i] = 0
        if tmp_label[1] == 0:
            maskL1[i] = 0
            forward_maskL1[i] = 0
        if tmp_label[2] == 0:
            maskR1[i] = 0
            forward_maskR1[i] = 0
        if tmp_label[3] == 0:
            maskR2[i] = 0
            forward_maskR2[i] = 0
    chamfer_losses = ChamferLoss()
    forward_lossR2,backward_lossR2 = chamfer_losses(target_R2,actualR2)
    forward_lossR1, backward_lossR1 = chamfer_losses(target_R1, actualR1)
    forward_lossL1, backward_lossL1 = chamfer_losses(target_L1, actualL1)
    forward_lossL2, backward_lossL2 = chamfer_losses(target_L2, actualL2)

    if cal_lsq:
        calculate_lsq_loss = Least_square_loss_v2(p.ground_truth_num, batch_size, order=4)
        R2_lsq_loss = calculate_lsq_loss(actualR2,target_R2)
        R1_lsq_loss = calculate_lsq_loss(actualR1,target_R1)
        L1_lsq_loss = calculate_lsq_loss(actualL1,target_L1)
        L2_lsq_loss = calculate_lsq_loss(actualL2,target_L2)
    else:
        R2_lsq_loss = torch.tensor(0).cuda()
        R1_lsq_loss = torch.tensor(0).cuda()
        L1_lsq_loss = torch.tensor(0).cuda()
        L2_lsq_loss = torch.tensor(0).cuda()

    ##add lane shape piror loss function by wangpan 2020.07.29
    loss_prior_R2 = lane_prior_loss(actualR2, maskR2)
    loss_prior_R1 = lane_prior_loss(actualR1, maskR1)
    loss_prior_L2 = lane_prior_loss(actualR2, maskL2)
    loss_prior_L1 = lane_prior_loss(actualR2, maskL1)

    loss_prior = loss_prior_L1 + loss_prior_L2 + loss_prior_R1 + loss_prior_R2


    R1_chamfer_loss = (forward_lossR1 * forward_maskR1).sum()/forward_maskR1.sum() + (backward_lossR1 * maskR1).sum()/maskR1.sum()
    R2_chamfer_loss = (forward_lossR2 * forward_maskR2).sum()/forward_maskR2.sum() + (backward_lossR2 * maskR2).sum()/maskR2.sum()
    L1_chamfer_loss = (forward_lossL1 * forward_maskL1).sum()/forward_maskL1.sum() + (backward_lossL1 * maskL1).sum()/maskL1.sum()
    L2_chamfer_loss = (forward_lossL2 * forward_maskL2).sum()/forward_lossL2.sum() + (backward_lossL2 * maskL2).sum()/maskL2.sum()
    loss = R1_chamfer_loss + R2_chamfer_loss + L1_chamfer_loss + L2_chamfer_loss

    lsq_loss = (R1_lsq_loss * maskR1).sum()/maskR1.sum() + (R2_lsq_loss * maskR2).sum()/maskR2.sum() \
               + (L1_lsq_loss * maskL1).sum()/maskL1.sum() + (L2_lsq_loss * maskL2).sum()/maskL2.sum()

    return loss, lsq_loss, loss_prior


def py_sigmoid_focal_loss(pred,
                          target,
                          weight = None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean'):
    pred_sigmoid = pred
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_cross_entropy(pred_sigmoid, target, number, reduction='sum'):
    target = target.type_as(pred_sigmoid)
    alpha = (number-target.sum())/number
    weight = (alpha * target + 1.1 * (1 - alpha) * (1 - target))
    loss = F.binary_cross_entropy_with_logits(
        pred_sigmoid, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def lane_prior_loss(pred_lane,mask, thesold_y = 0.07):
    '''
    :param pred_lane: the lane that the network pred
    :return: prior loss of the lane
    '''
    B,point_num, dim = pred_lane.shape
    lane_y = pred_lane[:,:,1]
    _, index_y = torch.sort(lane_y,dim=1)
    #loss of the x direction
    loss_x = torch.abs((pred_lane[0, index_y[0,0], 0] - pred_lane[0, index_y[0,1], 0]) -
                       (pred_lane[0, index_y[0,1], 0] - pred_lane[0, index_y[0,2], 0]))

    compare = torch.Tensor([thesold_y-torch.abs(pred_lane[0, index_y[0, 0], 1]-pred_lane[0, index_y[0, 1], 1]),0]).cuda()
    loss_y = torch.max(compare)
    for i in range(B):
        if mask[i].sum() > 0:
            for j in range(point_num-3):
                loss_x = torch.abs(torch.abs(pred_lane[i, index_y[i,j+1], 0] - pred_lane[i, index_y[i,j+2], 0]) -
                           torch.abs(pred_lane[i, index_y[i,j+2], 0] - pred_lane[i, index_y[i,j+3], 0])) + loss_x


            for k in range(point_num-2):
                compare = torch.Tensor([thesold_y-torch.abs(pred_lane[i, index_y[i,k+1], 1]-pred_lane[i, index_y[i,k+2], 1]),0]).cuda()
                loss_y = loss_y + torch.max(compare)

    loss_prior = (loss_y + loss_x)/mask.sum()

    return loss_prior


def calculate_point_loss_with_lsq(actualR2, target_R2, R2HeapMaps, actualR1, target_R1, R1HeapMaps, actualL2, target_L2, L2HeapMaps, actualL1, target_L1, L1HeapMaps, exist_label):
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
    lsq_loss = Least_square_loss_self(point_num=15,batch_size=batch_size,order=3,use_cholesky=False)
    lsq_loss_R2 = lsq_loss(actualR2)
    lsq_loss_R1 = lsq_loss(actualR1)
    lsq_loss_L1 = lsq_loss(actualL1)
    lsq_loss_L2 = lsq_loss(actualL2)


    loss_pointR2 = dsntnn.euclidean_losses(actualR2, target_R2)
    loss_regR2 = dsntnn.js_reg_losses(R2HeapMaps, target_R2, sigma_t=1.0)
    lossR2 = dsntnn.average_loss(loss_pointR2 + loss_regR2 + lsq_loss_R2, maskR2)
    loss_pointR1 = dsntnn.euclidean_losses(actualR1, target_R1)
    loss_regR1 = dsntnn.js_reg_losses(R1HeapMaps, target_R1, sigma_t=1.0)
    lossR1 = dsntnn.average_loss(loss_pointR1 + loss_regR1 + lsq_loss_R1, maskR1)
    loss_pointL1 = dsntnn.euclidean_losses(actualL1, target_L1)
    loss_regL1 = dsntnn.js_reg_losses(L1HeapMaps, target_L1, sigma_t=1.0)
    lossL1 = dsntnn.average_loss(loss_pointL1 + loss_regL1 + lsq_loss_L1, maskL1)
    loss_pointL2 = dsntnn.euclidean_losses(actualL2, target_L2)
    loss_regL2 = dsntnn.js_reg_losses(L2HeapMaps, target_L2, sigma_t=1.0)
    lossL2 = dsntnn.average_loss(loss_pointL2 + loss_regL2 + lsq_loss_L2, maskL2)

    loss = lossR2/maskR2.sum() + lossR1/maskR1.sum() + lossL1/maskL1.sum() + lossL2/maskL2.sum()


    return loss

def calculate_point_chamfer_loss_with_lsq(actualR2, target_R2, R2HeapMaps, actualR1, target_R1, R1HeapMaps, actualL2, target_L2, L2HeapMaps, actualL1, target_L1, L1HeapMaps, exist_label):
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
    forward_maskR2 = torch.Tensor([1]).expand_as(target_R2[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    forward_maskR1 = torch.Tensor([1]).expand_as(target_R1[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    forward_maskL2 = torch.Tensor([1]).expand_as(target_L2[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    forward_maskL1 = torch.Tensor([1]).expand_as(target_L1[:, :, 0].reshape(batch_size, p.ground_truth_num)).cuda()
    for i in range(batch_size):
        tmp_label = exist_label[i]
        if tmp_label[0] == 0:
            maskL2[i] = 0
            forward_maskL2[i] = 0
        if tmp_label[1] == 0:
            maskL1[i] = 0
            forward_maskL1[i] = 0
        if tmp_label[2] == 0:
            maskR1[i] = 0
            forward_maskR1[i] = 0
        if tmp_label[3] == 0:
            maskR2[i] = 0
            forward_maskR2[i] = 0
    chamfer_losses = ChamferLoss()
    forward_lossR2,backward_lossR2 = chamfer_losses(target_R2,actualR2)
    forward_lossR1, backward_lossR1 = chamfer_losses(target_R1, actualR1)
    forward_lossL1, backward_lossL1 = chamfer_losses(target_L1, actualL1)
    forward_lossL2, backward_lossL2 = chamfer_losses(target_L2, actualL2)

    lsq_loss = Least_square_loss_self(point_num=15, batch_size=batch_size, order=3, use_cholesky=False)
    lsq_loss_R2 = lsq_loss(actualR2)
    lsq_loss_R1 = lsq_loss(actualR1)
    lsq_loss_L1 = lsq_loss(actualL1)
    lsq_loss_L2 = lsq_loss(actualL2)

    R1_chamfer_loss = (forward_lossR1 * forward_maskR1).sum()/forward_maskR1.sum() + (backward_lossR1 * maskR1).sum()/maskR1.sum()
    R2_chamfer_loss = (forward_lossR2 * forward_maskR2).sum()/forward_maskR2.sum() + (backward_lossR2 * maskR2).sum()/maskR2.sum()
    L1_chamfer_loss = (forward_lossL1 * forward_maskL1).sum()/forward_maskL1.sum() + (backward_lossL1 * maskL1).sum()/maskL1.sum()
    L2_chamfer_loss = (forward_lossL2 * forward_maskL2).sum()/forward_lossL2.sum() + (backward_lossL2 * maskL2).sum()/maskL2.sum()

    point_lsq_loss = (lsq_loss_R2*maskR2).sum()/maskR2.sum() + (lsq_loss_R1*maskR1).sum()/maskR1.sum() + (lsq_loss_L1*maskL1).sum()/maskL1.sum() + (lsq_loss_L2*maskL2).sum()/maskL2.sum()
    loss = R1_chamfer_loss + R2_chamfer_loss + L1_chamfer_loss + L2_chamfer_loss + point_lsq_loss

    return loss















