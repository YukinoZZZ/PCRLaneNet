import numpy as np
import torch
from Parameter import Parameters
import cv2
from scipy.optimize import linear_sum_assignment
from scipy import interpolate



def LaneEval(image_data, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, lines_data,len_lanes):
    p = Parameters()
    batch_size = pred_exist.shape[0]
    ratio_w = p.x_size * 1.0 / image_data.shape[2]
    ratio_h = p.y_size * 1.0 / image_data.shape[1]
    image_size = np.array([p.x_size,p.y_size]).reshape(1, 2)
    TP = 0
    FP = 0
    FN = 0
    for i in range(batch_size):
        line_data = lines_data[i]
        len_lane = len_lanes[i]
        tmp_x = []
        tmp_y = []
        ground_line = []
        length = len(line_data)
        pred = pred_exist[i] >= 0.5
        if length>0:
            index = 0
            for m in range(len(len_lane)):
                for n in range(int(len_lane[m])):
                    tmp_x.append(line_data[index][0])
                    tmp_y.append(line_data[index][1])
                    index = index + 1
                tmp_line = np.stack((np.array(tmp_x),np.array(tmp_y)),axis=-1)
                tmp_y = []
                tmp_x = []
                ground_line.append(tmp_line)
            exist = getValue(pred)
            if np.sum(exist)<len(len_lane):
                FN = len(len_lane) - np.sum(exist) + FN
            if pred[0] == 1:
                tmp_L2Coors = getValue(L2Coors[i])
                tmp_L2Coors = tmp_L2Coors.reshape(p.point_num, 2)
                tmp_L2Coors = 0.5 * ((tmp_L2Coors + 1) * image_size - 1)
                tmp_L2Coors[:, 1] = tmp_L2Coors[:, 1] / ratio_h
                tmp_L2Coors[:, 0] = tmp_L2Coors[:, 0] / ratio_w
                chamfer_loss = []
                for k in range(len(len_lane)):
                    chamfer_loss.append(getValue(Chamfer_Loss(tmp_L2Coors,ground_line[k])))
                index = np.argmin(np.array(chamfer_loss))
                target_line = ground_line[index]
                z = np.polyfit(tmp_L2Coors[:,1],tmp_L2Coors[:,0],2)
                fun = np.poly1d(z)
                target_y = ground_line[index][:,1]
                target_x = fun(target_y).reshape(len(target_y),1)
                binary_image_target = np.zeros((image_data.shape[1], image_data.shape[2],1),np.uint8)
                binary_image_gt = np.zeros((image_data.shape[1], image_data.shape[2], 1), np.uint8)
                for k in range(len(target_x)-1):
                    cv2.line(binary_image_target,(int(target_x[k]),int(target_y[k])),
                             (int(target_x[k+1]),int(target_y[k+1])),255,p.line_width)
                    cv2.line(binary_image_gt, (int(target_line[k][0]), int(target_line[k][1])),
                             (int(target_line[k + 1][0]), int(target_line[k + 1][1])), 255, p.line_width)
                PR = binary_image_target.nonzero()[0].size
                GT = binary_image_gt.nonzero()[0].size
                TP_point = (binary_image_target*binary_image_gt).nonzero()[0].size
                union = PR + GT - TP_point
                iou = TP_point/union
                if iou>=p.threshold_iou:
                    TP = TP + 1
                else:
                    FP = FP + 1
            if pred[1] == 1:
                tmp_L1Coors = getValue(L1Coors[i])
                tmp_L1Coors = tmp_L1Coors.reshape(p.point_num, 2)
                tmp_L1Coors = 0.5 * ((tmp_L1Coors + 1) * image_size - 1)
                tmp_L1Coors[:, 1] = tmp_L1Coors[:, 1] / ratio_h
                tmp_L1Coors[:, 0] = tmp_L1Coors[:, 0] / ratio_w
                chamfer_loss = []
                for k in range(len(len_lane)):
                    chamfer_loss.append(getValue(Chamfer_Loss(tmp_L1Coors, ground_line[k])))
                index = np.argmin(np.array(chamfer_loss))
                target_line = ground_line[index]
                z = np.polyfit(tmp_L1Coors[:, 1], tmp_L1Coors[:, 0], 2)
                fun = np.poly1d(z)
                target_y = ground_line[index][:, 1]
                target_x = fun(target_y).reshape(len(target_y), 1)
                binary_image_target = np.zeros((image_data.shape[1], image_data.shape[2], 1), np.uint8)
                binary_image_gt = np.zeros((image_data.shape[1], image_data.shape[2], 1), np.uint8)
                for k in range(len(target_x)-1):
                    cv2.line(binary_image_target, (int(target_x[k]), int(target_y[k])),
                             (int(target_x[k + 1]), int(target_y[k + 1])), 255, p.line_width)
                    cv2.line(binary_image_gt, (int(target_line[k][0]), int(target_line[k][1])),
                             (int(target_line[k + 1][0]), int(target_line[k + 1][1])), 255, p.line_width)
                PR = binary_image_target.nonzero()[0].size
                GT = binary_image_gt.nonzero()[0].size
                TP_point = (binary_image_target * binary_image_gt).nonzero()[0].size
                union = PR + GT - TP_point
                iou = TP_point / union
                if iou >= p.threshold_iou:
                    TP = TP + 1
                else:
                    FP = FP + 1
            if pred[2] == 1:
                tmp_R1Coors = getValue(R1Coors[i])
                tmp_R1Coors = tmp_R1Coors.reshape(p.point_num, 2)
                tmp_R1Coors = 0.5 * ((tmp_R1Coors + 1) * image_size - 1)
                tmp_R1Coors[:, 1] = tmp_R1Coors[:, 1] / ratio_h
                tmp_R1Coors[:, 0] = tmp_R1Coors[:, 0] / ratio_w
                chamfer_loss = []
                for k in range(len(len_lane)):
                    chamfer_loss.append(getValue(Chamfer_Loss(tmp_R1Coors, ground_line[k])))
                index = np.argmin(np.array(chamfer_loss))
                target_line = ground_line[index]
                z = np.polyfit(tmp_R1Coors[:, 1], tmp_R1Coors[:, 0], 2)
                fun = np.poly1d(z)
                target_y = ground_line[index][:, 1]
                target_x = fun(target_y).reshape(len(target_y), 1)
                binary_image_target = np.zeros((image_data.shape[1], image_data.shape[2], 1), np.uint8)
                binary_image_gt = np.zeros((image_data.shape[1], image_data.shape[2], 1), np.uint8)
                for k in range(len(target_x)-1):
                    cv2.line(binary_image_target, (int(target_x[k]), int(target_y[k])),
                             (int(target_x[k + 1]), int(target_y[k + 1])), 255, p.line_width)
                    cv2.line(binary_image_gt, (int(target_line[k][0]), int(target_line[k][1])),
                             (int(target_line[k + 1][0]), int(target_line[k + 1][1])), 255, p.line_width)
                PR = binary_image_target.nonzero()[0].size
                GT = binary_image_gt.nonzero()[0].size
                TP_point = (binary_image_target * binary_image_gt).nonzero()[0].size
                union = PR + GT - TP_point
                iou = TP_point / union
                if iou >= p.threshold_iou:
                    TP = TP + 1
                else:
                    FP = FP + 1
            if pred[3] == 1:
                tmp_R2Coors = getValue(R2Coors[i])
                tmp_R2Coors = tmp_R2Coors.reshape(p.point_num, 2)
                tmp_R2Coors = 0.5 * ((tmp_R2Coors + 1) * image_size - 1)
                tmp_R2Coors[:, 1] = tmp_R2Coors[:, 1] / ratio_h
                tmp_R2Coors[:, 0] = tmp_R2Coors[:, 0] / ratio_w
                chamfer_loss = []
                for k in range(len(len_lane)):
                    chamfer_loss.append(getValue(Chamfer_Loss(tmp_R2Coors, ground_line[k])))
                index = np.argmin(np.array(chamfer_loss))
                target_line = ground_line[index]
                z = np.polyfit(tmp_R2Coors[:, 1], tmp_R2Coors[:, 0], 2)
                fun = np.poly1d(z)
                target_y = ground_line[index][:, 1]
                target_x = fun(target_y).reshape(len(target_y), 1)
                binary_image_target = np.zeros((image_data.shape[1], image_data.shape[2], 1), np.uint8)
                binary_image_gt = np.zeros((image_data.shape[1], image_data.shape[2], 1), np.uint8)
                for k in range(len(target_x)-1):
                    cv2.line(binary_image_target, (int(target_x[k]), int(target_y[k])),
                             (int(target_x[k + 1]), int(target_y[k + 1])), 255, p.line_width)
                    cv2.line(binary_image_gt, (int(target_line[k][0]), int(target_line[k][1])),
                             (int(target_line[k + 1][0]), int(target_line[k + 1][1])), 255, p.line_width)
                PR = binary_image_target.nonzero()[0].size
                GT = binary_image_gt.nonzero()[0].size
                TP_point = (binary_image_target * binary_image_gt).nonzero()[0].size
                union = PR + GT - TP_point
                iou = TP_point / union
                if iou >= p.threshold_iou:
                    TP = TP + 1
                else:
                    FP = FP + 1

        else:
            exist = getValue(pred)
            FP = FP + np.sum(exist)

    return TP, FP, FN


def LaneEval_SCNN(image_data, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, lines_data,len_lanes):
    '''
    Lane evaluate function relate to CULane
    '''
    p = Parameters()
    batch_size = pred_exist.shape[0]
    ratio_w = p.x_size * 1.0 / image_data.shape[2]
    ratio_h = p.y_size * 1.0 / image_data.shape[1]
    image_size = np.array([p.x_size,p.y_size]).reshape(1, 2)
    TP = 0
    FP = 0
    FN = 0
    for i in range(batch_size):
        line_data = lines_data[i]
        len_lane = len_lanes[i]
        tmp_x = []
        tmp_y = []
        ground_line = []
        length = len(line_data)
        pred = pred_exist[i] > 0.5
        if length>0:
            index = 0
            for m in range(len(len_lane)):
                for n in range(int(len_lane[m])):
                    tmp_x.append(line_data[index][0])
                    tmp_y.append(line_data[index][1])
                    index = index + 1
                tmp_line = np.stack((np.array(tmp_x),np.array(tmp_y)),axis=-1)
                tmp_y = []
                tmp_x = []
                ground_line.append(tmp_line)
            pred_lane = []
            if pred[0] == 1:
                tmp_L2Coors = getValue(L2Coors[i])
                tmp_L2Coors = tmp_L2Coors.reshape(p.point_num, 2)
                tmp_L2Coors = 0.5 * ((tmp_L2Coors + 1) * image_size - 1)
                tmp_L2Coors[:, 1] = tmp_L2Coors[:, 1] / ratio_h
                tmp_L2Coors[:, 0] = tmp_L2Coors[:, 0] / ratio_w
                pred_lane.append(tmp_L2Coors)
            if pred[1] == 1:
                tmp_L1Coors = getValue(L1Coors[i])
                tmp_L1Coors = tmp_L1Coors.reshape(p.point_num, 2)
                tmp_L1Coors = 0.5 * ((tmp_L1Coors + 1) * image_size - 1)
                tmp_L1Coors[:, 1] = tmp_L1Coors[:, 1] / ratio_h
                tmp_L1Coors[:, 0] = tmp_L1Coors[:, 0] / ratio_w
                pred_lane.append(tmp_L1Coors)
            if pred[2] == 1:
                tmp_R1Coors = getValue(R1Coors[i])
                tmp_R1Coors = tmp_R1Coors.reshape(p.point_num, 2)
                tmp_R1Coors = 0.5 * ((tmp_R1Coors + 1) * image_size - 1)
                tmp_R1Coors[:, 1] = tmp_R1Coors[:, 1] / ratio_h
                tmp_R1Coors[:, 0] = tmp_R1Coors[:, 0] / ratio_w
                pred_lane.append(tmp_R1Coors)
            if pred[3] == 1:
                tmp_R2Coors = getValue(R2Coors[i])
                tmp_R2Coors = tmp_R2Coors.reshape(p.point_num, 2)
                tmp_R2Coors = 0.5 * ((tmp_R2Coors + 1) * image_size - 1)
                tmp_R2Coors[:, 1] = tmp_R2Coors[:, 1] / ratio_h
                tmp_R2Coors[:, 0] = tmp_R2Coors[:, 0] / ratio_w
                pred_lane.append(tmp_R2Coors)

            similarity = np.zeros((len(len_lane),len(pred_lane)))
            for m in range(len(len_lane)):
                tmp_lane = ground_line[m]
                for n in range(len(pred_lane)):
                    tmp_pred_lane = pred_lane[n]
                    # z = np.polyfit(tmp_pred_lane[:, 1], tmp_pred_lane[:, 0], 2)
                    # fun = np.poly1d(z)
                    target_y = tmp_pred_lane[:, 1]
                    # target_x = fun(target_y).reshape(len(target_y), 1)
                    target_x = tmp_pred_lane[:, 0]
                    # f = interpolate.interp1d(target_y, target_x, kind="quadratic")
                    # start = np.min(target_y)
                    # end = np.max(target_y)
                    # target_y = np.linspace(start,end,20)
                    # target_x = f(target_y)
                    binary_image_target = np.zeros((image_data.shape[1], image_data.shape[2], 1), np.uint8)
                    binary_image_gt = np.zeros((image_data.shape[1], image_data.shape[2], 1), np.uint8)
                    for k in range(len(target_x) - 1):
                        cv2.line(binary_image_target, (int(target_x[k]), int(target_y[k])),
                                 (int(target_x[k + 1]), int(target_y[k + 1])), 255, p.line_width)

                    for k in range(len(tmp_lane) - 1):
                        cv2.line(binary_image_gt, (int(tmp_lane[k][0]), int(tmp_lane[k][1])),
                                 (int(tmp_lane[k + 1][0]), int(tmp_lane[k + 1][1])), 255, p.line_width)

                    PR = binary_image_target.nonzero()[0].size
                    GT = binary_image_gt.nonzero()[0].size
                    TP_point = (binary_image_target * binary_image_gt).nonzero()[0].size
                    union = PR + GT - TP_point
                    iou = TP_point / union
                    similarity[m][n] = -iou

            row_ind, col_ind = linear_sum_assignment(similarity)
            tmp_tp = 0
            for m in range(len(row_ind)):
                if np.abs(similarity[row_ind[m],col_ind[m]]) >= p.threshold_iou:
                    tmp_tp = tmp_tp + 1

            TP = TP + tmp_tp
            FP = len(pred_lane) - tmp_tp + FP
            FN = FN + len(len_lane) - tmp_tp

        else:
            exist = getValue(pred)
            FP = FP + np.sum(exist)

    return TP, FP, FN


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

def Chamfer_Loss(pred,gt):
    '''
    calculate two lines' chamfer loss
    '''
    M = gt.shape[0]
    N = pred.shape[0]
    gt = torch.Tensor(gt).cuda()
    pred = torch.Tensor(pred).cuda()

    target_points_expand = gt.unsqueeze(1).expand(M, N, 2)
    actual_points_expand = pred.unsqueeze(0).expand(M, N, 2)

    diff = torch.norm(target_points_expand - actual_points_expand, dim=2, keepdim=False)

    target_actual_min_dist,_ = torch.min(diff, dim=1, keepdim=False)
    forward_loss = target_actual_min_dist.mean()

    actual_target_min_dist,_ = torch.min(diff, dim=0, keepdim=False)
    backward_loss = actual_target_min_dist.mean()

    chamfer_pure = forward_loss + backward_loss

    return chamfer_pure
