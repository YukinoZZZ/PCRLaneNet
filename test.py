import os
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import time


from tools.evaluation import LaneEval, LaneEval_SCNN
from Parameter import Parameters
from data_loader import Generator
from Model.model_with_dilation import Fushion_model,Lane_Detection_CPLUS

from scipy.optimize import linear_sum_assignment
from scipy import interpolate
from Model.New_Fushion_Model import Fushion_Lane_detection

##python -m visdom.server
p=Parameters()

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device_id = [0]
cuda_avaliable = torch.cuda.is_available()
##first test 0.846
def Testing():
    print('Testing')

    print("Get dataset")
    loader = Generator()

    model = Fushion_model(in_channels=512, point_feature_channels=32, point_number=15)
    # model_no_fusion = Lane_Detection_CPLUS()
    if cuda_avaliable:
        model.cuda()
        # model_no_fusion.cuda()
    if len(device_id)>1:
        model = torch.nn.DataParallel(model)
        # model_no_fusion = torch.nn.DataParallel(model_no_fusion)
    model_path = '/data1/alpha-doujian/resnet101_models/' \
                 '12_0.6104534268379211_SGD_lane_detection_network.pkl'
    model = load_GPUS(model, model_path)
    # model.load_state_dict(torch.load('/home/doujian/Desktop/model/ERFNet_encoder/'
    #                                  '7_0.897279679775238_SGD_lane_detection_network.pkl'))#epoch22 loss0.37:0.862
    #loss 0.49:0.881 loss 0.25:0.892
    # model_no_fusion_path = '/data1/alpha-doujian/save_model_wp/dilated_resnet34_nofusion' \
    #                        '/12_0.37931859493255615_SGD_lane_detection_network.pkl'
    # model_no_fusion.load_state_dict(torch.load(model_no_fusion_path))
    #model_no_fusion = load_GPUS(model_no_fusion,model_no_fusion_path)

    ##resnet34 dilated loss0.75:0.897 loss 0.53:0.90


    ##############################l
    ## Loop for training
    ##############################
    print('Testing loop')

    # step = 0
    # for raw_image ,line_datas, len_lanes, image in loader.Generate_Test():
    #     model.eval()
    #     #model_no_fusion.eval()
    #         # testing
    #     raw_image = np.array(raw_image).reshape(1,3,288,800)
    #     raw_image = torch.Tensor(raw_image).cuda()
    #     image_data = Variable(raw_image)
    #
    #     pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps, \
    #     midR2Coors, midR2HeapMaps, midR1Coors, midR1HeapMaps, midL1Coors, midL1HeapMaps, midL2Coors, midL2HeapMaps = model(
    #         image_data)
    #     # feature_display(L2HeapMaps,L1HeapMaps,R1HeapMaps,R2HeapMaps,pred_exist,step,'fusion')
    #     test(image_data, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, line_datas, len_lanes, step)
    #     # pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps = model_no_fusion(image_data)
    #     # test_no_fusion(image_data, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, line_datas, len_lanes, step)
    #     # feature_display(L2HeapMaps, L1HeapMaps, R1HeapMaps, R2HeapMaps, pred_exist, step, 'no_fusion')
    #     # #test(image_data, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, line_datas, len_lanes, step)
    #     plot_gt(image_data,line_datas,len_lanes,step)
    #     # if step>=1000:
    #     #     test(image_data, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, line_datas, len_lanes, step)
    #
    #     step = step + 1
    #     if step>=1000:
    #         break
    # TP = 0
    # FP = 0
    # FN = 0
    # for raw_image, lines_data, len_lanes, images in loader.Generate_batch_test():
    #     model.eval()
    #     raw_image = np.array(raw_image)
    #     raw_image = torch.Tensor(raw_image).cuda()
    #     image_data = Variable(raw_image)
    #
    #     pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps = model(
    #         image_data)
    #
    #     mtp,mfp,mfn = LaneEval_SCNN(images, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, lines_data, len_lanes)
    #     TP = TP + mtp
    #     FP = FP + mfp
    #     FN = FN + mfn
    #     print("TP:{0},FP:{1},FN:{2}".format(TP,FP,FN))
    #
    #     Precision = TP / (TP + FP)
    #     Recall = TP / (TP + FN)
    #     beta = 1
    #     F1 = (1 + beta ** 2) * (Precision * Recall) / (beta ** 2 * (Precision + Recall))
    #     print("Precision:{:.3f}, Recall:{:.3f}, F1-measure:{:.3f}".format(Precision, Recall, F1))
    #
    # Precision = TP/(TP+FP)
    # Recall = TP/(TP+FN)
    # beta = 1
    # F1 = (1+beta**2)*(Precision*Recall)/(beta**2*(Precision+Recall))
    # print("Precision:{:.3f}, Recall:{:.3f}, F1-measure:{:.3f}".format(Precision,Recall,F1))

    step = 0
    test_file = '/home/doujian/Desktop/CULane/list/' + 'test0_normal.txt'
    test_list = []
    save_root = '/data1/alpha-doujian/test/'
    del_files(save_root)
    with open(test_file) as f:
        for _info in f:
            info_tmp = _info.strip(' ').split()
            test_list.append(info_tmp)

    for raw_image, line_datas, len_lanes, image in loader.Generate_Test():
        model.eval()
        # testing
        image_size = np.array([800, 288]).reshape(1, 2)
        ratio_w = p.x_size * 1.0 / 1640
        ratio_h = p.y_size * 1.0 / 590
        raw_image = np.array(raw_image).reshape(1, 3, 288, 800)
        raw_image = torch.Tensor(raw_image).cuda()
        image_data = Variable(raw_image)

        pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps, \
        midR2Coors, midR2HeapMaps, midR1Coors, midR1HeapMaps, midL1Coors, midL1HeapMaps, midL2Coors, midL2HeapMaps = model(
            image_data)
        # pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps = model(
        #     image_data)

        tmp_test_file = test_list[step]
        tmp_save_root = save_root + tmp_test_file[0][:-9]
        if not os.path.exists(tmp_save_root):
            os.makedirs(tmp_save_root)
        tmp_txt = tmp_save_root + tmp_test_file[0][-9:-4] + '.lines.txt'
        pred = pred_exist[0] > 0.5
        if pred[3] == 1:
            tmp_R2Coors = getValue(R2Coors[0])
            tmp_R2Coors = tmp_R2Coors.reshape(p.point_num, 2)
            tmp_R2Coors = 0.5 * ((tmp_R2Coors + 1) * image_size - 1)
            sort_index = np.argsort(tmp_R2Coors[:, 1])
            tmp_R2Coors = tmp_R2Coors[sort_index]
            tmp_R2Coors[:, 0] = tmp_R2Coors[:, 0] / ratio_w
            tmp_R2Coors[:, 1] = tmp_R2Coors[:, 1] / ratio_h
            # tmp_R2Coors = fitting((tmp_R2Coors))
            with open(tmp_txt, 'a') as f:
                for ii in range(tmp_R2Coors.shape[0]):
                    f.write(str(tmp_R2Coors[ii, 0]))
                    f.write(' ')
                    f.write(str(tmp_R2Coors[ii, 1]))
                    f.write(' ')
                f.write('\n')
            f.close()
        if pred[2] == 1:
            tmp_R1Coors = getValue(R1Coors[0])
            tmp_R1Coors = tmp_R1Coors.reshape(p.point_num, 2)
            tmp_R1Coors = 0.5 * ((tmp_R1Coors + 1) * image_size - 1)
            sort_index = np.argsort(tmp_R1Coors[:, 1])
            tmp_R1Coors = tmp_R1Coors[sort_index]
            tmp_R1Coors[:, 0] = tmp_R1Coors[:, 0] / ratio_w
            tmp_R1Coors[:, 1] = tmp_R1Coors[:, 1] / ratio_h
            # tmp_R1Coors = fitting((tmp_R1Coors))
            with open(tmp_txt, 'a') as f:
                for ii in range(tmp_R1Coors.shape[0]):
                    f.write(str(tmp_R1Coors[ii, 0]))
                    f.write(' ')
                    f.write(str(tmp_R1Coors[ii, 1]))
                    f.write(' ')
                f.write('\n')
            f.close()
        if pred[1] == 1:
            tmp_L1Coors = getValue(L1Coors[0])
            tmp_L1Coors = tmp_L1Coors.reshape(p.point_num, 2)
            tmp_L1Coors = 0.5 * ((tmp_L1Coors + 1) * image_size - 1)
            sort_index = np.argsort(tmp_L1Coors[:, 1])
            tmp_L1Coors = tmp_L1Coors[sort_index]
            tmp_L1Coors[:, 0] = tmp_L1Coors[:, 0] / ratio_w
            tmp_L1Coors[:, 1] = tmp_L1Coors[:, 1] / ratio_h
            # tmp_L1Coors = fitting((tmp_L1Coors))
            with open(tmp_txt, 'a') as f:
                for ii in range(tmp_L1Coors.shape[0]):
                    f.write(str(tmp_L1Coors[ii, 0]))
                    f.write(' ')
                    f.write(str(tmp_L1Coors[ii, 1]))
                    f.write(' ')
                f.write('\n')
            f.close()
        if pred[0] == 1:
            tmp_L2Coors = getValue(L2Coors[0])
            tmp_L2Coors = tmp_L2Coors.reshape(p.point_num, 2)
            tmp_L2Coors = 0.5 * ((tmp_L2Coors + 1) * image_size - 1)
            sort_index = np.argsort(tmp_L2Coors[:, 1])
            tmp_L2Coors = tmp_L2Coors[sort_index]
            tmp_L2Coors[:, 0] = tmp_L2Coors[:, 0] / ratio_w
            tmp_L2Coors[:, 1] = tmp_L2Coors[:, 1] / ratio_h
            # tmp_L2Coors = fitting((tmp_L2Coors))
            with open(tmp_txt, 'a') as f:
                for ii in range(tmp_L2Coors.shape[0]):
                    f.write(str(tmp_L2Coors[ii, 0]))
                    f.write(' ')
                    f.write(str(tmp_L2Coors[ii, 1]))
                    f.write(' ')
                f.write('\n')
            f.close()

        step = step + 1

    os.system("sh /home/doujian/Desktop/lane_evaluation/run.sh")

def feature_display(lane1_feature, lane2_feature, lane3_feature, lane4_feature, pred_exist, index, type):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    lane1_feature = lane1_feature.cpu().detach().numpy()
    lane1_feature = lane1_feature[0]
    lane1_feature = np.sum(lane1_feature,axis=0)
    lane2_feature = lane2_feature.cpu().detach().numpy()
    lane2_feature = lane2_feature[0]
    lane2_feature = np.sum(lane2_feature, axis=0)
    lane3_feature = lane3_feature.cpu().detach().numpy()
    lane3_feature = lane3_feature[0]
    lane3_feature = np.sum(lane3_feature, axis=0)
    lane4_feature = lane4_feature.cpu().detach().numpy()
    lane4_feature = lane4_feature[0]
    lane4_feature = np.sum(lane4_feature, axis=0)
    lane_feature = np.zeros_like(lane1_feature)
    pred = pred_exist[0]>0.5
    if pred[0]==1:
        lane_feature = lane_feature + lane1_feature
    if pred[1]==1:
        lane_feature = lane_feature + lane2_feature
    if pred[2]==1:
        lane_feature = lane_feature + lane3_feature
    if pred[3]==1:
        lane_feature = lane_feature + lane4_feature
    ax = sns.heatmap(lane_feature,cmap='rainbow')
    fig = plt.figure(1)
    plt.savefig('test_result/image_'+type+str(index)+'.png')
    plt.close(fig)


def plot_gt(test_image, line_datas, len_lanes, step):
    batch_size = test_image.shape[0]

    for i in range(batch_size):
        image = getValue(test_image[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0)
        image = image * 255
        image = image.astype(np.uint8).copy()

        cv2.imwrite('train_result/raw_image' + str(step) + '_' + str(i) + '.png', image)

        line_data = line_datas[i]
        len_lane = len_lanes[i]
        tmp_x = []
        tmp_y = []
        ground_line = []
        length = len(line_data)
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

            color_index = 0
            for line in ground_line:
                for k in range(len(line)):
                    image = cv2.circle(image, (int(line[k, 0]), int(line[k, 1])), 5,
                                       p.color[color_index], -1)

                color_index = color_index + 1



        cv2.imwrite('train_result/result_' + str(step) + '_' + str(i) + '.png', image)






def test(test_image, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, line_datas, len_lanes, step):
    batch_size = pred_exist.shape[0]
    image_size = np.array([p.x_size,p.y_size]).reshape(1,2)

    for i in range(batch_size):
        color_index = 0
        image = getValue(test_image[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0)
        image = image * 255
        image = image.astype(np.uint8).copy()
        pred = pred_exist[i]>0.5

        line_data = line_datas[i]
        len_lane = len_lanes[i]
        tmp_x = []
        tmp_y = []
        ground_line = []
        length = len(line_data)
        if length > 0:
            point_index = 0
            for m in range(len(len_lane)):
                for n in range(int(len_lane[m])):
                    tmp_x.append(line_data[point_index][0])
                    tmp_y.append(line_data[point_index][1])
                    point_index = point_index + 1
                tmp_line = np.stack((np.array(tmp_x), np.array(tmp_y)), axis=-1)
                tmp_y = []
                tmp_x = []
                ground_line.append(tmp_line)
            pred_lane = []

            if pred[0] == 1:
                tmp_L2Coors = getValue(L2Coors[i])
                tmp_L2Coors = tmp_L2Coors.reshape(p.point_num,2)
                tmp_L2Coors = 0.5 * ((tmp_L2Coors + 1) * image_size - 1)
                for index in range(p.point_num):
                    image = cv2.circle(image, (int(tmp_L2Coors[index, 0]), int(tmp_L2Coors[index, 1])), 5,
                                       p.color[color_index], -1)
                color_index = color_index + 1
                pred_lane.append(tmp_L2Coors)

            if pred[1] == 1:
                tmp_L1Coors = getValue(L1Coors[i])
                tmp_L1Coors = tmp_L1Coors.reshape(p.point_num, 2)
                tmp_L1Coors = 0.5 * ((tmp_L1Coors + 1) * image_size - 1)
                for index in range(p.point_num):
                    image = cv2.circle(image, (int(tmp_L1Coors[index, 0]), int(tmp_L1Coors[index, 1])), 5,
                                       p.color[color_index], -1)
                color_index = color_index + 1
                pred_lane.append(tmp_L1Coors)

            if pred[2] == 1:
                tmp_R1Coors = getValue(R1Coors[i])
                tmp_R1Coors = tmp_R1Coors.reshape(p.point_num, 2)
                tmp_R1Coors = 0.5 * ((tmp_R1Coors + 1) * image_size - 1)
                for index in range(p.point_num):
                    image = cv2.circle(image, (int(tmp_R1Coors[index, 0]), int(tmp_R1Coors[index, 1])), 5,
                                       p.color[color_index], -1)
                color_index = color_index + 1
                pred_lane.append(tmp_R1Coors)

            if pred[3]==1:
                tmp_R2Coors = getValue(R2Coors[i])
                tmp_R2Coors = tmp_R2Coors.reshape(p.point_num,2)
                tmp_R2Coors = 0.5 * ((tmp_R2Coors + 1) * image_size - 1)
                for index in range(p.point_num):
                    image = cv2.circle(image, (int(tmp_R2Coors[index,0]), int(tmp_R2Coors[index,1])), 5, p.color[color_index], -1)
                pred_lane.append(tmp_R2Coors)



            #calculate iou
            # similarity = np.zeros((len(len_lane), len(pred_lane)))
            # for m in range(len(len_lane)):
            #     tmp_lane = ground_line[m]
            #     for n in range(len(pred_lane)):
            #         tmp_pred_lane = pred_lane[n]
            #         # z = np.polyfit(tmp_pred_lane[:, 1], tmp_pred_lane[:, 0], 2)
            #         # fun = np.poly1d(z)
            #         target_y = tmp_pred_lane[:, 1]
            #         # target_x = fun(target_y).reshape(len(target_y), 1)
            #         target_x = tmp_pred_lane[:, 0]
            #         f = interpolate.interp1d(target_y, target_x, kind="quadratic")
            #         start = np.min(target_y)
            #         end = np.max(target_y)
            #         target_y = np.linspace(start, end, 20)
            #         target_x = f(target_y)
            #         binary_image_target = np.zeros((p.y_size, p.x_size, 1), np.uint8)
            #         binary_image_gt = np.zeros((p.y_size, p.x_size, 1), np.uint8)
            #         for k in range(len(target_x) - 1):
            #             cv2.line(binary_image_target, (int(target_x[k]), int(target_y[k])),
            #                      (int(target_x[k + 1]), int(target_y[k + 1])), 255, p.line_width)
            #
            #         for k in range(len(tmp_lane) - 1):
            #             cv2.line(binary_image_gt, (int(tmp_lane[k][0]), int(tmp_lane[k][1])),
            #                      (int(tmp_lane[k + 1][0]), int(tmp_lane[k + 1][1])), 255, p.line_width)
            #
            #         PR = binary_image_target.nonzero()[0].size
            #         GT = binary_image_gt.nonzero()[0].size
            #         TP_point = (binary_image_target * binary_image_gt).nonzero()[0].size
            #         union = PR + GT - TP_point
            #         iou = TP_point / union
            #         similarity[m][n] = -iou
            #
            # row_ind, col_ind = linear_sum_assignment(similarity)
            # for m in range(len(row_ind)):
            #     if np.abs(similarity[row_ind[m], col_ind[m]]) < p.threshold_iou:
            #         cv2.imwrite('train_result/result_' + str(step) + '_' + str(similarity[row_ind[m], col_ind[m]]) + '.png', image)
            #         plot_gt(test_image, line_datas, len_lanes, step)

        cv2.imwrite('test_result/result_' + str(step) + '_' + str(i) + '.png', image)



def test_no_fusion(test_image, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, line_datas, len_lanes, step):
    batch_size = pred_exist.shape[0]
    image_size = np.array([p.x_size,p.y_size]).reshape(1,2)

    for i in range(batch_size):
        color_index = 0
        image = getValue(test_image[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0)
        image = image * 255
        image = image.astype(np.uint8).copy()
        pred = pred_exist[i]>0.5

        line_data = line_datas[i]
        len_lane = len_lanes[i]
        tmp_x = []
        tmp_y = []
        ground_line = []
        length = len(line_data)
        if length > 0:
            point_index = 0
            for m in range(len(len_lane)):
                for n in range(int(len_lane[m])):
                    tmp_x.append(line_data[point_index][0])
                    tmp_y.append(line_data[point_index][1])
                    point_index = point_index + 1
                tmp_line = np.stack((np.array(tmp_x), np.array(tmp_y)), axis=-1)
                tmp_y = []
                tmp_x = []
                ground_line.append(tmp_line)
            pred_lane = []

            if pred[0] == 1:
                tmp_L2Coors = getValue(L2Coors[i])
                tmp_L2Coors = tmp_L2Coors.reshape(p.point_num,2)
                tmp_L2Coors = 0.5 * ((tmp_L2Coors + 1) * image_size - 1)
                for index in range(p.point_num):
                    image = cv2.circle(image, (int(tmp_L2Coors[index, 0]), int(tmp_L2Coors[index, 1])), 5,
                                       p.color[color_index], -1)
                color_index = color_index + 1
                pred_lane.append(tmp_L2Coors)

            if pred[1] == 1:
                tmp_L1Coors = getValue(L1Coors[i])
                tmp_L1Coors = tmp_L1Coors.reshape(p.point_num,2)
                tmp_L1Coors = 0.5 * ((tmp_L1Coors + 1) * image_size - 1)
                for index in range(p.point_num):
                    image = cv2.circle(image, (int(tmp_L1Coors[index, 0]), int(tmp_L1Coors[index, 1])), 5,
                                       p.color[color_index], -1)
                color_index = color_index + 1
                pred_lane.append(tmp_L1Coors)

            if pred[2] == 1:
                tmp_R1Coors = getValue(R1Coors[i])
                tmp_R1Coors = tmp_R1Coors.reshape(p.point_num,2)
                tmp_R1Coors = 0.5 * ((tmp_R1Coors + 1) * image_size - 1)
                for index in range(p.point_num):
                    image = cv2.circle(image, (int(tmp_R1Coors[index, 0]), int(tmp_R1Coors[index, 1])), 5,
                                       p.color[color_index], -1)
                color_index = color_index + 1
                pred_lane.append(tmp_R1Coors)

            if pred[3] == 1:
                tmp_R2Coors = getValue(R2Coors[i])
                tmp_R2Coors = tmp_R2Coors.reshape(p.point_num, 2)
                tmp_R2Coors = 0.5 * ((tmp_R2Coors + 1) * image_size - 1)
                for index in range(p.point_num):
                    image = cv2.circle(image, (int(tmp_R2Coors[index, 0]), int(tmp_R2Coors[index, 1])), 5,
                                       p.color[color_index], -1)
                pred_lane.append(tmp_R2Coors)

            #calculate iou
            # similarity = np.zeros((len(len_lane), len(pred_lane)))
            # for m in range(len(len_lane)):
            #     tmp_lane = ground_line[m]
            #     for n in range(len(pred_lane)):
            #         tmp_pred_lane = pred_lane[n]
            #         # z = np.polyfit(tmp_pred_lane[:, 1], tmp_pred_lane[:, 0], 2)
            #         # fun = np.poly1d(z)
            #         target_y = tmp_pred_lane[:, 1]
            #         # target_x = fun(target_y).reshape(len(target_y), 1)
            #         target_x = tmp_pred_lane[:, 0]
            #         f = interpolate.interp1d(target_y, target_x, kind="quadratic")
            #         start = np.min(target_y)
            #         end = np.max(target_y)
            #         target_y = np.linspace(start, end, 20)
            #         target_x = f(target_y)
            #         binary_image_target = np.zeros((p.y_size, p.x_size, 1), np.uint8)
            #         binary_image_gt = np.zeros((p.y_size, p.x_size, 1), np.uint8)
            #         for k in range(len(target_x) - 1):
            #             cv2.line(binary_image_target, (int(target_x[k]), int(target_y[k])),
            #                      (int(target_x[k + 1]), int(target_y[k + 1])), 255, p.line_width)
            #
            #         for k in range(len(tmp_lane) - 1):
            #             cv2.line(binary_image_gt, (int(tmp_lane[k][0]), int(tmp_lane[k][1])),
            #                      (int(tmp_lane[k + 1][0]), int(tmp_lane[k + 1][1])), 255, p.line_width)
            #
            #         PR = binary_image_target.nonzero()[0].size
            #         GT = binary_image_gt.nonzero()[0].size
            #         TP_point = (binary_image_target * binary_image_gt).nonzero()[0].size
            #         union = PR + GT - TP_point
            #         iou = TP_point / union
            #         similarity[m][n] = -iou
            #
            # row_ind, col_ind = linear_sum_assignment(similarity)
            # for m in range(len(row_ind)):
            #     if np.abs(similarity[row_ind[m], col_ind[m]]) < p.threshold_iou:
            #         cv2.imwrite('train_result/result_' + str(step) + '_' + str(similarity[row_ind[m], col_ind[m]]) + '.png', image)
            #         plot_gt(test_image, line_datas, len_lanes, step)

        cv2.imwrite('nofusion/result_' + str(step) + '_' + str(i) + '.png', image)









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


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

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


if __name__ == '__main__':
    Testing()