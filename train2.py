'''
NCR_LaneDetection train.py tensorboard version
by wangpan 2020.06.28
'''
import os
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

from Model.model import Lane_Detection,resnet_lane_detection, resnet_lane_detection_simple, SE_Resnet_Model
from Model.model_with_dilation import Fushion_Lane_detection

import visdom
from tensorboardX import SummaryWriter
from Parameter import Parameters
from data_loader import Generator
from tools.loss import calculate_exist_loss,calculate_point_loss, calculate_point_chamfer_loss, lane_emd_loss, calculate_point_chamfer_loss_with_lsq, calculate_point_l1_loss
from tools.MultiStepLR import MultiStepLR


##python -m visdom.server
p=Parameters()
writer = SummaryWriter('/home/doujian/Desktop/NCR_log')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device_id = [1,2,3]
cuda_avaliable = torch.cuda.is_available()
d_step = 1

def Training():
    print('Training')

    print("Get dataset")
    loader = Generator()


    model = Fushion_Lane_detection(in_channels=512,point_feature_channels=16,point_number=15,lane_fushion=True)
    if cuda_avaliable:
        model.cuda()
    if len(device_id)>1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(
        torch.load('/home/doujian/Desktop/savemodel/12_0.399679571390152_SGD_lane_detection_network.pkl'))

    weight_decay = p.weight_decay
    # if p.weight_decay > 0:
    #     parameters = add_weight_decay(model, weight_decay=p.weight_decay)
    #     weight_decay = 0
    # else:
    #     parameters = model.parameters()

    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    #optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=weight_decay)
    #scheduler = MultiStepLR(optim, iters_per_epoch=2777*10,warmup='linear',warmup_iters=1000)
##optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)
    step = 0

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    for epoch in range(p.n_epoch):
        model.train()
        for raw_image, target_R2, target_R1, target_L2, target_L1, exist_label, test_image in loader.Generate():
            # training
            print("epoch : " + str(epoch))
            print("step : " + str(step))
            image_size = [p.x_size, p.y_size]
            raw_image = np.array(raw_image)
            raw_image = torch.Tensor(raw_image).cuda()
            image_data = Variable(raw_image)
            test_image = np.array(test_image).reshape(1,3,288,800)
            test_image = torch.Tensor(test_image).cuda()
            test_image = Variable(test_image)
            target_R2 = np.array(target_R2)
            target_R2 = torch.Tensor(target_R2)
            target_R2 = (target_R2*2 + 1)/torch.Tensor(image_size) - 1
            target_R2 = target_R2.cuda()
            target_R2 = Variable(target_R2)
            target_R1 = np.array(target_R1)
            target_R1 = torch.Tensor(target_R1)
            target_R1 = (target_R1 * 2 + 1) / torch.Tensor(image_size) - 1
            target_R1 = target_R1.cuda()
            target_R1 = Variable(target_R1)
            target_L2 = np.array(target_L2)
            target_L2 = torch.Tensor(target_L2)
            target_L2 = (target_L2 * 2 + 1) / torch.Tensor(image_size) - 1
            target_L2 = target_L2.cuda()
            target_L2 = Variable(target_L2)
            target_L1 = np.array(target_L1)
            target_L1 = torch.Tensor(target_L1)
            target_L1 = (target_L1 * 2 + 1) / torch.Tensor(image_size) - 1
            target_L1 = target_L1.cuda()
            target_L1 = Variable(target_L1)
            exist_label = np.array(exist_label)
            exist_label = torch.Tensor(exist_label).cuda()
            exist_label = Variable(exist_label)

            pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps = model(
                image_data)



            loss_exist = calculate_exist_loss(pred_exist,exist_label)
            loss_point = calculate_point_loss(R2Coors, target_R2, R2HeapMaps, R1Coors, target_R1, R1HeapMaps, L2Coors, target_L2, L2HeapMaps, L1Coors, target_L1, L1HeapMaps, exist_label)
            loss = loss_exist + loss_point

            writer.add_scalar('/home/doujian/Desktop/NCR_log/loss_exist', loss_exist, step)
            writer.add_scalar('/home/doujian/Desktop/NCR_log/loss_point', loss_point, step)
            writer.add_scalar('/home/doujian/Desktop/NCR_log/loss', loss, step)

            if step%10 ==0:

                for image_index in range(raw_image.shape[0]):
                    color_index = 0
                    tmp_image_data = getValue(raw_image[image_index])
                    tmp_image_data = np.rollaxis(tmp_image_data, axis=2, start=0)
                    tmp_image_data = np.rollaxis(tmp_image_data, axis=2, start=0)
                    tmp_image_data = tmp_image_data * 255
                    b,g,r = cv2.split(tmp_image_data)
                    tmp_image_data = cv2.merge([r,g,b])
                    image = tmp_image_data.astype(np.uint8).copy()
                    pred = pred_exist[image_index] > 0.5
                    if pred[3] == 1:
                        tmp_R2Coors = getValue(R2Coors[image_index])
                        tmp_R2Coors = tmp_R2Coors.reshape(p.point_num, 2)
                        tmp_R2Coors = 0.5 * ((tmp_R2Coors + 1) * image_size - 1)
                        for index in range(p.point_num):
                            image = cv2.circle(image, (int(tmp_R2Coors[index, 0]), int(tmp_R2Coors[index, 1])), 5,
                                               p.color[color_index], -1)
                        color_index = color_index + 1
                    if pred[2] == 1:
                        tmp_R1Coors = getValue(R1Coors[image_index])
                        tmp_R1Coors = tmp_R1Coors.reshape(p.point_num, 2)
                        tmp_R1Coors = 0.5 * ((tmp_R1Coors + 1) * image_size - 1)
                        for index in range(p.point_num):
                            image = cv2.circle(image, (int(tmp_R1Coors[index, 0]), int(tmp_R1Coors[index, 1])), 5,
                                               p.color[color_index], -1)
                        color_index = color_index + 1
                    if pred[1] == 1:
                        tmp_L1Coors = getValue(L1Coors[image_index])
                        tmp_L1Coors = tmp_L1Coors.reshape(p.point_num, 2)
                        tmp_L1Coors = 0.5 * ((tmp_L1Coors + 1) * image_size - 1)
                        for index in range(p.point_num):
                            image = cv2.circle(image, (int(tmp_L1Coors[index, 0]), int(tmp_L1Coors[index, 1])), 5,
                                               p.color[color_index], -1)
                        color_index = color_index + 1

                    if pred[0] == 1:
                        tmp_L2Coors = getValue(L2Coors[image_index])
                        tmp_L2Coors = tmp_L2Coors.reshape(p.point_num, 2)
                        tmp_L2Coors = 0.5 * ((tmp_L2Coors + 1) * image_size - 1)
                        for index in range(p.point_num):
                            image = cv2.circle(image, (int(tmp_L2Coors[index, 0]), int(tmp_L2Coors[index, 1])), 5,
                                               p.color[color_index], -1)
                    name = '/home/doujian/Desktop/NCR_log/image_' + str(image_index)
                    figure = plt.figure('name')
                    plt.imshow(image)
                    writer.add_figure(name,figure,int(step/10))



            optim.zero_grad()
            loss.backward()
            optim.step()
            #scheduler.step()
            print(optim.param_groups[0]['lr'])


            loss_ = getValue(loss)

            if step % 1000 == 0 and step > 0:
                save_model(model, p, epoch, loss_)
                test(model, test_image, step, loss_)


            if step % 200 == 0:
                if epoch > 30:
                    test(model,test_image,step,loss_)


            if step % 500 == 0:
                if epoch > 30:
                    test2(image_data, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, step, loss)





            step = step + 1
            #adjust_learning_rate(optim,p.l_rate,step, p)


            del pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps,
            loss, loss_exist, loss_point,

    writer.close()



def test(model, test_image, step, loss):
    model.eval()
    pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps = model(
        test_image)
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
        if pred[3]==1:
            tmp_R2Coors = getValue(R2Coors[i])
            tmp_R2Coors = tmp_R2Coors.reshape(p.point_num,2)
            tmp_R2Coors = 0.5 * ((tmp_R2Coors + 1) * image_size - 1)
            for index in range(p.point_num):
                image = cv2.circle(image, (int(tmp_R2Coors[index,0]), int(tmp_R2Coors[index,1])), 5, p.color[color_index], -1)
            color_index = color_index + 1
        if pred[2] == 1:
            tmp_R1Coors = getValue(R1Coors[i])
            tmp_R1Coors = tmp_R1Coors.reshape(p.point_num,2)
            tmp_R1Coors = 0.5 * ((tmp_R1Coors + 1) * image_size - 1)
            for index in range(p.point_num):
                image = cv2.circle(image, (int(tmp_R1Coors[index, 0]), int(tmp_R1Coors[index, 1])), 5,
                                   p.color[color_index], -1)
            color_index = color_index + 1
        if pred[1] == 1:
            tmp_L1Coors = getValue(L1Coors[i])
            tmp_L1Coors = tmp_L1Coors.reshape(p.point_num,2)
            tmp_L1Coors = 0.5 * ((tmp_L1Coors + 1) * image_size - 1)
            for index in range(p.point_num):
                image = cv2.circle(image, (int(tmp_L1Coors[index, 0]), int(tmp_L1Coors[index, 1])), 5,
                                   p.color[color_index], -1)
            color_index = color_index + 1

        if pred[0] == 1:
            tmp_L2Coors = getValue(L2Coors[i])
            tmp_L2Coors = tmp_L2Coors.reshape(p.point_num,2)
            tmp_L2Coors = 0.5 * ((tmp_L2Coors + 1) * image_size - 1)
            for index in range(p.point_num):
                image = cv2.circle(image, (int(tmp_L2Coors[index, 0]), int(tmp_L2Coors[index, 1])), 5,
                                   p.color[color_index], -1)

        cv2.imwrite('test_result/result_' + str(step) + '_' + str(loss) + '_' + str(i) + '.png', image)

    model.train()


def test2(test_image, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, step, loss):
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
        if pred[3]==1:
            tmp_R2Coors = getValue(R2Coors[i])
            tmp_R2Coors = tmp_R2Coors.reshape(p.point_num,2)
            tmp_R2Coors = 0.5 * ((tmp_R2Coors + 1) * image_size - 1)
            for index in range(p.point_num):
                image = cv2.circle(image, (int(tmp_R2Coors[index,0]), int(tmp_R2Coors[index,1])), 5, p.color[color_index], -1)
            color_index = color_index + 1
        if pred[2] == 1:
            tmp_R1Coors = getValue(R1Coors[i])
            tmp_R1Coors = tmp_R1Coors.reshape(p.point_num,2)
            tmp_R1Coors = 0.5 * ((tmp_R1Coors + 1) * image_size - 1)
            for index in range(p.point_num):
                image = cv2.circle(image, (int(tmp_R1Coors[index, 0]), int(tmp_R1Coors[index, 1])), 5,
                                   p.color[color_index], -1)
            color_index = color_index + 1
        if pred[1] == 1:
            tmp_L1Coors = getValue(L1Coors[i])
            tmp_L1Coors = tmp_L1Coors.reshape(p.point_num,2)
            tmp_L1Coors = 0.5 * ((tmp_L1Coors + 1) * image_size - 1)
            for index in range(p.point_num):
                image = cv2.circle(image, (int(tmp_L1Coors[index, 0]), int(tmp_L1Coors[index, 1])), 5,
                                   p.color[color_index], -1)
            color_index = color_index + 1

        if pred[0] == 1:
            tmp_L2Coors = getValue(L2Coors[i])
            tmp_L2Coors = tmp_L2Coors.reshape(p.point_num,2)
            tmp_L2Coors = 0.5 * ((tmp_L2Coors + 1) * image_size - 1)
            for index in range(p.point_num):
                image = cv2.circle(image, (int(tmp_L2Coors[index, 0]), int(tmp_L2Coors[index, 1])), 5,
                                   p.color[color_index], -1)

        cv2.imwrite('train_result/result_' + str(step) + '_' + str(loss) + '_' + str(i) + '.png', image)

def adjust_learning_rate(optimizer, lr, step, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    lr = lr * (0.95**(step//args.lr_decay_every))
    print("current learning rate: {:.6f}".format(lr))
    param_group = optimizer.param_groups
    for i in range(len(param_group)):
        param_group[i]['lr'] = lr

    return optimizer


def save_model(model, p, epoch, loss):
    torch.save(
            model.state_dict(),
            p.save_path+str(epoch)+'_'+str(loss)+'_'+'SGD'+'_'+'lane_detection_network.pkl'
        )


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

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


if __name__ == '__main__':
    Training()