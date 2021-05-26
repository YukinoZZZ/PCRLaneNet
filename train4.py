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
#mport csaps

from Model.model import Lane_Detection,resnet_lane_detection, resnet_lane_detection_simple, SE_Resnet_Model
from Model.model_with_dilation import Fushion_model, Lane_Detection_CPLUS

#import visdom
from tensorboardX import SummaryWriter
from Parameter import Parameters
from data_loader import Generator
from tools.loss import calculate_exist_loss,calculate_point_loss
from tools.MultiStepLR import MultiStepLR


##python -m visdom.server
p=Parameters()
writer = SummaryWriter('/home/doujian/Desktop/NCR_log')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device_id = [0,3]
cuda_avaliable = torch.cuda.is_available()
d_step = 1

def Training():
    print('Training')

    print("Get dataset")
    loader = Generator()

    model = Fushion_model(in_channels=512, point_feature_channels=32, point_number=10)
    #model = Lane_Detection_CPLUS()
    if cuda_avaliable:
        model.cuda()
    if len(device_id)>1:
        model = torch.nn.DataParallel(model)
    #model_path = '/home/doujian/Desktop/savemodel/15_0.471724271774292_SGD_lane_detection_network.pkl'
    # model = load_GPUS(model, model_path)
    # model.load_state_dict(torch.load('/data1/alpha-doujian/resnet101_models/5_1.095084547996521_SGD_lane_detection_network.pkl'))

    weight_decay = p.weight_decay
    # if p.weight_decay > 0:
    #     parameters = add_weight_decay(model, weight_decay=p.weight_decay)
    #     weight_decay = 0
    # else:
    #     parameters = model.parameters()
    optim = torch.optim.Adam(model.parameters(), lr=6e-5, weight_decay=weight_decay)
    #optim = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9,weight_decay=weight_decay)
    scheduler = MultiStepLR(optim, iters_per_epoch=5555*10, warmup='linear',warmup_iters=1000)
##optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)
    step = 0#96077

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    for epoch in range(p.n_epoch):
        model.train()
        # epoch = epoch + 19
        for raw_image, target_R2, target_R1, target_L2, target_L1, exist_label, test_image in loader.Generate():
            # training
            print("epoch : " + str(epoch))
            print("step : " + str(step))
            image_size = [p.x_size, p.y_size]
            raw_image = np.array(raw_image)
            raw_image = torch.Tensor(raw_image).cuda()
            image_data = Variable(raw_image)
            test_image = np.array(test_image).reshape(1,3,p.y_size,p.x_size)
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

            pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps, \
            midR2Coors, midR2HeapMaps, midR1Coors, midR1HeapMaps, midL1Coors, midL1HeapMaps, midL2Coors, midL2HeapMaps = model(
                image_data)
            # pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps,  = model(
            #     image_data)



            loss_exist = calculate_exist_loss(pred_exist,exist_label)
            loss_point = calculate_point_loss(R2Coors, target_R2, R2HeapMaps, R1Coors, target_R1,
                                              R1HeapMaps, L2Coors, target_L2, L2HeapMaps, L1Coors, target_L1, L1HeapMaps, exist_label)
            midloss_point = calculate_point_loss(midR2Coors, target_R2, midR2HeapMaps, midR1Coors, target_R1,
                                                 midR1HeapMaps, midL2Coors, target_L2, midL2HeapMaps, midL1Coors, target_L1,
                                                 midL1HeapMaps, exist_label)
            loss = p.alpha * loss_exist + loss_point

            # if p.flood_level>0:
            #     loss = (loss-p.flood_level).abs()+p.flood_level

            writer.add_scalar('/home/doujian/Desktop/NCR_log/loss_exist', loss_exist, step)
            writer.add_scalar('/home/doujian/Desktop/NCR_log/loss_point', loss_point, step)
            writer.add_scalar('/home/doujian/Desktop/NCR_log/midloss_point', midloss_point, step)
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
            midloss_point.backward(retain_graph=True)
            loss.backward()
            optim.step()
            scheduler.step()
            print(optim.param_groups[0]['lr'])


            loss_ = getValue(loss)

            if step % 1000 == 0 and step > 0:
                if epoch<30:
                    save_model(model, p, epoch, loss_)
            # if loss_< 0.15:
            #     save_model(model, p, epoch, loss_)


            if step % 1000 == 0:
                if epoch > 30:
                    test(model,test_image,step,loss_)
                    save_model(model, p, epoch, loss_)



            # if step % 500 == 0:
            #     if epoch >= 30:
            #         #test2(image_data, pred_exist, R2Coors, R1Coors, L1Coors, L2Coors, step, loss)
            #         save_model(model, p, epoch, loss_)
                    # print("starting write result----------")
                    # write_result(loader,model)
                    # print("epoch:{:d}".format(epoch))
                    # print("model loss:{:.3f}".format(loss_))
                    # print("write result done! starting calculating F1-Scores---------")
                    # os.system("sh /home/doujian/Desktop/lane_evaluation/run.sh")





            step = step + 1
            #adjust_learning_rate(optim,p.l_rate,step, p)


            del pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps, \
            midR2Coors, midR2HeapMaps, midR1Coors, midR1HeapMaps, midL1Coors, midL1HeapMaps, midL2Coors, midL2HeapMaps,\
            loss, loss_exist, loss_point, midloss_point

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

def write_result(loader,model):
    step = 0
    test_file = '/home/doujian/Desktop/CULane/list/' + 'test0_normal.txt'
    test_list = []
    save_root = '/home/doujian/Desktop/test/'
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

        pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps = model(
            image_data)

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
            #tmp_R2Coors = fitting((tmp_R2Coors))
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
            #tmp_R1Coors = fitting((tmp_R1Coors))
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
            #tmp_L1Coors = fitting((tmp_L1Coors))
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
            #tmp_L2Coors = fitting((tmp_L2Coors))
            with open(tmp_txt, 'a') as f:
                for ii in range(tmp_L2Coors.shape[0]):
                    f.write(str(tmp_L2Coors[ii, 0]))
                    f.write(' ')
                    f.write(str(tmp_L2Coors[ii, 1]))
                    f.write(' ')
                f.write('\n')
            f.close()

        step = step + 1

def fitting(Coors):
    sort_index = np.argsort(Coors[:, 1])
    Coors = Coors[sort_index]
    sp = csaps.CubicSmoothingSpline(Coors[:,1],Coors[:,0])
    min_y = min(Coors[:,1])
    max_y = max(Coors[:,1])

    last = 0
    last_second = 0
    last_y = 0
    last_second_y = 0
    temp_x = []
    temp_y = []
    for pts in range(62,-1,-1):
        h = 590-5*pts -1
        temp_y.append(h)
        if h<min_y:
            temp_x.append(-2)
        elif min_y<=h and h<=max_y:
            temp_x.append(sp([h])[0])
            last = temp_x[-1]
            last_y = temp_y[-1]
            if len(temp_x)<2:
                last_second = temp_x[-1]
                last_second_y = temp_y[-1]
            else:
                last_second = temp_x[-2]
                last_second_y = temp_y[-2]
        else:
            if last<last_second:
                l = int(last_second - float(-last_second_y+h)*abs(last_second-last)/abs(last_second_y+0.0001-last_y))
                if l>1640 or l<0:
                    temp_x.append(-2)
                else:
                    temp_x.append(l)
            else:
                l = int(last_second + float(-last_second_y+h)*abs(last_second-last)/abs(last_second_y+0.0001-last_y))
                if l>1640 or l<0:
                    temp_x.append(-2)
                else:
                    temp_x.append(l)

    temp_x = np.array(temp_x)
    temp_y = np.array(temp_y)
    temp_y = temp_y[temp_x!=-2]
    temp_x = temp_x[temp_x!=-2]

    fit_Coors = np.stack((temp_x,temp_y),axis=-1)

    return fit_Coors

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
    Training()