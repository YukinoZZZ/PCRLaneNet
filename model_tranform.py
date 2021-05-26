import torch
import torchvision
from Model.model import Lane_Detection,Lane_exist, Lane_Detection_CPLUS
from Model.model_with_dilation import Fushion_model
import os

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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# device_id = [0,1]
# cuda_avaliable = torch.cuda.is_available()
model = Fushion_model(in_channels=512, point_feature_channels=32, point_number=15)
model_path = '/data1/alpha-doujian/save_model_wp/dilated_resnet34/90560.pkl'
model = load_GPUS(model, model_path)
# if cuda_avaliable:
#     model.cuda()
# if len(device_id) > 1:
#     model = torch.nn.DataParallel(model)

model.cuda()
example = torch.rand(1,3,288,800).cuda()

model.eval()


traced_script_module = torch.jit.trace(model,example)
pred_exist, R2Coors, R2HeapMaps, R1Coors, R1HeapMaps, L1Coors, L1HeapMaps, L2Coors, L2HeapMaps, \
            midR2Coors, midR2HeapMaps, midR1Coors, midR1HeapMaps, midL1Coors, midL1HeapMaps, midL2Coors, midL2HeapMaps = traced_script_module(example)
print(pred_exist)

traced_script_module.save('/data1/alpha-doujian/lane_detection_test.pt')



