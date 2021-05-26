from data_loader import Generator
import numpy as np
loader = Generator()
Allcurve = 0
curve = 0
for raw_image, target_R2, target_R1, target_L2, target_L1, exist_label, lane_kinds, test_image in loader.Generate():
    exist_label = np.array(exist_label)
    lane_kinds = np.array(lane_kinds)
    num = exist_label * lane_kinds
    curve = curve + np.sum(num)
    Allcurve = np.sum(exist_label) + Allcurve
    print('curve num all:{:.2f}'.format(Allcurve))
    print('curve num :{:.2f}'.format(curve))
