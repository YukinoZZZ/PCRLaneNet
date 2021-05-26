import numpy as np
from scipy import interpolate
from Parameter import Parameters

def fit_point(tmp_x,tmp_y,p):
    if tmp_x[0] > tmp_x[1]:
        end = min(tmp_x)
        index = tmp_x.index(min(tmp_x))
    else:
        end = max(tmp_x)
        index = tmp_x.index(max(tmp_x))
    start1 = 0
    start2 = 0
    for i in range(len(tmp_x)):
        if tmp_x[i] > 0 and tmp_x[i] < 800 and tmp_y[i] > 0 and tmp_y[i] < 288:
            start1 = tmp_x[i]
            break

    for i in range(len(tmp_x)):
        if tmp_x[len(tmp_x) - i - 1] > 0 and tmp_x[len(tmp_x) - i - 1] < 800 and tmp_y[len(tmp_x) - i - 1] > 0 and \
                tmp_y[len(tmp_x) - i - 1] < 288:
            start2 = tmp_x[len(tmp_x) - i - 1]
            break


    xn1 = np.linspace(start1, end, int(np.floor(index/p.ground_truth_num*p.point_num)))
    xn2 = np.linspace(end, start2, p.point_num-int(np.floor(index/p.ground_truth_num*p.point_num))+1)
    if len(tmp_x[0:index + 1])>2:
        fn1 = interpolate.interp1d(tmp_x[0:index + 1], tmp_y[0:index + 1], 'quadratic')
    else:
        fn1 = interpolate.interp1d(tmp_x[0:index + 1], tmp_y[0:index + 1], 'slinear')
    if len(tmp_x[index:])>2:
        fn2 = interpolate.interp1d(tmp_x[index:], tmp_y[index:], 'quadratic')
    else:
        fn2 = interpolate.interp1d(tmp_x[index:], tmp_y[index:], 'slinear')
    yn1 = fn1(xn1)
    yn2 = fn2(xn2)
    xn = [i for i in xn1]
    yn = [i for i in yn1]
    for i in range(len(xn2) - 1):
        xn.append(xn2[i + 1])
        yn.append(yn2[i + 1])
    return np.array(xn), np.array(yn)