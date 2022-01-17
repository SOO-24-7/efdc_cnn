import numpy as np
from sklearn.metrics import r2_score # explained_variance_score, mean_squared_error, mean_absolute_error
from ai4water.postprocessing.SeqMetrics import RegressionMetrics

# for ss in np.arange(EFDC_tr.shape[0]): # 900개 (0~899)
#     EFDC_tr_sam = EFDC_tr[ss:ss+1,] # (1, 3, 3, 8) = (No. of sampel, size,size,No.of features)
#     SWMM_tr_sam = SWMM_tr[ss:ss+1,] # (1,2160, 2)
#     tox_tr_sam = tox_tr[ss] # (1,)


# errors = RegressionMetrics(cnn_pred, lr_rgr_pred)

def nonzeroidx(a):
    return [i for i, e in enumerate(a) if e != 0]



def FindCoefIdxfromNZ(list1, list2): # Find where coef index comes from (No. of coef>No. of nonzero vlaue in SWMM)
    list3=[]
    for i in range(len(list1)):
        if list1[i] not in list2:
            list3.append(list1[i])
        else : pass
    return list3, print('coef indx from nonzero input:', list3)


def Revert_1d_to_ndarr(Input): # Re-packing 1darray to be original shape

    if Input.ndim == 1:
        if Input.shape[0] == 72: # EFDC
            Input_re = Input.reshape(1,3,3,8)
        elif Input.shape[0] == 4320: # SWMM
            Input_re = Input.reshape(1,2160,2)
        else:
            raise ValueError('Wrong data shape')
    else:
        raise ValueError
    return Input_re


def test_reshape(Input): # Testing reshape between 1d and ndarray is completed well
    Input_1d = Input.reshape(-1, )  # making 1d and copying it (1,3,3,8) -> (72,)

    if Input.ndim == 3:  # Input SWMM(1,2160,2): Input_1d(4320,)->Input_re(1,2160,2)
        Input_re = Input_1d.reshape(Input.shape[0], Input.shape[1], Input.shape[2])
    elif Input.ndim == 4:  # Input EFDC(1,3,3,8): Input_1d(72,)->Input_re(1,3,3,8)
        Input_re = Input_1d.reshape(Input.shape[0], Input.shape[1], Input.shape[2], Input.shape[3])
    else:
        raise ValueError('Wrong input dimension')
    # ** 3) Revert to the orginal shape
    tf = (Input_re == Input)  # Compare reverted matrix w/ original one
    if np.count_nonzero(tf == 0) == 0:  # Count num of False element (0)
        print('Reshape success')
    else:
        raise ValueError('Reshape fail')
    return








# def PrintRegScore(y_true, y_pred):
    err = (y_true - y_pred)**2

    # print('mean_squared_errors: {}'.format(mean_squared_error(y_true, y_pred)))
    # print('r2_score: {}'.format(r2_score(y_true, y_pred)))
# calculate error

# # plotting
# plt.plot(lr_rgr.coef_)  # lr_rgr.coef_    # 기울기 slope
# plt.show()
#
# print('np.count_nonzero(lr_rgr.coef_)')
# print('np.count_nonzero(lr_rgr.coef_[0:72])')
# print('np.count_nonzero(SWMM_tr_sam.reshape(-1,))')
#
# # normalize
# a = lr_rgr.coef_ / np.sum(lr_rgr.coef_)
# print(a.sum()) # 1.000001
# print(a[0:72].sum()) # 0.07255049 EFDC
# print(a[72:].sum()) # 0.92745113 SWMM

