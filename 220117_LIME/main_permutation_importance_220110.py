## 22.01.08-22.01.10
# LIME 방법 적용하려고 코드 짰는데,, 내가 짠 방법론은 permutation importance와 가까웠음...
# (하나의 feature만 perturb하고, 나머지 feature는 고정)
# --> 그래서 수정하기로 모든 feature 다 perturb

import os
import site
site.addsitedir("D:\\AI4Water")

import mat73
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy import io
from ai4water import Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from numpy.random import default_rng, gamma # gamma function

DATA_PATH1 = os.path.join(os.getcwd())

EFDC_tr = io.loadmat(os.path.join(DATA_PATH1, 'soo_in_tr_new.mat'))['soo_in_tr']  # (900, 3, 3, 8) # input1(EFDC)
EFDC_val = io.loadmat(os.path.join(DATA_PATH1, 'soo_in_vl_new.mat'))['soo_in_vl']

tox_tr = io.loadmat(os.path.join(DATA_PATH1, 'soo_out_tr_nlg.mat'))['soo_out_tr_nlg']   #output
tox_val = io.loadmat(os.path.join(DATA_PATH1, 'soo_out_vl_nlg.mat'))['soo_out_vl_nlg']

EFDC_map = io.loadmat(os.path.join(DATA_PATH1, 'input_tot.mat'))['input_tot']  #전체값
tox_map= io.loadmat(os.path.join(DATA_PATH1, 'output_tot.mat'))['output_tot']

# SWMM TS data
SWMM_tr = io.loadmat(os.path.join(DATA_PATH1, 'SWMM_TS_tr.mat'))['SWMM_TS_tr']  # (900, 2160, 2) input2(SWMM)_tr
SWMM_val = io.loadmat(os.path.join(DATA_PATH1, 'SWMM_TS_vl.mat'))['SWMM_TS_vl']  #input2_val
SWMM_TS_tot = mat73.loadmat(os.path.join(DATA_PATH1, 'SWMM_TS_tot.mat'))['SWMM_TS_tot']  #input2 total

# Make numpy array
EFDC_tr = np.array(EFDC_tr,dtype='float32') # float32 bit로 dtype 지정 = 뒤에 모델 weight type과 일치시키기 위하여
EFDC_val = np.array(EFDC_val,dtype='float32')

tox_tr = np.array(tox_tr,dtype='float32')
tox_val = np.array(tox_val,dtype='float32')

EFDC_map = np.array(EFDC_map,dtype='float32')
tox_map = np.array(tox_map,dtype='float32')

SWMM_tr = np.array(SWMM_tr,dtype='float32')
SWMM_val = np.array(SWMM_val,dtype='float32')
SWMM_TS_tot = np.array(SWMM_TS_tot,dtype='float32')

# EFDC_tr.shape    # (900, 3, 3, 8)
# SWMM_tr.shape # (900, 2160, 2)
# tox_tr.shape # (900, 1)

# ** Extract one sample # 우선 하나 정함..--> 나중에 for loop추가해야
EFDC_tr_samp = EFDC_tr[0] # (3, 3, 8) = (size,size,No.of features)
SWMM_tr_samp = SWMM_tr[0] # (2160, 2)
tox_tr_samp = tox_tr[0] # (1,)

# ** Perturb one sample (making neighboringhood instances )
num_perturb = 100

# Perturb one target feature in EFDC
n_feat = EFDC_tr.shape[-1] # 8 : num of features
_n_feat = np.arange(n_feat)

# while fixing SWMM data
SWMM_tr_samp_1d = SWMM_tr_samp.flatten() # (4320,)
SWMM_1d.append(SWMM_tr_samp_1d)

for idx_ft in _n_feat:
    idx_ft = 0 # 우선 하나 정함..
    idx_non_ft = _n_feat[_n_feat != idx_ft] # non-target feature index
    # print(f'idx_ft={idx_ft}')
    # print(f'idx_non_ft={idx_non_ft}')
    trgt_feat = EFDC_tr_samp[:, :, [idx_ft]]  # target feature for perturbing
    non_trgt_feat = EFDC_tr_samp[:, :, idx_non_ft] # fixed features EXCEPT perturbed one

    # Find the indices of nonzero and zero values
    idx_nonz = np.nonzero(trgt_feat) # or np.transpose(np.nonzero(feature))
    idx_zero = np.nonzero(trgt_feat==0)  # the indices of zero value

    # Extract value w/o 0
    raw_mean = np.mean(trgt_feat[idx_nonz]) # mean(non-zero values)
    raw_std = np.std(trgt_feat[idx_nonz]) # std(non-zero values)

    # ** Generate random error w/ distirbution **
    rng = default_rng(12345) # default_rng: the recommended constructor for the random number class Generator .

    # 1. Uniform Distribution
    s_uni = np.random.uniform(0,1,num_perturb)  # random.uniform(low=0.0, high=1.0, size=None)

    data_prtrb, EFDC_1d, inp_concat_1d = list(), list(), list()
    for i in np.arange(num_perturb):
        feat_tmp = np.zeros(trgt_feat.shape)  # zero matrix
        feat_tmp[idx_nonz] = trgt_feat[idx_nonz] + (s_uni[i] * raw_std + raw_mean) * 0.1 # Ref: 10 %, 50%, 100% Pyo(2020)
        EFDC_tr_samp[:, :, [idx_ft]] = feat_tmp # target feature value replaced by perturbed value
        data_prtrb.append(feat_tmp) # dataset of perturbed target feature

        # ** Convert Ndarray to 1D array
        EFDC_tr_samp_1d = EFDC_tr_samp.flatten()  # (72,)
        EFDC_1d.append(EFDC_tr_samp_1d) # perturb된 feature를 포함한 EFDC input --> 1d array 결과

        # ** Concat two 1d array (EFDC, SWMM)
        inp_cc_1d = np.concatenate([EFDC_tr_samp_1d,SWMM_tr_samp_1d]) # (4392,) = (72,) + (4320,)
        inp_concat_1d.append(inp_cc_1d)

    print(f'Purturbing:{num_perturb == len(inp_concat_1d)}')
    print(f'Concat inputs:{inp_concat_1d[-1].shape[0] == 4392}')


# ** Linear regression with from scratch **
# x_offset = np.average(x, axis=0) # centering
# x_ = x - x_offset
# y_offset = np.average(y)    # centering
# y_ = y - y_offset
#
# coeff, res, rank, s = np.linalg.lstsq(x_,y_) # training
# intercept = y_offset - np.dot(x_offset, coeff.T)
#
# pred = np.matmul(x, coeff.T) + intercept # prediction




