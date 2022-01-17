import os
import site
site.addsitedir("D:\\AI4Water")

import mat73
import numpy as np
from scipy import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from ai4water import Model
# from ai4water.postprocessing.SeqMetrics import RegressionMetrics
from numpy.random import default_rng
import matplotlib.pyplot as plt
from lime_utils import Make1dArrandRevert,nonzeroidx,FindCoefIdxfromNZ

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

# ** Extract one sample
ex_num=100 # example number
EFDC_tr_sam = np.expand_dims(EFDC_tr[ex_num], axis=0) # (1, 3, 3, 8) = (No. of sampel, size,size,No.of features)
SWMM_tr_sam = np.expand_dims(SWMM_tr[ex_num], axis=0) # (1,2160, 2)
tox_tr_sam = tox_tr[ex_num] # (1,)

# ** Perturb one sample
n_perturb = 5000 # num of perturbation

# Generate multiplying factor for perturbation (making neighboringhood instances)
def multi_fac(low, high, size, random_state=365):
    rng = default_rng(random_state)  # default_rng: the recommended constructor for the random number class Generator .
    return rng.uniform(low, high, size=size)

low_,high_ = 0.9, 1.1 # range of multi_fac
multi_fac_ = multi_fac(low_, high_, n_perturb)  # rng.uniform(low=0.9 high=1.1, size=100)


config_f_path = '20211130_123508_0.662\\config.json'
weight_f_path ='D:\\UNIST_SBKim\\B_paper\\3_[JHM]EFDC_CNN (tox)\\CNN_SB\\220113_lime\\20211130_123508_0.662\weights\\weights_4014_0.01794.hdf5'

# load json and build model
def build_cnn_model_from_config():
    _model = Model.from_config_file(config_path=os.path.join(DATA_PATH1, config_f_path))
    _model.update_weights(weight_f_path)
    return _model

cnn_model = build_cnn_model_from_config()

def CNNModel(_model, x):
    return _model.predict(x=x)

EFDC_prtb_set,EFDC_prtb_set_1d,SWMM_prtb_set,SWMM_prtb_set_1d,inp_prtb_set_1d, TOX_prtb_set = list(),list(),list(), list(), list(), list() # Dataset of neighborhood data

for ii in np.arange(n_perturb):
    # ** Convert Ndarray to 1D array and perturb it **
    EFDC_prtb = EFDC_tr_sam * multi_fac_[ii]  # (1, 3, 3, 8): perturbed input1(EFDC) # new_x1
    EFDC_prtb_1d = EFDC_prtb.reshape(-1,)  # (72,) : make 1d array

    EFDC_prtb_set.append(EFDC_prtb) # list of perturbed 3d array(EFDC)
    EFDC_prtb_set_1d.append(EFDC_prtb_1d) # list of perturbed 1d array (EFDC)

    SWMM_prtb = SWMM_tr_sam * multi_fac_[ii] # (1, 2160, 2): perturbed input2(SWMM) # new_x2
    SWMM_prtb_1d = SWMM_prtb.reshape(-1,)  # (4320,) : make 1d array

    SWMM_prtb_set.append(SWMM_prtb) # list of perturbed 1d array (SWMM)
    SWMM_prtb_set_1d.append(SWMM_prtb_1d) # list of perturbed 1d array (SWMM)

    ## ** Concat two 1d array (EFDC, SWMM) to make linear model
    inp_prtb_1d = np.concatenate([EFDC_prtb_1d, SWMM_prtb_1d]) # (4392,) = (72,) + (4320,)
    inp_prtb_set_1d.append(inp_prtb_1d) # (100, 4392)

    ## 1) trained CNN모델에 perturbed input(new_x) --> new_y
    new_y = CNNModel(cnn_model, x=[EFDC_prtb,SWMM_prtb]) # (1, 1) new_y (TOX)
    TOX_prtb_set.append(new_y)

# perturbed된 example의 dataset
inp_prtb_set_1d = np.array(inp_prtb_set_1d) # numpy array
TOX_prtb_set = np.array(TOX_prtb_set).reshape(-1,)

## 2) Linear regression w/ peturbed input data
lr_rgr = LinearRegression()
lr_rgr.fit(inp_prtb_set_1d, TOX_prtb_set)

## 3) lr_rgr과 cnn모델로 original data 예측값 비교
inp_orig_1d = np.concatenate([EFDC_tr_sam.flatten(),SWMM_tr_sam.flatten()])  # (4392,) = (72,) + (4320,)
lr_rgr_pred =  lr_rgr.predict([inp_orig_1d]) # regression 예측값
cnn_pred = CNNModel(cnn_model, x=[EFDC_tr_sam, SWMM_tr_sam])  # (1, 1) new_y (TOX)
cnn_pred=np.array(cnn_pred).reshape(-1, ) # (1,)

print('Tox_true:', tox_tr_sam.item())
print('Tox_cnn:', cnn_pred.item())
print('Tox_rgr:', lr_rgr_pred.item())

# integer -> float OK
# float -> integer not safe
# Calculate erros
print('rmse(cnn-rgr):', mean_squared_error(cnn_pred, lr_rgr_pred))
print('rmse(true-cnn):', mean_squared_error(tox_tr_sam, cnn_pred))
print('rmse(true-rgr):', mean_squared_error(tox_tr_sam, lr_rgr_pred))

print('num_NZ_coef:', np.count_nonzero(lr_rgr.coef_))
print('num_NZ_coef_EFDC:', np.count_nonzero(lr_rgr.coef_[0:72]))
print('num_NZ_coef_SWMM:', np.count_nonzero(lr_rgr.coef_[72:]))

print('num_NZ_inp_EFDC:', np.count_nonzero(EFDC_tr_sam.flatten()))
print('num_NZ_inp_SWMM:', np.count_nonzero(SWMM_tr_sam.flatten()))

NZIdx_coef = nonzeroidx(lr_rgr.coef_) # Find nonzero value idx
NZIdx_coef_EFDC =nonzeroidx(lr_rgr.coef_[0:72]) # Find nonzero value idx
NZIdx_coef_SWMM =nonzeroidx(lr_rgr.coef_[72:])
NZIdx_inp_EFDC = nonzeroidx(EFDC_tr_sam.reshape(-1,)) # Find nonzero value idx
NZIdx_inp_SWMM = nonzeroidx(SWMM_tr_sam.reshape(-1,)) # Find nonzero value idx

# Find where coef index comes from (No. of coef>No. of nonzero vlaue in SWMM)
# list3 = FindCoefIdxfromNZ(NZIdx_coef_SWMM,NZIdx_inp_SWMM)
# new_list3 = [x+72 for x in list3]

# ** 3) Revert to the orginal shape
# NZIdx_coef_EFDC= Make1dArrandRevert(NZIdx_coef_EFDC)
# NZIdx_coef_SWMM= Make1dArrandRevert(NZIdx_coef_SWMM)

# normalize
a = lr_rgr.coef_ / np.sum(lr_rgr.coef_)
print(a.sum()) # 1.000001
print(a[0:72].sum()) # 0.07255049 EFDC
print(a[72:].sum()) # 0.92745113 SWMM


# plotting
plt.plot(lr_rgr.coef_)
plt.xlabel('Input features')
plt.ylabel('Coefficient of linear regression')
plt.show()

plt.plot(lr_rgr.coef_[0:280])
plt.xlabel('Input features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

# np.savetxt('D:\\UNIST_SBKim\\B_paper\\3_[JHM]EFDC_CNN (tox)\\CNN_SB\\220116_LIME\\perturbation\\5000\\multi_fac100.txt',multi_fac_)
