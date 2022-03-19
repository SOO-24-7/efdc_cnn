import os
import site
site.addsitedir("D:\\AI4Water")

import numpy as np

from scipy import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ai4water import Model
# from ai4water.postprocessing.SeqMetrics import RegressionMetrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from easy_mpl import plot, imshow

from lime_utils import select_one_ex, make_prtb_1d, multiply_fac_pertb, build_cnn_model_from_config, cnn_model_predict, revert_1d_to_ndarr, test_reshape, nonzero_idx, find_coefidx_from_nonzero
from import_mat import import_mat_file

Default_PATH = os.path.join(os.getcwd())

# import inputdata
EFDC_tr, EFDC_val, SWMM_tr, SWMM_val, \
tox_tr, tox_val, \
EFDC_map, tox_map, SWMM_map = import_mat_file(Default_PATH) # EFDC_tr(900, 3, 3, 8), SWMM_tr(900, 2160, 2), tox_tr.shape (900, 1)

# ******** Options --> Apply HPO concept later ********
n_perturb = 500  # num of perturbation
low, high = 0.9, 1.1  # range of multi_fac
random_state= 365 # rndseed num for random sampling

# Info for CNN model build
config_f_path = r'20211130_123508_0.662\config.json'
weight_f_path = r'20211130_123508_0.662\weights\weights_4014_0.01794.hdf5'

# LIME
# ex_num = np.random.randint(0, EFDC_tr.shape[0], 1).item() # ** Select one example (based on ex_num) for perturbing
ex_num = 440
EFDC_tr_ex, SWMM_tr_ex, tox_tr_ex = select_one_ex(ex_num, EFDC_tr, SWMM_tr, tox_tr)

multi_fac = multiply_fac_pertb(low, high, n_perturb, random_state) # Generate multiplying factor for perturbation

# CNN model build
cnn_model = build_cnn_model_from_config(Default_PATH,config_f_path,weight_f_path) # load json and build model

EFDC_prtb_set,EFDC_prtb_set_1d,SWMM_prtb_set,SWMM_prtb_set_1d,inp_prtb_set_1d, TOX_prtb_set = list(),list(),list(), list(), list(), list() # Dataset of neighborhood data

for ii in np.arange(n_perturb):
    # ** Convert Ndarray to 1D array and perturb it **

    EFDC_prtb, EFDC_prtb_1d = make_prtb_1d(EFDC_tr_ex,multi_fac[ii]) # EFDC_prtb(1, 3, 3, 8), EFDC_prtb_1d(72,)
    SWMM_prtb, SWMM_prtb_1d = make_prtb_1d(SWMM_tr_ex, multi_fac[ii]) # SWMM_prtb(1, 2160, 2), SWMM_prtb_1d(4320,)
    ## ** Concat two 1d array (EFDC, SWMM) to make linear model
    inp_prtb_1d = np.concatenate([EFDC_prtb_1d, SWMM_prtb_1d]) # (4392,) = (72,) + (4320,)

    EFDC_prtb_set.append(EFDC_prtb) # list of perturbed 3d array(EFDC)
    SWMM_prtb_set.append(SWMM_prtb) # list of perturbed 2d array (SWMM)

    EFDC_prtb_set_1d.append(EFDC_prtb_1d) # list of
    # perturbed 1d array (EFDC)
    SWMM_prtb_set_1d.append(SWMM_prtb_1d) # list of perturbed 1d array (SWMM)
    inp_prtb_set_1d.append(inp_prtb_1d) # (100, 4392)

    ## 2-1) trained CNN모델에 perturbed input(new_x) --> new_y
    new_y = cnn_model_predict(cnn_model, x=[EFDC_prtb,SWMM_prtb]) # (1, 1) new_y (TOX)
    TOX_prtb_set.append(new_y)

# perturbed된 example의 dataset
inp_prtb_set_1d = np.array(inp_prtb_set_1d) # numpy array
TOX_prtb_set = np.array(TOX_prtb_set).reshape(-1,)

## 2-2) Linear regression w/ peturbed input data
lr_rgr = LinearRegression()
lr_rgr.fit(inp_prtb_set_1d, TOX_prtb_set)

## 2-3) lr_rgr과 cnn모델로 original data 예측값 비교
inp_orig_1d = np.concatenate([EFDC_tr_ex.flatten(),SWMM_tr_ex.flatten()])  # (4392,) = (72,) + (4320,)
lr_rgr_pred =  lr_rgr.predict([inp_orig_1d]) # regression 예측값
cnn_pred = cnn_model_predict(cnn_model, x=[EFDC_tr_ex, SWMM_tr_ex])  # (1, 1) new_y (TOX)
cnn_pred=np.array(cnn_pred).reshape(-1, ) # (1,)

print('Tox_true:', tox_tr_ex.item())
print('Tox_cnn:', cnn_pred.item())
print('Tox_rgr:', lr_rgr_pred.item())

# Calculate erros
print('rmse(cnn-rgr):', mean_squared_error(cnn_pred, lr_rgr_pred))
print('rmse(true-cnn):', mean_squared_error(tox_tr_ex, cnn_pred))
print('rmse(true-rgr):', mean_squared_error(tox_tr_ex, lr_rgr_pred))

# ** 3) Revert to the orginal shape
lime_EFDC= revert_1d_to_ndarr(lr_rgr.coef_[0:72])
lime_SWMM= revert_1d_to_ndarr(lr_rgr.coef_[72:])

mean_import_EFDC = list()

for ft_num in np.arange(8): # no. of efdc features
    mean_import_efdc = lime_EFDC[0, :, :, ft_num].mean()
    print("EFDC mean importance(ft-%d): %.15f" % (ft_num, mean_import_efdc))
    mean_import_EFDC.append(mean_import_efdc)

imp_swmm1_mean = np.array((lime_SWMM[0, 0:720, 0].mean(), lime_SWMM[0, 720:1440, 0].mean(), lime_SWMM[0, 1440:2160, 0].mean()))
imp_swmm2_mean = np.array((lime_SWMM[0, 0:720, 1].mean(), lime_SWMM[0, 720:1440, 1].mean(), lime_SWMM[0, 1440:2160, 1].mean()))
print("SWMM mean importance(ft1): %.15f %.15f %.15f" % (imp_swmm1_mean[0],imp_swmm1_mean[1],imp_swmm1_mean[2]))
print("SWMM mean importance(ft2): %.15f %.15f %.15f" % (imp_swmm2_mean[0],imp_swmm2_mean[1],imp_swmm2_mean[2]))
mean_import_SWMM = np.concatenate((imp_swmm1_mean,imp_swmm2_mean))

mean_import_EF_SW = np.concatenate((mean_import_EFDC,mean_import_SWMM))
imshow(mean_import_EF_SW.reshape(-1,1).T, colorbar=True, colorbar_orientation="horizontal")


# *** normalize  ***
coef_norm = lr_rgr.coef_ / np.sum(lr_rgr.coef_)
print(coef_norm.sum()) # 1.000001
print(coef_norm[0:72].sum()) # 0.07255049 EFDC
print(coef_norm[72:].sum()) # 0.92745113 SWMM

# ** 3-1) Revert to the orginal shape w/ normalize
coef_norm_efdc= revert_1d_to_ndarr(coef_norm[0:72])
coef_norm_swmm= revert_1d_to_ndarr(coef_norm[72:])

coef_norm_EFDC_mean = list()
for ft_num in np.arange(8): # no. of efdc features
    coef_norm_efdc_mean = coef_norm_efdc[0, :, :, ft_num].mean()
    print("EFDC mean importance(ft-%d): %.15f" % (ft_num, coef_norm_efdc_mean))
    coef_norm_EFDC_mean.append(coef_norm_efdc_mean)

coef_norm_swmm1_mean = np.array((coef_norm_swmm[0, 0:720, 0].mean(), coef_norm_swmm[0, 720:1440, 0].mean(), coef_norm_swmm[0, 1440:2160, 0].mean())) # swmm feature1
coef_norm_swmm2_mean = np.array((coef_norm_swmm[0, 0:720, 1].mean(), coef_norm_swmm[0, 720:1440, 1].mean(), coef_norm_swmm[0, 1440:2160, 1].mean())) # swmm feature2
print("SWMM mean importance(ft1): %.15f %.15f %.15f" % (coef_norm_swmm1_mean[0],coef_norm_swmm1_mean[1],coef_norm_swmm1_mean[2]))
print("SWMM mean importance(ft2): %.15f %.15f %.15f" % (coef_norm_swmm2_mean[0],coef_norm_swmm2_mean[1],coef_norm_swmm2_mean[2]))
coef_norm_SWMM_mean = np.concatenate((coef_norm_swmm1_mean,coef_norm_swmm2_mean))

coef_norm_mean_EF_SW = np.concatenate((coef_norm_EFDC_mean,coef_norm_SWMM_mean))
imshow(coef_norm_mean_EF_SW.reshape(-1,1).T, colorbar=True, colorbar_orientation="horizontal")
print(coef_norm_mean_EF_SW)
# find index of top N values and indices
idx =(-tox_tr.reshape(-1,)).argsort()[:10]


