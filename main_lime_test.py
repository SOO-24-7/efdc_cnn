import os
import site
site.addsitedir("D:\\AI4Water")

import numpy as np
from scipy import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ai4water import Model
# from ai4water.postprocessing.SeqMetrics import RegressionMetrics
import matplotlib.pyplot as plt
from lime_utils import SlctOneExample,MultiFac4Pertb,Revert_1d_to_ndarr,test_reshape,nonzeroidx,FindCoefIdxfromNZ
from import_mat import ImportMatfile

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

for ex_num in np.arange(EFDC_tr.shape[0]): # ex_num: example number within data (0~899)
    # ** Select one example (based on ex_num) for perturbing
    EFDC_tr_ex, SWMM_tr_ex, tox_tr_ex = SlctOneExample(ex_num, EFDC_tr, SWMM_tr, tox_tr)

    # Step: Perturbing one example
    multi_fac = MultiFac4Pertb(low, high, n_perturb, random_state) # Generate multiplying factor for perturbation

    # CNN model build
    cnn_model = build_cnn_model_from_config(Default_PATH,config_f_path,weight_f_path) # load json and build model

    EFDC_prtb_set,EFDC_prtb_set_1d,SWMM_prtb_set,SWMM_prtb_set_1d,inp_prtb_set_1d, TOX_prtb_set = list(),list(),list(), list(), list(), list() # Dataset of neighborhood data

    for ii in np.arange(n_perturb):
        # ** Convert Ndarray to 1D array and perturb it **

        EFDC_prtb, EFDC_prtb_1d = Make_Prtb_1d(EFDC_tr_ex,multi_fac[ii]) # EFDC_prtb(1, 3, 3, 8), EFDC_prtb_1d(72,)
        SWMM_prtb, SWMM_prtb_1d = Make_Prtb_1d(SWMM_tr_ex, multi_fac[ii]) # SWMM_prtb(1, 2160, 2), SWMM_prtb_1d(4320,)
        ## ** Concat two 1d array (EFDC, SWMM) to make linear model
        inp_prtb_1d = np.concatenate([EFDC_prtb_1d, SWMM_prtb_1d]) # (4392,) = (72,) + (4320,)

        EFDC_prtb_set.append(EFDC_prtb) # list of perturbed 3d array(EFDC)
        SWMM_prtb_set.append(SWMM_prtb) # list of perturbed 2d array (SWMM)

        EFDC_prtb_set_1d.append(EFDC_prtb_1d) # list of perturbed 1d array (EFDC)
        SWMM_prtb_set_1d.append(SWMM_prtb_1d) # list of perturbed 1d array (SWMM)
        inp_prtb_set_1d.append(inp_prtb_1d) # (100, 4392)

        ## 1) trained CNN모델에 perturbed input(new_x) --> new_y
        new_y = cnn_model_predict(cnn_model, x=[EFDC_prtb,SWMM_prtb]) # (1, 1) new_y (TOX)
        TOX_prtb_set.append(new_y)

    # perturbed된 example의 dataset
    inp_prtb_set_1d = np.array(inp_prtb_set_1d) # numpy array
    TOX_prtb_set = np.array(TOX_prtb_set).reshape(-1,)

    ## 2) Linear regression w/ peturbed input data
    lr_rgr = LinearRegression()
    lr_rgr.fit(inp_prtb_set_1d, TOX_prtb_set)

    ## 3) lr_rgr과 cnn모델로 original data 예측값 비교
    inp_orig_1d = np.concatenate([EFDC_tr_ex.flatten(),SWMM_tr_ex.flatten()])  # (4392,) = (72,) + (4320,)
    lr_rgr_pred =  lr_rgr.predict([inp_orig_1d]) # regression 예측값
    cnn_pred = cnn_model_predict(cnn_model, x=[EFDC_tr_ex, SWMM_tr_ex])  # (1, 1) new_y (TOX)
    cnn_pred=np.array(cnn_pred).reshape(-1, ) # (1,)

    print('Tox_true:', tox_tr_ex.item())
    print('Tox_cnn:', cnn_pred.item())
    print('Tox_rgr:', lr_rgr_pred.item())

    # integer -> float OK
    # float -> integer not safe
    # Calculate erros
    print('rmse(cnn-rgr):', mean_squared_error(cnn_pred, lr_rgr_pred))
    print('rmse(true-cnn):', mean_squared_error(tox_tr_ex, cnn_pred))
    print('rmse(true-rgr):', mean_squared_error(tox_tr_ex, lr_rgr_pred))

    print('num_NZ_coef:', np.count_nonzero(lr_rgr.coef_))
    print('num_NZ_coef_EFDC:', np.count_nonzero(lr_rgr.coef_[0:72]))
    print('num_NZ_coef_SWMM:', np.count_nonzero(lr_rgr.coef_[72:]))

    print('num_NZ_inp_EFDC:', np.count_nonzero(EFDC_tr_ex.flatten()))
    print('num_NZ_inp_SWMM:', np.count_nonzero(SWMM_tr_ex.flatten()))

    NZIdx_coef = nonzeroidx(lr_rgr.coef_) # Find nonzero value idx
    NZIdx_coef_EFDC =nonzeroidx(lr_rgr.coef_[0:72]) # Find nonzero value idx
    NZIdx_coef_SWMM =nonzeroidx(lr_rgr.coef_[72:])
    NZIdx_inp_EFDC = nonzeroidx(EFDC_tr_ex.reshape(-1,)) # Find nonzero value idx
    NZIdx_inp_SWMM = nonzeroidx(SWMM_tr_ex.reshape(-1,)) # Find nonzero value idx

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
