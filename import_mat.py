import os
import mat73
import numpy as np
from scipy import io

def import_mat_file(Default_PATH):
    f_path = 'Data'
    DATA_PATH1 = os.path.join(Default_PATH, f_path)
    # input1 (EFDC)
    EFDC_tr = io.loadmat(os.path.join(DATA_PATH1, 'soo_in_tr_new.mat'))['soo_in_tr']  # (900, 3, 3, 8) # input1(EFDC)
    EFDC_val = io.loadmat(os.path.join(DATA_PATH1, 'soo_in_vl_new.mat'))['soo_in_vl']
    # input2 (SWMM)
    SWMM_tr = io.loadmat(os.path.join(DATA_PATH1, 'SWMM_TS_tr.mat'))['SWMM_TS_tr']  # (900, 2160, 2) input2(SWMM)_tr
    SWMM_val = io.loadmat(os.path.join(DATA_PATH1, 'SWMM_TS_vl.mat'))['SWMM_TS_vl']  #input2_val
    # output (EFDC tox)
    tox_tr = io.loadmat(os.path.join(DATA_PATH1, 'soo_out_tr_nlg.mat'))['soo_out_tr_nlg']   #output
    tox_val = io.loadmat(os.path.join(DATA_PATH1, 'soo_out_vl_nlg.mat'))['soo_out_vl_nlg']

    # data for mapping
    EFDC_map = io.loadmat(os.path.join(DATA_PATH1, 'input_tot.mat'))['input_tot']  #전체값
    SWMM_map = mat73.loadmat(os.path.join(DATA_PATH1, 'SWMM_TS_tot.mat'))['SWMM_TS_tot']  #input2 total
    tox_map= io.loadmat(os.path.join(DATA_PATH1, 'output_tot.mat'))['output_tot']

    # Make numpy array
    EFDC_tr = np.array(EFDC_tr,dtype='float32') # float32 bit로 dtype 지정 = 뒤에 모델 weight type과 일치시키기 위하여
    EFDC_val = np.array(EFDC_val,dtype='float32')

    SWMM_tr = np.array(SWMM_tr, dtype='float32')
    SWMM_val = np.array(SWMM_val, dtype='float32')

    tox_tr = np.array(tox_tr,dtype='float32')
    tox_val = np.array(tox_val,dtype='float32')

    EFDC_map = np.array(EFDC_map,dtype='float32')
    tox_map = np.array(tox_map,dtype='float32')
    SWMM_map = np.array(SWMM_map, dtype='float32')

    return EFDC_tr, EFDC_val, SWMM_tr, SWMM_val, \
           tox_tr, tox_val, \
           EFDC_map, tox_map, SWMM_map