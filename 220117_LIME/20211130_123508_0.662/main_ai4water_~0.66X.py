import os
import site
site.addsitedir("D:\\AI4Water")

import mat73
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import io
from ai4water import Model
from sklearn.metrics import r2_score
import wandb

DATA_PATH1 = os.path.join(os.getcwd())

parm_total_tr = io.loadmat(os.path.join(DATA_PATH1, 'soo_in_tr_new.mat'))['soo_in_tr']  #input1(EFDC)
parm_total_val = io.loadmat(os.path.join(DATA_PATH1, 'soo_in_vl_new.mat'))['soo_in_vl']

wq_tr = io.loadmat(os.path.join(DATA_PATH1, 'soo_out_tr_nlg.mat'))['soo_out_tr_nlg']   #output
wq_val = io.loadmat(os.path.join(DATA_PATH1, 'soo_out_vl_nlg.mat'))['soo_out_vl_nlg']

map_tot = io.loadmat(os.path.join(DATA_PATH1, 'input_tot.mat'))['input_tot']  #전체값
vmap_tot= io.loadmat(os.path.join(DATA_PATH1, 'output_tot.mat'))['output_tot']

# SWMM TS data
SWMM_TS_tr = io.loadmat(os.path.join(DATA_PATH1, 'SWMM_TS_tr.mat'))['SWMM_TS_tr']  #Input2(SWMM)_tr
SWMM_TS_vl = io.loadmat(os.path.join(DATA_PATH1, 'SWMM_TS_vl.mat'))['SWMM_TS_vl']  #input2_val
SWMM_TS_tot = mat73.loadmat(os.path.join(DATA_PATH1, 'SWMM_TS_tot.mat'))['SWMM_TS_tot']  #input2 total

# Make numpy array
parm_total_tr = np.array(parm_total_tr,dtype='float32') # float32 bit로 dtype 지정 = 뒤에 모델 weight type과 일치시키기 위하여
parm_total_val = np.array(parm_total_val,dtype='float32')

wq_tr = np.array(wq_tr,dtype='float32')
wq_val = np.array(wq_val,dtype='float32')

map_tot = np.array(map_tot,dtype='float32')
vmap_tot = np.array(vmap_tot,dtype='float32')

SWMM_TS_tr = np.array(SWMM_TS_tr,dtype='float32')
SWMM_TS_vl = np.array(SWMM_TS_vl,dtype='float32')
SWMM_TS_tot = np.array(SWMM_TS_tot,dtype='float32')


from ai4water.hyperopt import Integer, Real, Categorical

model = Model(
    model = {"layers":
                 {
                     "Input": {"shape": (3,3,8), "name": "EFDC_inputs"},
                     "Conv2D": {"filters": 15, "kernel_size": [2,2], "strides": (1,1), "padding": "same"},
                     "BatchNormalization": {"epsilon": 0.001, "center": True, "scale": False, "momentum": 0.9},
                     "crelu": {},
                     "Conv2D_2": {"filters": 17, "kernel_size": [2,2], "padding": "same"},
                     "BatchNormalization_2": {"epsilon": 0.001, "center": True, "scale": False, "momentum": 0.9},
                     "crelu_2": {},
                     "Conv2D_3": {"filters": 17, "kernel_size": [2,2], "padding": "same"},
                     "BatchNormalization_3": {"epsilon": 0.001, "center": True, "scale": False, "momentum": 0.9},
                     "crelu_3": {},
                     "Conv2D_4": {"filters": 17, "kernel_size": [2,2], "padding": "same"},
                     "BatchNormalization_4": {"epsilon": 0.001, "center": True, "scale": False, "momentum": 0.9},
                     "crelu_4": {},
                     "Dropout": 0.4,
                     "Flatten": {},
                     "Dense": {"units": 16, "activation": "crelu", "name": "efdc_outputs"},

                     "Input_1": {"shape": (2160, 2), "name": "SWMM_inputs"},
                     "Flatten_2": {"config": {},
                                  "inputs": "SWMM_inputs"},
#                    "Dropout": 0.5
                     "Dense_1": {"config": {"units": 17, "activation": "crelu"},
                                 "inputs": "Flatten_2"},
                     "Dense_2": {"config": {"units": 16, "activation": "crelu"},
                                 "inputs": "Dense_1"},
                     "Dense_3": {"config": {"units": 16, "activation": "crelu"},
                                 "inputs": "Dense_2"}, # add dense for test
#                    "Flatten_3": {"config": {},
#                                   "inputs": "Dense_2"},
                     "Dense_4": {"config": {"units": 16, "activation": "crelu", "name": "swmm_outputs"},
                               "inputs": "Dense_3"},

                     "Concatenate": {"config": {"axis": 1},
                                     "inputs": ["efdc_outputs", "swmm_outputs"]},

                     "Dense_5": {"config": {"units": 16, "activation": "crelu"},
                                 "inputs": "Concatenate"},
                     #"Dropout_1": 0.5,
                     "Dense_6": {"config": {"units": 16, "activation": "crelu"},
                                 "inputs": "Dense_5"},
                     "Dense_7": {"config": {"units": 16, "activation": "crelu"},
                                 "inputs": "Dense_6"},
                     "Dropout_2": 0.3,
                     "Dense_8": {"config": 1,"name": "model_output"}

                 }
    },
    lookback=1,
    batch_size=Categorical([32, 64, 128]),
    lr=Real(0.0001, 0.01, num_samples=10), # 10->5 of number samples for test
    patience=20000,
    epochs=20000,
    wandb_config = {'training_data': True, 'validation_data': True, 'project':"my-test-project", 'entity':"hyein"}
)


# optimizer = model.optimize_hyperparameters("bayes", num_iterations=10,
#                                            train_data={'x':[parm_total_tr, SWMM_TS_tr], 'y': wq_tr},
#                                            val_data={'x': [parm_total_val, SWMM_TS_vl], 'y': wq_val})

#print('best parameters', optimizer.best_paras())


hist = model.fit(x=[parm_total_tr, SWMM_TS_tr], y=wq_tr,
                  validation_data=([parm_total_val, SWMM_TS_vl], wq_val),
                  epochs=20000,
# val_metric=["mse", "r2", "nse"]
                  )

# config_path = os.path.join(os.getcwd(), "results", "20211116_단순 CNN 구성", "config.json")
# model = Model.from_config(config_path=config_path)
# model.update_weights(weights, weight_file=None) # "weights_6343_0.02867.hdf5"

from ai4water.postprocessing.SeqMetrics import RegressionMetrics

plt.close('all')
true1,pred1  = model.predict(x=[parm_total_tr,SWMM_TS_tr], y=wq_tr, return_true=True)
errors = RegressionMetrics(true1, pred1)
r2 = errors.r2()
print(r2_score(true1, pred1))

plt.close('all')
true, pred = model.predict(x=[parm_total_val, SWMM_TS_vl], y=wq_val, return_true=True)
errors = RegressionMetrics(true, pred)
r2 = errors.r2()
print(r2_score(true, pred))

#wandb.init(project="my-test-project", entity="hyein")
#config = wandb.config
#wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }
#wandb.log({"loss": loss})

# # Optional
#wandb.watch(model)
