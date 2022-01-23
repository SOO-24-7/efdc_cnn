import os
import site
site.addsitedir("D:\\AI4Water")

import numpy as np

from lime_utils import build_cnn_model_from_config
from import_mat import import_mat_file

from ai4water.postprocessing.explain import ShapExplainer
from easy_mpl import plot

default_path = os.getcwd()
config_f_path = r'20211130_123508_0.662\config.json'
weight_f_path = r'20211130_123508_0.662\weights\weights_4014_0.01794.hdf5'

model = build_cnn_model_from_config(
    default_path,
    config_f_path,
    weight_f_path
)

# import inputdata
EFDC_tr, EFDC_val, SWMM_tr, SWMM_val, \
tox_tr, tox_val, \
EFDC_map, tox_map, SWMM_map = import_mat_file(default_path) # EFDC_tr(900, 3, 3, 8), SWMM_tr(900, 2160, 2), tox_tr.shape (900, 1)

train_x = [EFDC_tr, SWMM_tr]
train_y_pred = model.predict(x= train_x)
val_x = [EFDC_val, SWMM_val]
val_y_pred = model.predict(x= val_x)

expl = ShapExplainer(model, val_x, train_x, explainer="DeepExplainer") # GradientExplainer, PermutationExplainer
sv = expl.shap_values

sv_efdc, sv_swmm = sv[0] # sv_efdc(300, 3, 3, 8), sv_efdc(300, 2160, 2)

total_sv = np.concatenate((sv_efdc.reshape(-1,), sv_swmm.reshape(-1,)))

idx = np.argmax(val_y_pred).item() # 171

base_val = train_y_pred.mean() # expected value
text = f"""local prediction: {val_y_pred[idx].item()}, base_value: {base_val} 
efdc: {sv_efdc[idx].sum()}, swmm:  {sv_swmm[idx].sum()}
total {base_val + sv_efdc[idx].sum() + sv_swmm[idx].sum()}
"""
print(text)