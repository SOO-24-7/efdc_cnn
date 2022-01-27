import os
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2

import matplotlib.colors as mcl
import seaborn as sns

import numpy as np
import pandas as pd

from easy_mpl import plot, imshow

from lime_utils import select_one_ex, make_prtb_1d, multiply_fac_pertb, build_cnn_model_from_config, cnn_model_predict, revert_1d_to_ndarr, test_reshape, nonzero_idx, find_coefidx_from_nonzero
from import_mat import import_mat_file

Default_PATH = os.path.join(os.getcwd())

# import inputdata
EFDC_tr, EFDC_val, SWMM_tr, SWMM_val, \
tox_tr, tox_val, \
EFDC_map, tox_map, SWMM_map = import_mat_file(Default_PATH) # EFDC_tr(900, 3, 3, 8), SWMM_tr(900, 2160, 2), tox_tr.shape (900, 1)

d = pd.read_csv('for_heatmap.csv')
d.head()

fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(d, ax=ax,
            linewidths=.5,
            cmap="Reds",
            vmin=0, vmax=1)

plt.tight_layout()
# plt.savefig(fname = "heatmap")
# plt.show()

# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
# ax1
# ax2
# sns.heatmap(d, annot=True, ax=ax1)
# plot(tox_tr, '--*', ax=ax2, show=False)
# plt.show()

## 독성결과 plotting
# idx = [13, 899, 861, 579, 383, 215, 411, 171, 440, 783]
idx = [783, 440, 171, 411, 215, 383, 579, 861, 899, 13]
np.array(idx)
tox_tr[idx]

plt2.rcParams["figure.figsize"] = (10,3)

plt2.plot(tox_tr[idx],'-*', linewidth=2)
plt.xticks(np.arange(0, 10, step=1))
plt.tight_layout()
# plt.show()
plt.savefig(fname = "tox_conc")
