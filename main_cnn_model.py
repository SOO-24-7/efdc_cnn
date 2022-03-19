import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
# from tensorflow import keras
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from scipy import io
from ai4water import Model
from cnn_layer import layers
from import_mat import import_mat_file
from ai4water.hyperopt import HyperOpt, Integer, Real, Categorical

# option for model training
epochs = 5 #3000
# batch_size = 64
# lr=0.0003
batch_size = Categorical([32, 64, 128]),
lr = Real(0.0001, 0.01,name="lr")

Default_PATH = os.path.join(os.getcwd())
## 1. import input dataset
EFDC_tr, EFDC_val, SWMM_tr, SWMM_val, \
tox_tr, tox_val, \
EFDC_map, SWMM_map, tox_map = import_mat_file(Default_PATH) # EFDC_tr(900, 3, 3, 8), SWMM_tr(900, 2160, 2), tox_tr.shape (900, 1)

## 2. Model Construction ###
model = Model(
    model={'layers': layers},
    patience=500,
    lr = lr,
    loss = 'mse', # default is mse
    metrics=['mae'], # monitor / List of metrics to monitor
)

## 2.1 Visualize cnn model architecture
# plot_model(model, to_file='model_shapes.png', show_shapes=True)

optimizer = model.optimize_hyperparameters(train_data={'x':[EFDC_tr, SWMM_tr], 'y': tox_tr},
                                           val_data={'x': [EFDC_val, SWMM_val], 'y': tox_val},
                                           algorithm = "bayes", # choose between 'random', 'grid' or 'atpe'
                                           num_iterations = 5
                                           )
#print('best parameters', optimizer.best_paras())


# 3. Model training
# print("Fit model on training data")
# hist = model.fit(x=[EFDC_tr, SWMM_tr], y=tox_tr,
#                  validation_data=([EFDC_val, SWMM_val], tox_val),
#                  batch_size=batch_size,
#                  epochs=epochs,
#                  verbose=1,
# # val_metric=["mse", "r2", "nse"]
#                   )
#
# # 4. Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x=[EFDC_val, SWMM_val], y=tox_val, batch_size=128)
# print("test loss, test acc:", results)
#
# # 5. Generate predictions (probabilities -- the output of the last layer)
# # on new data using `predict`
# predictions = model.predict(x=[EFDC_val, SWMM_val])
# print("predictions shape:", predictions.shape)


# config_path = os.path.join(os.getcwd(), "results", "20211116_단순 CNN 구성", "config.json")
# model = Model.from_config(config_path=config_path)
# model.update_weights(weights, weight_file=None) # "weights_6343_0.02867.hdf5"

# from ai4water.postprocessing.SeqMetrics import RegressionMetrics
#
# plt.close('all')
# true1,pred1  = model.predict(x=[EFDC_tr, SWMM_tr], y=tox_tr, return_true=True)
# errors = RegressionMetrics(true1, pred1)
# r2 = errors.r2()
# print(r2_score(true1, pred1))
#
# plt.close('all')
# true, pred = model.predict(x=[EFDC_val, SWMM_val], y=tox_val, return_true=True)
# errors = RegressionMetrics(true, pred)
# r2 = errors.r2()
# print(r2_score(true, pred))