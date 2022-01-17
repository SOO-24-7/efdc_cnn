clear all; close all; clc;

load soo_plz_tr_p.mat
load soo_plz_vl_p.mat
load soo_plz_map_p.mat
% load soo_out.mat
load soo_out_tr_nlg.mat
load soo_out_vl_nlg.mat
load output_tot
% load soo_out_tr_lg.mat
% load soo_out_vl_lg.mat
load val_RM_p.mat

output_tot=((10.^output_tot)-1)/1000;
%%
[tr_nse, tr_r2, tr_pb, tr_rmse]=stat(result_tr,soo_out_tr_nlg);
[vl_nse, vl_r2, vl_pb, vl_rmse]=stat(result_val,soo_out_vl_nlg);

Accuray_tr = [tr_nse tr_r2 tr_rmse]
Accuray_vl = [vl_nse vl_r2 vl_rmse]


figure(1)
scatter(soo_out_tr_nlg, result_tr);
title('training')

figure(2)
scatter(soo_out_vl_nlg, result_val);
title('validation')

figure(3)
plot(val_RM)

figure(4)
scatter(output_tot, result_map);
title('validation')

%% 
% [tr_nse, tr_r2, tr_pb, tr_rmse]=stat(result_tr,soo_out_tr_lg);
% [vl_nse, vl_r2, vl_pb, vl_rmse]=stat(result_val,soo_out_vl_lg);
% 
% Accuray_tr = [tr_nse tr_r2 tr_rmse]
% Accuray_vl = [vl_nse vl_r2 vl_rmse]
% 
% 
% figure(1)
% scatter(soo_out_tr_lg, result_tr);
% title('training')
% 
% figure(2)
% scatter(soo_out_vl_lg, result_val);
% title('validation')
% 
% figure(3)
% plot(val_RM)

%% Log Normalization
% load output_tn_hh_wohyp_l.mat
% load output_vn_hh_wohyp_l.mat
% 
% [tr_nse, tr_r2, tr_pb, tr_rmse]=stat(result_tr,obs_tr_l);
% [vl_nse, vl_r2, vl_pb, vl_rmse]=stat(result_val,obs_val_l);
% 
% Accuray_tr = [tr_nse tr_r2 tr_rmse]
% Accuray_vl = [vl_nse vl_r2 vl_rmse]
% 
% 
% figure(1)
% scatter(obs_tr_l, result_tr);
% title('training')
% 
% figure(2)
% scatter(obs_val_l, result_val);
% title('validation')
% 
% figure(3)
% plot(val_RM)