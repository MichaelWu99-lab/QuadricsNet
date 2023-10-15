import numpy as np
import os
import pandas as pd
import csv

resolution = 1*1e-2

dataset = "logs/results"
experiment_alias = '11_train_e2e_mode_0_normalsInputFi-Fe_0100-1-lamb0-1_0.2_0.2_0.2_0.2_0.2-0.2-0.2-0.1_cluIter_5_bt_12_lr_0.0001_knn_80_knnStep_2_more_run_pca_ABC'
# 0-index
num_object = 999

thres = 0.01

dist_all = []
charmfer_distance_all = []
p_cov_1_all = []
p_cov_2_all = []
s_cov_1_all = []
s_cov_2_all = []
cov_1_all = []
cov_2_all = []

for n in range(num_object):
    continue_signal = 0
    print("processing: {}/{}", n, num_object)
    Points_clustered_scaled_dir = dataset+'/'+experiment_alias+'/clustered_points/'+'object_pre_'+str(n)+'/'
    Points_clustered_scaled_input_dir = dataset+'/'+experiment_alias+'/clustered_points/'+'object_pre_input_'+str(n)+'/'

    Points_labeled_raw_gt_dir = dataset+'/'+experiment_alias+'/clustered_points/'+'object_gt_raw_'+str(n)+'/'

    if not os.path.exists(dataset+'/'+experiment_alias+'/points_raw'+'/object_'+str(n)+'.txt'):
        print("\n")
        continue_signal = 1
        continue
    # Points_raw = pd.read_table(dataset+'/'+experiment_alias+'/points_raw'+'/object_'+str(n)+'.txt',header=None).to_numpy()
    Points_raw = np.loadtxt(dataset+'/'+experiment_alias+'/points_raw'+'/object_'+str(n)+'.txt',delimiter=',')

    # 导入quadrics coefficients
    n = n + 1
    n = 2*n-1
    rows_0 = pd.read_csv(dataset+'/'+experiment_alias+'/quadrics.csv',header=None).values.tolist()

    shape_type_object_gt = [int(i[1:-1]) for i in rows_0[n][12][1:-1].split("\n ")]
    shape_type_object_pre = [int(i[1:-1]) for i in rows_0[n][13][1:-1].split("\n ")]
    shape_points_num = [int(i[1:-1]) for i in rows_0[n][19][1:-1].split("\n ")]
    shape_type_name_list = ["sphere","plane","cylinder","cone"]

    # 归一化后的quadrics
    q_object_0 = float(rows_0[n][15]) # gt
    q_object_1 = float(rows_0[n+1][15])
    
    # # 反归一化后的quadrics，原始的
    # q_object_0_raw = str2num(rows_0(n,15)) # gt
    # q_object_1_raw = str2num(rows_0(n+1,15))
    
    # T_object = reshape(str2num(rows_0(n,17)),[4,4])'
    # T_sample_object = str2num(rows_0(n,18))
    
    # dist_object = [];
    # points_object_pre = [];
    # points_object_gt = [];