from codecs import utf_16_le_decode
import os
from re import T
import sys
from turtle import shape
import h5py
import shutil
import numpy as np
import open3d as o3d


from src.read_config_e2e import Config
from src.net_dection import PrimitivesEmbeddingDGCNGn
from src.dataset_segments import generator_iter
from src.dataset_objects import Dataset
from src.residual_utils import Evaluation
from src.loss import (
    EmbeddingLoss,
    primitive_loss,
)
from src.visualization.utils_vis import utils_vis
from torch.utils.data import DataLoader
import torch
from src.utils import rescale_input_outputs_quadrics_e2e
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(precision=3,linewidth=200)  # 设置输出样式

config = Config(sys.argv[1])

model_name = config.model_path.format(
    config.mode,
    "".join(list(map(str, config.if_fitting_normals))),
    int(config.if_detection_normals),
    config.lamb_0_0,
    config.lamb_0_1,
    config.lamb_0_2,
    config.lamb_0_3,
    config.lamb_0_4,
    config.lamb_0_5,
    config.lamb_0_6,
    config.lamb_1,
    config.cluster_iterations,
    config.batch_size,
    config.lr,
    config.knn,
    config.knn_step,
    config.more
)

# set batch size to 1 for testing
config.batch_size = 1

print("Model name: ", model_name)
Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)

model = PrimitivesEmbeddingDGCNGn(
    embedding=True,
    emb_size=128,
    primitives=True,
    num_primitives=config.num_primitives,
    loss_function=Loss.triplet_loss,
    mode=config.mode,
    if_normals=config.if_detection_normals,
    knn=config.knn,
    knn_step=config.knn_step
)

# single GPU
# quadrics_detection模型
model.load_state_dict(torch.load(config.detection_model_path+"if_normals_{}/".format(int(config.if_detection_normals)) + "train_loss_singleGPU.pth"))
# model.load_state_dict(torch.load("logs/trained_models/{}/train_loss_singleGPU.pth".format(model_name)))

model.cuda()

evaluation = Evaluation(config)
dataset = Dataset(config,shuffle=False)

get_test_data = dataset.get_test(d_mean=config.d_mean, d_scale=config.d_scale)

loader = generator_iter(get_test_data, int(1e10))
get_test_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)

prev_test_loss = 1e4
print("started testing!")

if torch.cuda.device_count() > 1:
    alt_gpu = 1
else:
    alt_gpu = 0

lamb_0 = [config.lamb_0_0, config.lamb_0_1, config.lamb_0_2, config.lamb_0_3,config.lamb_0_4,config.lamb_0_5,config.lamb_0_6]
lamb_1 = config.lamb_1
    
test_iou = []
test_seg_iou = []

# clear results dir
results_dir = "logs/results_vis/{}".format(model_name)
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
    os.makedirs(results_dir,exist_ok=True,)
else:
    os.makedirs(results_dir,exist_ok=True,)

torch.cuda.empty_cache()
model.eval()
utils_vis = utils_vis()

for test_b_id in range(dataset.test_points.shape[0] // config.batch_size):

    seg_iou = 0
    type_iou = 0
    distance = 0

    save_result_dir_object = results_dir+"/{}".format(test_b_id)
    if os.path.exists(save_result_dir_object):
        shutil.rmtree(save_result_dir_object)
        os.makedirs(save_result_dir_object,exist_ok=True,)
    else:
        os.makedirs(save_result_dir_object,exist_ok=True,)

    points_,points_raw,normals_, quadrics_,quadrics_raw, T_batch, labels_, primitives_,test_data_index = next(get_test_data)[0]
    
    points = torch.from_numpy(points_.astype(np.float32)).cuda()
    normals = torch.from_numpy(normals_.astype(np.float32)).cuda()

    quadrics = torch.from_numpy(quadrics_.astype(np.float32)).cuda(alt_gpu)
    primitives = torch.from_numpy(primitives_.astype(np.int64)).cuda()

    with torch.no_grad():
        if config.if_detection_normals:
            embedding, primitives_log_prob, embed_loss = model(torch.cat([points.permute(0, 2, 1),normals.permute(0, 2, 1)],1), torch.from_numpy(labels_).cuda(), True
            )
        else:
            embedding, primitives_log_prob, embed_loss = model(
                points.permute(0, 2, 1), torch.from_numpy(labels_).cuda(), True
            )

        prim_loss = primitive_loss(primitives_log_prob, primitives)
        embed_loss = torch.mean(embed_loss)
        
        metric, _,T_batch_sample,scale_quadrics_batch_sample,quadrics_pre_batch,quadrics_gt_batch,clustered_points_batch,clustered_points_input_batch,clustered_primitives_batch,clustered_primitives_gt_batch,clustered_labels_gt_batch = evaluation.fitting_loss(
                embedding.permute(0, 2, 1).to(torch.device("cuda:{}".format(alt_gpu))),
                points.to(torch.device("cuda:{}".format(alt_gpu))),
                normals.to(torch.device("cuda:{}".format(alt_gpu))),
                quadrics,
                labels_,
                primitives.to(torch.device("cuda:{}".format(alt_gpu))),
                primitives_log_prob.to(torch.device("cuda:{}".format(alt_gpu))),
                quantile=0.025,
                # iterations=config.cluster_iterations,
                iterations=50,
                lamb=lamb_0,
                eval=True,
                if_fitting_normals = config.if_fitting_normals
            )

        h5_gt_separately_dir = config.dataset_path_separately + "h5/"
        h5_gt_separately = os.listdir(h5_gt_separately_dir)
        # Sort in descending order for the same sequence number and name
        h5_gt_separately.sort(key = lambda x: int(x[:-3]))
        h5_gt_separately_file = h5_gt_separately_dir + h5_gt_separately[test_data_index]
        with h5py.File(h5_gt_separately_file, "r") as h5_gt_separately_file_read:
            points_gt_separately = np.array(h5_gt_separately_file_read.get("points")).astype(np.float32)
            labels_gt_separately = np.array(h5_gt_separately_file_read.get("labels")).astype(np.int64)

        every_clustered_num = []
        object_clustered_points = []
        object_gt_raw_points = []
        invalid_points_num = False
        clustered_points_input_batch = [clustered_points_input_batch[0][i] for i in clustered_points_input_batch[0]]
        clustered_points_batch = [clustered_points_batch[0][i] for i in clustered_points_batch[0]]
        for i in range(len(clustered_points_input_batch)):

            # Skip objects with too few points in the point cloud
            if clustered_points_input_batch[i] == None:
                invalid_points_num = True
                break

            # Save clustered points (object normalization only)
            object_clustered_points.append(clustered_points_batch[i].data.cpu().numpy().reshape(-1,3))
            # Save the number of clustered points
            every_clustered_num.append(clustered_points_batch[i].shape[0])

            # Save the corresponding label's gt point cloud, according to the clustering results
            object_gt_raw_points.append(points_gt_separately[labels_gt_separately==[clustered_labels_gt_batch[0][i] for i in clustered_labels_gt_batch[0]][i]].reshape(-1,3))

        # When the number of point clouds in the shape is too small or there is only one cluster, skip
        if invalid_points_num or i==0:
            continue

    # metric：[res_loss, quadrics_reg_loss, quadrics_function_loss, r,s,t,reguRo,seg_iou, type_iou]
    res_loss = metric[0].to(torch.device("cuda:0"))

    seg_iou, type_iou = metric[8:]
    loss = embed_loss + prim_loss + lamb_1 * res_loss

    # Quadrics coefficients that have not been recovered for preprocessing
    quadrics_gt_batch_scaled = np.squeeze(torch.stack(quadrics_gt_batch[0],0).data.cpu().numpy())
    quadrics_pre_batch_scaled = np.squeeze(torch.stack(quadrics_pre_batch[0],0).data.cpu().numpy())

    # Quadrics coefficients that have been recovered for preprocessing
    if config.d_scale:
        quadrics_pre_batch,quadrics_gt_batch = rescale_input_outputs_quadrics_e2e(T_batch,T_batch_sample,scale_quadrics_batch_sample, quadrics_gt_batch, quadrics_pre_batch, config.batch_size)


    quadrics_gt_batch = np.squeeze(quadrics_gt_batch,0)
    quadrics_pre_batch = np.squeeze(quadrics_pre_batch,0)

    T_batch_sample = np.squeeze(torch.stack(T_batch_sample[0],0).data.cpu().numpy())
    T_object = T_batch[0]

    clustered_primitives_batch = np.stack([clustered_primitives_batch[0][i].data.cpu().numpy() for i in clustered_primitives_batch[0]])
    clustered_primitives_gt_batch = np.stack([clustered_primitives_gt_batch[0][i].data.cpu().numpy() for i in clustered_primitives_gt_batch[0]])

    points_clustered_reconstruction_object = []
    points_raw_gt_object = []
    # vislization
    for shape_index in range(quadrics_gt_batch_scaled.shape[0]):
        
        resolution = 1*1e-2
        thres = 0.01

        T_shape = T_batch_sample[shape_index]
        q_pre = quadrics_pre_batch_scaled[shape_index]
        q_gt = quadrics_gt_batch_scaled[shape_index]
        primitive_shape_pre = clustered_primitives_batch[shape_index]

        points_shape_clusterd = object_clustered_points[shape_index]
        points_shape_clusterd_scaled = np.matmul(T_shape,np.concatenate((points_shape_clusterd,np.ones((points_shape_clusterd.shape[0],1))),1).transpose()).transpose()[:,0:3]

        points_shape_gt_raw = object_gt_raw_points[shape_index]
        T_shape_object = np.matmul(T_shape, T_object)
        points_shape_gt_raw_scaled = np.matmul(T_shape_object,np.concatenate((points_shape_gt_raw,np.ones((points_shape_gt_raw.shape[0],1))),1).transpose()).transpose()[:,0:3]

        if_trim="1"
        if primitive_shape_pre == 1:
            # plane
            mesh_size = utils_vis.bound_box(points_shape_gt_raw_scaled)+0.1*np.array([[-1,1],[-1,1],[-1,1]])
            error = 1e-3
            res = resolution*(mesh_size[:,1]- mesh_size[:,0])
        elif primitive_shape_pre == 0:
            # sphere
            mesh_size = utils_vis.bound_box(points_shape_gt_raw_scaled)+0.1*np.array([[-1,1],[-1,1],[-1,1]])
            res = resolution*(mesh_size[:,1]- mesh_size[:,0])
            error = 1e-3
            margin_pre = [1,1,1]
        else:
            mesh_size = utils_vis.bound_box(points_shape_gt_raw_scaled)+0.1*np.array([[-1,1],[-1,1],[-1,1]])
            res = resolution*(mesh_size[:,1]- mesh_size[:,0])
            error = 1e-6
            margin_pre = [0,0,0]

        try:
            if primitive_shape_pre == 1:
                # continue
                points_clustered_reconstruction_shape_temp = utils_vis.plane_trim(points_shape_gt_raw_scaled,q_pre,mesh_size,res,error)
            else:
                points_clustered_reconstruction_shape_temp = utils_vis.others_trim(q_gt,points_shape_gt_raw_scaled,q_pre,"1",mesh_size,res,error,margin_pre,if_trim,primitive_shape_pre)
            
            if points_clustered_reconstruction_shape_temp.shape[0] == 0:
                continue_signal = 1
                break
        except:
            continue_signal = 1
            break

        points_clustered_reconstruction_shape =  np.matmul(np.linalg.inv(T_shape),np.concatenate((points_clustered_reconstruction_shape_temp,np.ones((points_clustered_reconstruction_shape_temp.shape[0],1))),1).transpose()).transpose()[:,0:3]
        points_shape_gt_raw_scaled = np.matmul(np.linalg.inv(T_shape),np.concatenate((points_shape_gt_raw_scaled,np.ones((points_shape_gt_raw_scaled.shape[0],1))),1).transpose()).transpose()[:,0:3]

        points_clustered_reconstruction_object.append(points_clustered_reconstruction_shape)
        points_raw_gt_object.append(points_shape_gt_raw_scaled)

    for semgent_index, semgent in enumerate(points_clustered_reconstruction_object):
        if semgent_index == 0:
            points_clustered_reconstruction_object_save = semgent
            points_raw_gt_object_save = points_raw_gt_object[semgent_index]
        else:
            points_clustered_reconstruction_object_save = np.concatenate((points_clustered_reconstruction_object_save,semgent),0)
            points_raw_gt_object_save = np.concatenate((points_raw_gt_object_save,points_raw_gt_object[semgent_index]),0)

    points_reconstruction_temp = np.expand_dims(points_clustered_reconstruction_object_save,1)
    points_gt_temp = np.expand_dims(points_raw_gt_object_save,0)
    # diff: [num_reconstruction_points, num_gt_points, 3]
    diff = points_reconstruction_temp - points_gt_temp
    # diff: [num_reconstruction_points, num_gt_points]
    diff = utils_vis.guard_sqrt(np.sum(diff ** 2,2))

    # diff = np.sqrt(np.sum(diff ** 2,2))
    distance_0 = np.mean(np.min(diff,1))
    distance_1 = np.mean(np.min(diff,0))
    distance = np.mean([distance_0,distance_1])

    pcd_reconstruction = o3d.geometry.PointCloud()
    pcd_reconstruction.points = o3d.utility.Vector3dVector(points_clustered_reconstruction_object_save)
    o3d.io.write_point_cloud(save_result_dir_object+"/reconstruction.ply", pcd_reconstruction)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points_raw_gt_object_save)
    o3d.io.write_point_cloud(save_result_dir_object+"/gt.ply", pcd_gt)

    print("Sample: {}, seg_iou: {:.4}, type_iou: {:.4}, res: {:.4}".format(
            test_b_id,seg_iou.item(),type_iou.item(),distance.item()
        )
    )