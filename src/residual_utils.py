import sys
sys.path.append("../")

import time
from scipy import stats
from src.primitives import ResidualLoss
from src.primitive_forward import fit_one_shape_torch
from src.fitting_optimization import FittingModule
import numpy as np

from src.fitting_utils import (
    to_one_hot,
    match,
)
from src.mean_shift import MeanShift
from src.segment_utils import SIOU_matched_segments
import torch

class Evaluation:
    def __init__(self, config):

        config.sphere_path = config.fitting_model_path + "if_normals_{}/".format(int(config.if_fitting_normals[0])) + "sphere"
        config.plane_path = config.fitting_model_path + "if_normals_{}/".format(int(config.if_fitting_normals[1])) + "plane"
        config.cylinder_path = config.fitting_model_path + "if_normals_{}/".format(int(config.if_fitting_normals[2])) + "cylinder"
        config.cone_path = config.fitting_model_path + "if_normals_{}/".format(int(config.if_fitting_normals[3])) + "cone"

        self.res_loss = ResidualLoss()
        self.fitter = FittingModule(config)

        for param in self.fitter.sphere_decoder.parameters():
            param.requires_grad = False
        for param in self.fitter.plane_decoder.parameters():
            param.requires_grad = False
        for param in self.fitter.cylinder_decoder.parameters():
            param.requires_grad = False
        for param in self.fitter.cone_decoder.parameters():
            param.requires_grad = False

        self.ms = MeanShift()

    def guard_mean_shift(self, embedding, quantile, iterations, kernel_type="gaussian"):
        """
        Some times if band width is small, number of cluster can be larger than 50, that
        but we would like to keep max clusters 50 as it is the max number in our dataset.
        In that case you increase the quantile to increase the band width to decrease
        the number of clusters.
        """
        while True:
            _, center, bandwidth, cluster_ids = self.ms.mean_shift(
                embedding, 10000, quantile, iterations, kernel_type=kernel_type
            )
            if torch.unique(cluster_ids).shape[0] > 49:
                quantile *= 1.2
            else:
                break
        return center, bandwidth, cluster_ids

    def fitting_loss(
            self,
            embedding,
            points,
            normals,
            quadrics,
            labels,
            primitives,
            primitives_log_prob,
            quantile=0.125,
            iterations=50,
            lamb=1.0,
            eval=False,
            if_fitting_normals=[0,0,0,0]
    ):

        batch_size = embedding.shape[0]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=2)
        parameters = None
        weights_batch = []
        matching_batch = []
        cluster_ids_batch = []
        res_loss_batch = []
        loss_quadarics_reg_batch = []
        loss_quadrics_function_batch = []
        loss_decomposition_r_batch = []
        loss_decomposition_s_batch = []
        loss_decomposition_t_batch = []
        loss_normals_deviation_batch = []
        loss_r_regularization_batch = []

        T_batch = []
        scale_quadrics_batch = []
        quadrics_pre_batch = []
        quadrics_gt_batch = []
        clustered_points_batch = []
        clustered_points_input_batch = []
        clustered_primitives_batch = []
        clustered_primitives_gt_batch = []
        clustered_labels_gt_batch = []

        primitives_log_prob = torch.max(primitives_log_prob, 1)[1]
        # primitives_log_prob = primitives_log_prob.data.cpu().numpy()

        for b in range(batch_size):
            center, bandwidth, cluster_ids = self.guard_mean_shift(
                embedding[b], quantile, iterations, kernel_type="gaussian"
            )

            # 计算残差
            if not eval:
                res_loss, loss_quadarics_reg, loss_quadrics_function, loss_decomposition_r,loss_decomposition_s,loss_decomposition_t,loss_normals_deviation,loss_r_regularization,T,scale_quadrics,quadrics_pre,quadrics_gt,clustered_points,clustered_points_input,clustered_primitives,clustered_primitives_gt,clustered_labels_gt,parameters, rows, cols, weights = self.residual_train_mode(
                    points[b],
                    normals[b],
                    quadrics[b],
                    labels[b],
                    cluster_ids,
                    primitives[b],
                    primitives_log_prob[b],
                    lamb=lamb,
                    if_fitting_normals = if_fitting_normals,
                    eval=eval,
                )
            else:
                res_loss, loss_quadarics_reg, loss_quadrics_function, loss_decomposition_r,loss_decomposition_s,loss_decomposition_t,loss_normals_deviation,loss_r_regularization,T,scale_quadrics,quadrics_pre,quadrics_gt,clustered_points,clustered_points_input,clustered_primitives,clustered_primitives_gt,clustered_labels_gt,parameters, rows, cols, weights = self.residual_eval_mode(
                    points[b],
                    normals[b],
                    quadrics[b],
                    labels[b],
                    cluster_ids,
                    primitives[b],
                    primitives_log_prob[b],
                    lamb=lamb,
                    if_fitting_normals = if_fitting_normals,
                    eval=eval,
                )

            weights_batch.append(weights)
            matching_batch.append([rows, cols])
            cluster_ids_batch.append(cluster_ids.data.cpu().numpy())
            res_loss_batch.append(res_loss)
            loss_quadarics_reg_batch.append(loss_quadarics_reg)
            loss_quadrics_function_batch.append(loss_quadrics_function)
            loss_decomposition_r_batch.append(loss_decomposition_r)
            loss_decomposition_s_batch.append(loss_decomposition_s)
            loss_decomposition_t_batch.append(loss_decomposition_t)
            loss_normals_deviation_batch.append(loss_normals_deviation)
            loss_r_regularization_batch.append(loss_r_regularization)
            T_batch.append(T)
            scale_quadrics_batch.append(scale_quadrics)
            quadrics_pre_batch.append(quadrics_pre)
            quadrics_gt_batch.append(quadrics_gt)
            clustered_points_batch.append(clustered_points)
            clustered_points_input_batch.append(clustered_points_input)
            clustered_primitives_batch.append(clustered_primitives)
            clustered_primitives_gt_batch.append(clustered_primitives_gt)
            clustered_labels_gt_batch.append(clustered_labels_gt)

        with torch.no_grad():
            s_iou, p_iou, _, _ = SIOU_matched_segments(labels, cluster_ids_batch,primitives_log_prob.cpu().numpy(), primitives.cpu().numpy(), weights_batch,matching_batch)
            
        metric = [torch.mean(torch.stack(res_loss_batch)), torch.mean(torch.stack(loss_quadarics_reg_batch)), torch.mean(torch.stack(loss_quadrics_function_batch)),torch.mean(torch.stack(loss_decomposition_r_batch)),torch.mean(torch.stack(loss_decomposition_s_batch)),torch.mean(torch.stack(loss_decomposition_t_batch)),torch.mean(torch.stack(loss_normals_deviation_batch)),torch.mean(torch.stack(loss_r_regularization_batch))]+ [s_iou, p_iou]
        return metric, [parameters, cluster_ids_batch, weights_batch], T_batch,scale_quadrics_batch,quadrics_pre_batch,quadrics_gt_batch,clustered_points_batch,clustered_points_input_batch,clustered_primitives_batch,clustered_primitives_gt_batch,clustered_labels_gt_batch

    def residual_train_mode(
            self, points,normals, quadrics, labels, cluster_ids, primitives, pred_primitives,lamb=1.0,if_fitting_normals = [0,0,0,0],eval=False):

        if not isinstance(cluster_ids, np.ndarray):
            cluster_ids = cluster_ids.data.cpu().numpy()

        weights = to_one_hot(cluster_ids,
                       np.unique(cluster_ids).shape[0],device_id=points.get_device())

        rows, cols, unique_target, unique_clustered = match(labels, cluster_ids)

        data = []
        clustered_quadrics = {}
        for index, i in enumerate(unique_clustered):

            gt_indices_i = labels == cols[index]
            clustered_indices_i = cluster_ids == i

            if (np.sum(gt_indices_i) == 0) or (np.sum(clustered_indices_i) == 0):
                continue

            primitives_clustered = torch.mode(pred_primitives[clustered_indices_i])[0]
            primitives_clustered_gt = torch.mode(primitives[clustered_indices_i])[0]
            quadrics_clustered = torch.mode(quadrics[clustered_indices_i][:],dim=0)[0]

            clustered_quadrics[i] = quadrics_clustered

            segment_label_gt = cols[index]

            data.append([points[clustered_indices_i], normals[clustered_indices_i],quadrics_clustered, primitives_clustered, clustered_indices_i, (index, i),primitives_clustered_gt,segment_label_gt])

        clustered_points,clustered_points_input,clustered_normals_input, clustered_Ts,clustered_primitives,clustered_primitives_gt,clustered_labels_gt = fit_one_shape_torch(
            data, self.fitter, None, eval=eval,if_fitting_normals=if_fitting_normals
        )

        distance = self.res_loss.residual_loss(
            clustered_points_input,clustered_normals_input,clustered_quadrics,self.fitter.parameters,clustered_Ts,eval=eval
        )
        # res_loss = loss_quadrics_function * lamb + loss_quadrics_function * (1-lamb)
        res_loss, loss_quadarics_reg, loss_quadrics_function,loss_decomposition_r,loss_decomposition_s,loss_decomposition_t,loss_normals_deviation,loss_r_regularization,T,scale_quadrics,quadrics_pre,quadrics_gt= self.separate_losses(distance, clustered_points_input, lamb=lamb)

        return res_loss, loss_quadarics_reg, loss_quadrics_function, loss_decomposition_r,loss_decomposition_s,loss_decomposition_t,loss_normals_deviation,loss_r_regularization,T,scale_quadrics,quadrics_pre,quadrics_gt,clustered_points,clustered_points_input,clustered_primitives,clustered_primitives_gt,clustered_labels_gt,self.fitter.parameters, rows, cols, weights

    def residual_eval_mode(
            self,
            points,
            normals,
            quadrics,
            labels,
            cluster_ids,
            primitives,
            pred_primitives,
            lamb=1.0,
            if_fitting_normals = [0,0,0,0],
            eval=True,
    ):
        """
        Computes residual error in eval mode.
        """
        if not isinstance(cluster_ids, np.ndarray):
            cluster_ids = cluster_ids.data.cpu().numpy()

        # weights = weights.data.cpu().numpy()
        weights = to_one_hot(cluster_ids,
                       np.unique(cluster_ids).shape[0],
            device_id=points.get_device())

        rows, cols, unique_target, unique_clustered = match(labels, cluster_ids)

        data = []
        gt_quadrics = {}
        for index, i in enumerate(unique_clustered):
            # TODO some labels might be missing from unique_pred
            gt_indices_i = labels == cols[index]
            clustered_indices_i = cluster_ids == i

            if (np.sum(gt_indices_i) == 0) or (np.sum(clustered_indices_i) == 0):
                continue

            primitives_clustered = torch.mode(pred_primitives[clustered_indices_i])[0]
            primitives_clustered_gt = torch.mode(primitives[clustered_indices_i])[0]
            quadrics_clustered = torch.mode(quadrics[clustered_indices_i][:],dim=0)[0]
            gt_quadrics[i] = quadrics_clustered

            segment_label_gt = cols[index]

            data.append(
                [
                    points[clustered_indices_i],
                    normals[clustered_indices_i],
                    quadrics_clustered,
                    primitives_clustered,
                    clustered_indices_i,
                    (index, i),
                    primitives_clustered_gt,
                    segment_label_gt
                ]
                )

        clustered_points,clustered_points_input, clustered_normals_input,clustered_Ts,clustered_primitives,clustered_primitives_gt,cluster_labels_gt= fit_one_shape_torch(
            data,
            self.fitter,
            weights,
            eval=eval,
            if_fitting_normals=if_fitting_normals
        )

        distance = self.res_loss.residual_loss(
            clustered_points_input,clustered_normals_input,gt_quadrics,self.fitter.parameters,clustered_Ts,eval=eval
        )

        res_loss, loss_quadarics_reg, loss_quadrics_function,loss_decomposition_r,loss_decomposition_s,loss_decomposition_t,loss_normals_deviation,loss_r_regularization,T,scale_quadrics,quadrics_pre,quadrics_gt = self.separate_losses(distance, clustered_points_input, lamb=lamb)
 
        return res_loss, loss_quadarics_reg, loss_quadrics_function,loss_decomposition_r,loss_decomposition_s,loss_decomposition_t,loss_normals_deviation,loss_r_regularization,T,scale_quadrics, quadrics_pre,quadrics_gt,clustered_points,clustered_points_input,clustered_primitives,clustered_primitives_gt,cluster_labels_gt,self.fitter.parameters, rows, cols, weights

    def separate_losses(self, distance, clustered_points, lamb=1.0):

        res_loss = []
        loss_quadarics_reg = []
        loss_quadrics_function = []
        loss_decomposition_r = []
        loss_decomposition_s = []
        loss_decomposition_t = []
        loss_normals_deviation = []
        loss_r_regularization = []

        T = []
        scale_quadrics = []
        quadrics_pre = []
        quadrics_gt = []
        for item, v in enumerate(sorted(clustered_points.keys())):

            if clustered_points[v] is None:
                continue

            # frrstnr
            res_loss.append(distance[v][1][1] * lamb[0] + distance[v][1][0] * lamb[1] + distance[v][1][2] * lamb[2] + distance[v][1][3] * lamb[3] + distance[v][1][4] * lamb[4] + distance[v][1][5] * lamb[5] + distance[v][1][6] * lamb[6])
            loss_quadarics_reg.append(distance[v][1][0])
            loss_quadrics_function.append(distance[v][1][1])
            loss_decomposition_r.append(distance[v][1][2])
            loss_decomposition_s.append(distance[v][1][3])
            loss_decomposition_t.append(distance[v][1][4])
            loss_normals_deviation.append(distance[v][1][5])
            loss_r_regularization.append(distance[v][1][6])

            T.append(distance[v][2])
            scale_quadrics.append(distance[v][3])
            quadrics_pre.append(distance[v][4])
            quadrics_gt.append(distance[v][5])

        res_loss = torch.mean(torch.stack(res_loss))
        loss_quadarics_reg = torch.mean(torch.stack(loss_quadarics_reg))
        loss_quadrics_function = torch.mean(torch.stack(loss_quadrics_function))
        loss_decomposition_r = torch.mean(torch.stack(loss_decomposition_r))
        loss_decomposition_s = torch.mean(torch.stack(loss_decomposition_s))
        loss_decomposition_t = torch.mean(torch.stack(loss_decomposition_t))
        loss_normals_deviation = torch.mean(torch.stack(loss_normals_deviation))
        loss_r_regularization = torch.mean(torch.stack(loss_r_regularization))

        return res_loss, loss_quadarics_reg, loss_quadrics_function,loss_decomposition_r,loss_decomposition_s,loss_decomposition_t,loss_r_regularization,loss_normals_deviation,T,scale_quadrics,quadrics_pre,quadrics_gt