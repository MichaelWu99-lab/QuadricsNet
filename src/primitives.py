import numpy as np
from src.fitting_utils import match
from src.utils import quadrics_reg_distance, quadrics_function_distance,quadrics_decomposition_distance,normals_deviation_distance
import torch
EPS = np.finfo(np.float32).eps


class ResidualLoss:
    """
    Defines distance of points sampled on a patch with corresponding
    predicted patch for different primitives. There is a closed form
    formula for distance from geometric primitives, whereas for splines
    we use chamfer distance as an approximation.
    """

    def __init__(self):
        self.routines = {"sphere": self.distance_from_quadrics,
                         "plane": self.distance_from_quadrics,
                         "cylinder": self.distance_from_quadrics,
                         "cone": self.distance_from_quadrics,
                         }

    def residual_loss(self, points, gt_normals,gt_quadrics ,parameters ,gt_Ts,eval=False):
        distances = {}
        for k, v in parameters.items():
            if v is None:
                continue

            T = gt_Ts[k]
            
            Q_temp = torch.tensor([[gt_quadrics[k][0], gt_quadrics[k][3], gt_quadrics[k][4], gt_quadrics[k][6]],
                               [gt_quadrics[k][3], gt_quadrics[k][1], gt_quadrics[k][5], gt_quadrics[k][7]],
                               [gt_quadrics[k][4], gt_quadrics[k][5], gt_quadrics[k][2], gt_quadrics[k][8]],
                               [gt_quadrics[k][6], gt_quadrics[k][7], gt_quadrics[k][8], gt_quadrics[k][9]]]).float().cuda(points[k].device)
            T_inv = torch.inverse(T)
            Q_temp = torch.matmul(torch.matmul(T_inv.transpose(1,0),Q_temp),T_inv)

            quadrics = torch.tensor([Q_temp[0,0],Q_temp[1,1],Q_temp[2,2],Q_temp[0,1],Q_temp[0,2], Q_temp[1,2],Q_temp[0,3],Q_temp[1,3],Q_temp[2,3],Q_temp[3,3]]).cuda(points[k].device)
            scale_quadrics = torch.norm(quadrics)
            quadrics = quadrics / scale_quadrics

            dist = self.routines[v[0]](points[k],gt_normals[k],quadrics, v[1][0],v[3][0],v[5][0],eval=eval,shape=v[0])

            distances[k] = [v[0], dist, T,scale_quadrics, v[1][0], quadrics]
        return distances

    def distance_from_quadrics(self, points,normals,gt_quadrics, params,trans_inv,C,eval=False,shape=''):

        pred_quadrics = params
        distance_quadrics_reg = quadrics_reg_distance(pred_quadrics,gt_quadrics)
        distance_quadrics_function = quadrics_function_distance(pred_quadrics, points)
        distance_decomposition_r,distance_decomposition_s,distance_decomposition_t = quadrics_decomposition_distance(pred_quadrics,gt_quadrics,trans_inv,C,eval=eval,shape=shape)
        distance_normals_deviation = normals_deviation_distance(pred_quadrics,points,normals,gt_quadrics)

        trans_r = trans_inv[0:3,0:3]
        regularization_trans_r = torch.mean((torch.matmul(trans_r,trans_r.transpose(1,0)) - torch.eye(3).cuda(trans_inv.device))**2)

        return distance_quadrics_reg,distance_quadrics_function,distance_decomposition_r,distance_decomposition_s,distance_decomposition_t,distance_normals_deviation,regularization_trans_r
