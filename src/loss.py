import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.nn import MSELoss
from src.mean_shift import MeanShift
from torch.nn import ReLU

meanshift = MeanShift()
WEIGHT = False
relu = ReLU()

if WEIGHT:
    nllloss = torch.nn.NLLLoss(weight=old_weight)
else:
    nllloss = torch.nn.NLLLoss()

class EmbeddingLoss:
    def __init__(self, margin=1.0, if_mean_shift=False):
        """
        Defines loss function to train embedding network.
        :param margin: margin to be used in triplet loss.
        :param if_mean_shift: bool, whether to use mean shift
        iterations. This is only used in end to end training.
        """
        self.margin = margin
        self.if_mean_shift = if_mean_shift

    def triplet_loss(self, output, labels: np.ndarray, iterations=5):
        """
        Triplet loss
        :param output: output embedding from the network. size: B x 128 x N
        where B is the batch size, 128 is the dim size and N is the number of points.
        :param labels: B x N
        """
        max_segments = 5
        batch_size = output.shape[0]
        N = output.shape[2]
        loss_diff = torch.tensor([0.], requires_grad=True).cuda()
        relu = torch.nn.ReLU()

        output = output.permute(0, 2, 1)
        output = torch.nn.functional.normalize(output, p=2, dim=2)
        new_output = []

        if self.if_mean_shift:
            for b in range(batch_size):
                new_X, bw = meanshift.mean_shift(output[b], 4000,
                                                 0.015, iterations=iterations,
                                                 nms=False)
                new_output.append(new_X)
            output = torch.stack(new_output, 0)

        num_sample_points = {}
        sampled_points = {}
        for i in range(batch_size):
            sampled_points[i] = {}
            p = labels[i]
            unique_labels = np.unique(p)

            # number of points from each cluster.
            num_sample_points[i] = min([N // unique_labels.shape[0] + 1, 30])
            for l in unique_labels:
                ix = np.isin(p, l)
                sampled_indices = np.where(ix)[0]
                # point indices that belong to a certain cluster.
                sampled_points[i][l] = np.random.choice(
                    list(sampled_indices),
                    num_sample_points[i],
                    replace=True)

        sampled_predictions = {}
        for i in range(batch_size):
            sampled_predictions[i] = {}
            for k, v in sampled_points[i].items():
                pred = output[i, v, :]
                sampled_predictions[i][k] = pred

        all_satisfied = 0
        only_one_segments = 0
        for i in range(batch_size):
            len_keys = len(sampled_predictions[i].keys())
            keys = list(sorted(sampled_predictions[i].keys()))
            num_iterations = min([max_segments * max_segments, len_keys * len_keys])
            normalization = 0
            if len_keys == 1:
                only_one_segments += 1
                continue

            loss_shape = torch.tensor([0.], requires_grad=True).cuda()
            for _ in range(num_iterations):
                k1 = np.random.choice(len_keys, 1)[0]
                k2 = np.random.choice(len_keys, 1)[0]
                if k1 == k2:
                    continue
                else:
                    normalization += 1

                pred1 = sampled_predictions[i][keys[k1]]
                pred2 = sampled_predictions[i][keys[k2]]

                Anchor = pred1.unsqueeze(1)
                Pos = pred1.unsqueeze(0)
                Neg = pred2.unsqueeze(0)

                diff_pos = torch.sum(torch.pow((Anchor - Pos), 2), 2)
                diff_neg = torch.sum(torch.pow((Anchor - Neg), 2), 2)
                constraint = diff_pos - diff_neg + self.margin
                constraint = relu(constraint)

                # remove diagonals corresponding to same points in anchors
                loss = torch.sum(constraint) - constraint.trace()

                satisfied = torch.sum(constraint > 0) + 1.0
                satisfied = satisfied.type(torch.cuda.FloatTensor)

                loss_shape = loss_shape + loss / satisfied.detach()  # 这里应该是求平均损失

            loss_shape = loss_shape / (normalization + 1e-8)
            loss_diff = loss_diff + loss_shape
        loss_diff = loss_diff / (batch_size - only_one_segments + 1e-8)
        return loss_diff


def evaluate_miou(gt_labels, pred_labels):
    N = gt_labels.shape[0]
    C = pred_labels.shape[2]
    pred_labels = np.argmax(pred_labels, 2)
    IoU_category = 0

    for n in range(N):
        label_gt = gt_labels[n]
        label_pred = pred_labels[n]
        IoU_part = 0.0

        for label_idx in range(C):
            locations_gt = (label_gt == label_idx)
            locations_pred = (label_pred == label_idx)
            I_locations = np.logical_and(locations_gt, locations_pred)
            U_locations = np.logical_or(locations_gt, locations_pred)
            I = np.sum(I_locations) + np.finfo(np.float32).eps
            U = np.sum(U_locations) + np.finfo(np.float32).eps
            IoU_part = IoU_part + I / U
        IoU_sample = IoU_part / C
        IoU_category += IoU_sample
    return IoU_category / N


def primitive_loss(pred, gt):
    return nllloss(pred, gt)


def quadrics_reg_loss(output, quadrics,config):
    q = output

    q = torch.unsqueeze(q, 2)

    # N x 8 x grid_size x grid_size x 3
    loss_quadrics_reg = (q - quadrics) ** 2
    loss_quadrics_reg = torch.mean(loss_quadrics_reg)

    return loss_quadrics_reg


def quadrics_function_loss(output, points, config, quadrics=0):
    q = output

    for i in range(config.batch_size):
        Q_temp = torch.tensor([[[q[i, 0], q[i, 3], q[i, 4], q[i, 6]],
                                [q[i, 3], q[i, 1], q[i, 5], q[i, 7]],
                                [q[i, 4], q[i, 5], q[i, 2], q[i, 8]],
                                [q[i, 6], q[i, 7], q[i, 8], q[i, 9]]]])
        if i == 0:
            Q = Q_temp
        else:
            Q = torch.cat((Q, Q_temp), 0)

    Q = Q.cuda()

    append_one = torch.ones(points.size(0), 1, points.size(2))
    append_one = append_one.cuda()
    points_append = torch.cat((points, append_one), 1)

    # loss_quadrics_function = mean(x*Q*xT)
    for i in range(config.batch_size):
        points_each = points_append[i].unsqueeze(0)
        points_each = points_each.permute(2, 0, 1)
        Q_each = Q[i]
        loss_quadrics_function_each = torch.matmul(torch.matmul(points_each, Q_each), points_each.transpose(2, 1)).pow(2)
        if i == 0:
            loss_quadrics_function = loss_quadrics_function_each
        else:
            loss_quadrics_function = torch.cat((loss_quadrics_function, loss_quadrics_function_each), 0)
    loss_quadrics_function = torch.mean(loss_quadrics_function)

    return loss_quadrics_function

def Taubin_distance_loss(output, points, config, quadrics=0):
    q = output

    for i in range(config.batch_size):
        Q_temp = torch.tensor([[[q[i, 0], q[i, 3], q[i, 4], q[i, 6]],
                                [q[i, 3], q[i, 1], q[i, 5], q[i, 7]],
                                [q[i, 4], q[i, 5], q[i, 2], q[i, 8]],
                                [q[i, 6], q[i, 7], q[i, 8], q[i, 9]]]])
        if i == 0:
            Q = Q_temp
        else:
            Q = torch.cat((Q, Q_temp), 0)

    Q = Q.cuda()

    append_one = torch.ones(points.size(0), 1, points.size(2))
    append_one = append_one.cuda()
    points_append = torch.cat((points, append_one), 1)

    # loss_quadrics_function = mean(x*Q*xT)
    for i in range(config.batch_size):
        points_each = points_append[i].unsqueeze(0)
        points_each = points_each.permute(2, 0, 1)
        Q_each = Q[i]
        quadrics_function_each = torch.matmul(torch.matmul(points_each, Q_each), points_each.transpose(2, 1)).pow(2).squeeze()
        deta_function_each = torch.norm(compute_normals_analytically_torch(points[i].permute(1,0),q[i],if_normalize=False),dim=1,p=2).pow(2)
        loss_Taubin_distance_each = quadrics_function_each / (deta_function_each + 1e-8)
        if i == 0:
            loss_Taubin_distance = loss_Taubin_distance_each
        else:
            loss_Taubin_distance = torch.cat((loss_Taubin_distance, loss_Taubin_distance_each), 0)
    loss_Taubin_distance = torch.mean(loss_Taubin_distance)

    return loss_Taubin_distance

def quadrics_decomposition_loss(output, config, quadrics,trans_inv,C,mode="train"):

    q_gt = quadrics.squeeze(2)
    q_pre = output

    for i in range(config.batch_size):
        C_pre_each = C[i]
        if (mode == "eval"):
            if config.shape in ["sphere","ellipsoid","cylinder","elliptic_cylinder"]:
                Q_gt_each = torch.tensor([[q_gt[i, 0], q_gt[i, 3], q_gt[i, 4], q_gt[i, 6]],
                                        [q_gt[i, 3], q_gt[i, 1], q_gt[i, 5], q_gt[i, 7]],
                                        [q_gt[i, 4], q_gt[i, 5], q_gt[i, 2], q_gt[i, 8]],
                                        [q_gt[i, 6], q_gt[i, 7], q_gt[i, 8], q_gt[i, 9]]]).cuda(q_gt.device)
                Q_pre_each = torch.tensor([[q_pre[i, 0], q_pre[i, 3], q_pre[i, 4], q_pre[i, 6]],
                                        [q_pre[i, 3], q_pre[i, 1], q_pre[i, 5], q_pre[i, 7]],
                                        [q_pre[i, 4], q_pre[i, 5], q_pre[i, 2], q_pre[i, 8]],
                                        [q_pre[i, 6], q_pre[i, 7], q_pre[i, 8], q_pre[i, 9]]]).cuda(q_pre.device)
                Q_gt_each,_ = quadrics_scale_identification(Q_gt_each)
                Q_pre_each,_ = quadrics_scale_identification(Q_pre_each)
            
            elif config.shape in ["plane","cone","elliptic_cone"]:

                q_gt_ = F.normalize(q_gt[i],p=2,dim=0)
                q_pre_ = F.normalize(q_pre[i],p=2,dim=0)

                Q_gt_each = torch.tensor([[q_gt_[0], q_gt_[3], q_gt_[4], q_gt_[6]],
                                        [q_gt_[3], q_gt_[1], q_gt_[5], q_gt_[7]],
                                        [q_gt_[4], q_gt_[5], q_gt_[2], q_gt_[8]],
                                        [q_gt_[6], q_gt_[7], q_gt_[8], q_gt_[9]]]).cuda(q_gt.device)
                Q_pre_each = torch.tensor([[q_pre_[0], q_pre_[3], q_pre_[4], q_pre_[6]],
                                        [q_pre_[3], q_pre_[1], q_pre_[5], q_pre_[7]],
                                        [q_pre_[4], q_pre_[5], q_pre_[2], q_pre_[8]],
                                        [q_pre_[6], q_pre_[7], q_pre_[8], q_pre_[9]]]).cuda(q_pre.device)

        elif mode == "train":
            if config.shape in ["sphere","ellipsoid","cylinder","elliptic_cylinder"]:
                Q_gt_each = torch.tensor([[q_gt[i, 0], q_gt[i, 3], q_gt[i, 4], q_gt[i, 6]],
                                        [q_gt[i, 3], q_gt[i, 1], q_gt[i, 5], q_gt[i, 7]],
                                        [q_gt[i, 4], q_gt[i, 5], q_gt[i, 2], q_gt[i, 8]],
                                        [q_gt[i, 6], q_gt[i, 7], q_gt[i, 8], q_gt[i, 9]]]).cuda(q_gt.device)
                _,scale_identification_gt = quadrics_scale_identification(Q_gt_each)
                Is_add = 1
            elif config.shape in ["plane","cone","elliptic_cone"]:
                Q_gt_each = torch.tensor([[q_gt[i, 0], q_gt[i, 3], q_gt[i, 4], q_gt[i, 6]],
                                        [q_gt[i, 3], q_gt[i, 1], q_gt[i, 5], q_gt[i, 7]],
                                        [q_gt[i, 4], q_gt[i, 5], q_gt[i, 2], q_gt[i, 8]],
                                        [q_gt[i, 6], q_gt[i, 7], q_gt[i, 8], q_gt[i, 9]]]).cuda(q_gt.device)
                scale_identification_gt = 0
                Is_add = 0

            
        ###########################
        # trans_inv=[R.T, -R.T*t
        #             0  ,  1  ]
        trans_t_ = trans_inv[i][0:3,3] # -R.T*t
        trans_r = trans_inv[i][0:3,0:3].transpose(0,1)
        trans_t = -torch.matmul(trans_r,trans_t_)

        # upper 3x3 matrix  
        E_gt_each = Q_gt_each[0:3,0:3]

        ###########################
        # gt eigen
        # get the eigenvalue and eigenvector and sort they in the descend order.
        value_gt_each,vector_gt_each = torch.eig(E_gt_each, eigenvectors=True)
        value_gt_each_sorted,idx_gt_each = torch.sort(value_gt_each[:,0],descending=True)
        Is_gt_each,Ir_gt_each,It_gt_each = quadrics_judgment(value_gt_each_sorted)
        vector_gt_each_sorted = vector_gt_each[:,idx_gt_each]
        scale_gt_each_sorted = torch.sqrt(1 / ((Is_gt_each * value_gt_each_sorted) + 1e-8))
        scale_gt_each_sorted = torch.diag_embed(scale_gt_each_sorted).cuda(q_gt.device)
        value_gt_each_sorted = torch.diag_embed(value_gt_each_sorted).cuda(q_gt.device)

        ###########################
        # pre eigen
        if mode == "train":
            value_pre_each_sorted,idx_pre_each = torch.sort(torch.diag(C_pre_each)[0:3],descending=True)
            vector_pre_each_sorted = trans_r[:,idx_pre_each]

            scale_pre_each_sorted = torch.sqrt(1 / ((Is_gt_each * value_pre_each_sorted)+ 1e-8))
        elif mode == "eval":
            E_pre_each = Q_pre_each[0:3,0:3]
            value_pre_each,vector_pre_each = torch.eig(E_pre_each, eigenvectors=True)
            value_pre_each_sorted,idx_pre_each = torch.sort(value_pre_each[:,0],descending=True)
            vector_pre_each_sorted = vector_pre_each[:,idx_pre_each]

            scale_pre_each_sorted = torch.sqrt(torch.abs(1 / ((Is_gt_each * value_pre_each_sorted)+ 1e-8)))
        scale_pre_each_sorted = torch.diag_embed(scale_pre_each_sorted).cuda(q_gt.device)
        value_pre_each_sorted = torch.diag_embed(value_pre_each_sorted).cuda(q_gt.device)

        ###########################
        if (config.shape in ["cone","elliptic_cone"]) and mode=="eval":
            if sum(torch.diag(value_gt_each_sorted) < 0) == 1:
                factor_gt = -value_gt_each_sorted[value_gt_each_sorted < 0]
                value_gt_each_sorted = value_gt_each_sorted / factor_gt
                Q_gt_each = Q_gt_each / factor_gt

            if sum(torch.diag(value_pre_each_sorted) < 0) == 1:
                factor_pre = -value_pre_each_sorted[value_pre_each_sorted < 0]
                value_pre_each_sorted = value_pre_each_sorted / factor_pre

        ###########################
        # ldr
        if torch.sum(Ir_gt_each) == 0:
            # sphere 
            loss_decomposition_r_each = torch.sum(Ir_gt_each)
        else:
            # sum(((vector_gt x vector_pre) * I_r_gt)^2) / (sum(I_r_gt)*3)
            loss_decomposition_r_each = torch.sum((torch.matmul(torch.cross(vector_gt_each_sorted, vector_pre_each_sorted,dim=0),torch.diag_embed(Ir_gt_each)) **2))/(torch.sum(Ir_gt_each)*3)

        ###########################
        # lds
        # sum(((value_gt - value_pre) * I_r_gt)^2) / (sum(I_r_gt))
        if torch.sum(Is_gt_each) == 0:
            # plane [0 0 0 0]
            loss_decomposition_s_each = torch.sum(Is_gt_each)
        else:
            if mode=="train":
                loss_decomposition_s_each = torch.sum(((value_gt_each_sorted - value_pre_each_sorted)**2))
                loss_decomposition_s_each = loss_decomposition_s_each + ((C_pre_each[3,3] + scale_identification_gt)) ** 2
                loss_decomposition_s_each = loss_decomposition_s_each/(torch.count_nonzero(value_gt_each_sorted)+Is_add)
            elif mode == "eval":
                loss_decomposition_s_each = torch.sum((torch.matmul((scale_gt_each_sorted - scale_pre_each_sorted),torch.diag_embed(Is_gt_each))**2))/torch.sum(Is_gt_each)

        ###########################
        # ldt
        # lamb * v.T * t +  v.T * l   (It=1)
        if torch.sum(It_gt_each) == 0:
            loss_decomposition_t_each = torch.sum(It_gt_each)
        else:
            loss_decomposition_t_each = torch.sum(torch.matmul(torch.matmul(torch.matmul(value_gt_each_sorted,vector_gt_each_sorted.transpose(1,0)),trans_t) + torch.matmul(vector_gt_each_sorted.transpose(1,0),Q_gt_each[0:3,3]),torch.diag_embed(It_gt_each))**2)/torch.sum(It_gt_each)
        
        ###########################
        if i == 0:
            loss_decomposition_r = loss_decomposition_r_each.unsqueeze(0)
            loss_decomposition_s = loss_decomposition_s_each.unsqueeze(0)
            loss_decomposition_t = loss_decomposition_t_each.unsqueeze(0)

        else:
            loss_decomposition_r = torch.cat((loss_decomposition_r, loss_decomposition_r_each.unsqueeze(0)), 0)
            loss_decomposition_s = torch.cat((loss_decomposition_s, loss_decomposition_s_each.unsqueeze(0)), 0)
            loss_decomposition_t = torch.cat((loss_decomposition_t, loss_decomposition_t_each.unsqueeze(0)), 0)

    loss_decomposition_r = torch.mean(loss_decomposition_r)
    loss_decomposition_s = torch.mean(loss_decomposition_s)
    loss_decomposition_t = torch.mean(loss_decomposition_t)

    return loss_decomposition_r,loss_decomposition_s,loss_decomposition_t


def quadrics_scale_identification(Q):
    eigenvalue_Q,_ = torch.eig(Q,eigenvectors=False)
    eigenvalue_Q = eigenvalue_Q[:,0]

    eigenvalue_Q_sum = torch.sum(torch.abs(eigenvalue_Q))
    eigenvalue_Q = eigenvalue_Q[torch.where(torch.abs(eigenvalue_Q) > (eigenvalue_Q_sum * 0.001))]

    scale_Q = torch.tensor([1]).cuda()
    for i in eigenvalue_Q:
        scale_Q = scale_Q * i

    eigenvalue_E,_ = torch.eig(Q[0:3,0:3],eigenvectors=False)
    eigenvalue_E = eigenvalue_E[:,0]

    eigenvalue_E_sum = torch.sum(torch.abs(eigenvalue_E))
    eigenvalue_E = eigenvalue_E[torch.where(torch.abs(eigenvalue_E) > (eigenvalue_E_sum * 0.001))]
    scale_E = torch.tensor([1]).cuda()

    for i in eigenvalue_E:
        scale_E = scale_E * i
    
    scale_identification = torch.abs(scale_E / scale_Q)
    Q = scale_identification * Q
    return Q,np.squeeze(1/scale_identification)

def quadrics_judgment(eigenvalue):

    margin = 1e-5
    x = eigenvalue[1]/eigenvalue[0]
    y = eigenvalue[2]/eigenvalue[0]

    # translation degeneration
    It = (torch.abs(eigenvalue)>margin).float().cuda(eigenvalue.device)

    # scale degeneration
    
    Is = It

    # in case of plane [1 0 0 0]
    if torch.abs(x) < margin and torch.abs(y) < margin:
        Is = torch.tensor([0,0,0]).float().cuda(eigenvalue.device)
    # in case of cylinder [1 1 0 -1]
    if x > margin and torch.abs(y) < margin:
        Is = torch.tensor([1,1,0]).float().cuda(eigenvalue.device)
    # in case of cone [1 1 -1 0]
    if x > margin and y < -margin:
        Is = torch.tensor([1,1,0]).float().cuda(eigenvalue.device)

    # rotation degeneration
    Ir = torch.ones(3).cuda(eigenvalue.device)

    if torch.abs(x - 1) < margin:
        Ir[1] = 0
        Ir[0] = 0
    if torch.abs(x - y) < margin:
        Ir[1] = 0
        Ir[2] = 0
    
    return Is,Ir,It

def normals_deviation_loss(output,points,normals,config,quadrics):
    for i in range(config.batch_size):
        normals_analytical = compute_normals_analytically_torch(points[i].transpose(1,0),output[i])
        loss_normals_deviation_each = torch.mean(torch.abs(torch.cross(normals_analytical,normals[i].transpose(1,0))))

        # normals_analytical = compute_normals_analytically_torch(points[i].transpose(1,0),quadrics[i].squeeze(1))
        # loss_normals_deviation_each_gt = torch.mean(torch.abs(torch.cross(normals_analytical,normals[i].transpose(1,0))))
        if i == 0:
            loss_normals_deviation = loss_normals_deviation_each.unsqueeze(0)
        else:
            loss_normals_deviation = torch.cat((loss_normals_deviation, loss_normals_deviation_each.unsqueeze(0)), 0)

    loss_normals_deviation = torch.mean(loss_normals_deviation)
    return loss_normals_deviation

def compute_normals_analytically_torch(points_temp,quadrics_temp,if_normalize=True):
    untils_zeros = torch.zeros([points_temp.shape[0],1]).cuda()
    untils_ones = torch.ones([points_temp.shape[0],1]).cuda()
    untils_points_x = torch.unsqueeze(points_temp[:,0],axis=-1)
    untils_points_y = torch.unsqueeze(points_temp[:,1],axis=-1)
    untils_points_z = torch.unsqueeze(points_temp[:,2],axis=-1)

    deta_v_0 = torch.cat((2*untils_points_x,untils_zeros,untils_zeros,2*untils_points_y,2*untils_points_z,untils_zeros,2*untils_ones,untils_zeros,untils_zeros,untils_zeros),axis=1)
    deta_v_1 = torch.cat((untils_zeros,2*untils_points_y,untils_zeros,2*untils_points_x,untils_zeros,2*untils_points_z,untils_zeros,2*untils_ones,untils_zeros,untils_zeros),axis=1)
    deta_v_2 = torch.cat((untils_zeros,untils_zeros,2*untils_points_z,untils_zeros,2*untils_points_x,2*untils_points_y,untils_zeros,untils_zeros,2*untils_ones,untils_zeros),axis=1)

    deta_v_0 = torch.unsqueeze(deta_v_0,axis=1)
    deta_v_1 = torch.unsqueeze(deta_v_1,axis=1)
    deta_v_2 = torch.unsqueeze(deta_v_2,axis=1)

    deta_v = torch.cat((deta_v_0,deta_v_1,deta_v_2),axis=1)
    normlas_temp = torch.squeeze(torch.matmul(deta_v,torch.unsqueeze(quadrics_temp,axis=-1)),2)

    if if_normalize:
        normlas_temp = F.normalize(normlas_temp,p=2, dim=1)

    return normlas_temp

mse = MSELoss(size_average=True, reduce=True)