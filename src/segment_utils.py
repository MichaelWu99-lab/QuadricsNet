import numpy as np

random_state = 170
from sklearn.cluster import SpectralClustering, KMeans, MeanShift, estimate_bandwidth

from lapsolver import solve_dense
import torch

def cluster(X, number_cluster, bandwidth=None, alg="kmeans"):
    X = X.astype(np.float32)
    if alg == "kmeans":
        y_pred = KMeans(n_clusters=number_cluster, random_state=random_state).fit_predict(X)

    elif alg == "spectral":
        y_pred = SpectralClustering(n_clusters=number_cluster, random_state=random_state, n_jobs=10).fit_predict(X)

    elif alg == "meanshift":
        # There is a little insight here, the number of neighbors are somewhat
        # dependent on the number of neighbors used in the dynamic graph network.
        if bandwidth:
            pass
        else:
            bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=1000)
        seeds = X[np.random.choice(np.arange(X.shape[0]), 5000)]
        # y_pred = MeanShift(bandwidth=bandwidth).fit_predict(X)
        clustering = MeanShift(bandwidth=bandwidth, seeds=seeds, n_jobs=32).fit(X)
        y_pred = clustering.predict(X)

    if alg == "meanshift":
        return y_pred, clustering.cluster_centers_, bandwidth
    else:
        return y_pred


def cluster_prob(embedding, centers):

    # should of size N x C
    dot_p = np.dot(centers, embedding.transpose()).transpose()

    prob = np.exp(dot_p) / np.expand_dims(np.sum(np.exp(dot_p), 1), 1)
    return prob


def cluster_prob(embedding, centers, band_width):

    dist = 2 - 2 * centers @ embedding.T
    prob = np.exp(-dist / 2 / (band_width)) / np.sqrt(2 * np.pi * band_width)
    return prob


def cluster_prob_mutual(embedding, centers, bandwidth, if_normalize=False):

    # dim: C x N
    dist = np.exp(centers @ embedding.T / bandwidth)
    prob = dist / np.sum(dist, 0, keepdims=True)

    if if_normalize:
        prob = prob - np.min(prob, 1, keepdims=True)
        prob = prob / np.max(prob, 1, keepdims=True)
    return prob


def dot_product_from_cluster_centers(embedding, centers):
    return centers @ embedding.T


def mean_IOU_one_sample(pred, gt, C):
    IoU_part = 0.0
    for label_idx in range(C):
        locations_gt = (gt == label_idx)
        locations_pred = (pred == label_idx)
        I_locations = np.logical_and(locations_gt, locations_pred)
        U_locations = np.logical_or(locations_gt, locations_pred)
        I = np.sum(I_locations) + np.finfo(np.float32).eps
        U = np.sum(U_locations) + np.finfo(np.float32).eps
        IoU_part = IoU_part + I / U
    return IoU_part / C


def SIOU_matched_segments(target, clustered_labels, primitives_pred, primitives, weights,matching):

    primitives_pred_hot = to_one_hot_batch(primitives_pred, 10, weights[0].device.index).float()

    # this gives you what primitive type the predicted segment has.
    prim_pred = primitive_type_segment_torch(primitives_pred_hot, weights)
    # target = np.expand_dims(target, 0)
    # clustered_labels = np.expand_dims(clustered_labels, 0)
    # prim_pred = np.expand_dims(prim_pred, 0)
    # primitives = np.expand_dims(primitives, 0)

    segment_iou, primitive_iou, IOU_prims = mean_IOU_primitive_segment(matching, clustered_labels, target, prim_pred,primitives)
    return segment_iou, primitive_iou, matching, IOU_prims


def mean_IOU_primitive_segment(matching, clustered_labels, labels, pred_prim, gt_prim):

    batch_size = labels.shape[0]
    IOU = []
    IOU_prim = []
    IOU_prims = []

    for b in range(batch_size):
        iou_b = []
        iou_b_prim = []
        iou_b_prims = []
        len_labels = np.unique(clustered_labels[b]).shape[0]
        rows, cols = matching[b]
        count = 0
        for r, c in zip(rows, cols):
            pred_indices = clustered_labels[b] == r
            gt_indices = labels[b] == c

            # use only matched segments for evaluation
            if (np.sum(gt_indices) == 0) or (np.sum(pred_indices) == 0):
                continue

            # also remove the gt labels that are very small in number
            if np.sum(gt_indices) < 100:
                continue

            iou = np.sum(np.logical_and(pred_indices, gt_indices)) / (
                        np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)
            iou_b.append(iou)

            # evaluation of primitive type prediction performance
            gt_prim_type_k = gt_prim[b][gt_indices][0]
            try:
                predicted_prim_type_k = pred_prim[b][r]
            except:
                import ipdb;
                ipdb.set_trace()

            # iou_b_pri不是每个点，而是每一个聚类类别的iou
            iou_b_prim.append(gt_prim_type_k == predicted_prim_type_k)
            iou_b_prims.append([gt_prim_type_k, predicted_prim_type_k])

        # find the mean of IOU over this shape
        IOU.append(np.mean(iou_b))
        IOU_prim.append(np.mean(iou_b_prim))
        IOU_prims.append(iou_b_prims)
    return np.mean(IOU), np.mean(IOU_prim), IOU_prims


def primitive_type_segment(pred, weights):

    d = np.expand_dims(pred, 2) * np.expand_dims(weights, 1)
    d = np.sum(d, 0)
    return np.argmax(d, 0)


def primitive_type_segment_torch(pred, weights):

    batch_size = pred.shape[0]
    primitive_type = []
    for b in range(batch_size):
        d = torch.unsqueeze(pred[b], 2) * torch.unsqueeze(weights[b], 1)
        d = torch.sum(d, 0)
        primitive_type_batch = torch.max(d, 0)[1].data.cpu().numpy()
        primitive_type.append(primitive_type_batch)
    
    return primitive_type


def to_one_hot(target, maxx=50, device_id=0):
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target.astype(np.int8)).cuda(device_id)
    N = target.shape[0]
    target_one_hot = torch.zeros((N, maxx))

    target_one_hot = target_one_hot.cuda(device_id)
    target_t = target.unsqueeze(1)
    target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)
    return target_one_hot

def to_one_hot_batch(target, maxx=50, device_id=0):
    batch_size = target.shape[0]
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target.astype(np.int64)).cuda(device_id)
    N = target.shape[1]
    target_one_hot = torch.zeros((batch_size, N, maxx))

    target_one_hot = target_one_hot.cuda(device_id)
    target_t = target.unsqueeze(2)
    # scatter_第一个参数为dim，即shape中的dim，决定了按照第几个dim进行填充
    target_one_hot = target_one_hot.scatter_(2, target_t.long(), 1)
    return target_one_hot


def matching_iou(matching, predicted_labels, labels):

    batch_size = labels.shape[0]
    IOU = []
    new_pred = []
    for b in range(batch_size):
        iou_b = []
        len_labels = np.unique(predicted_labels[b]).shape[0]
        rows, cols = matching[b]
        count = 0
        for r, c in zip(rows, cols):
            pred_indices = predicted_labels[b] == r
            gt_indices = labels[b] == c

            # if both input and predictions are empty, ignore that.
            if (np.sum(gt_indices) == 0) and (np.sum(pred_indices) == 0):
                continue
            iou = np.sum(np.logical_and(pred_indices, gt_indices)) / (
                        np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)
            iou_b.append(iou)

        # find the mean of IOU over this shape
        IOU.append(np.mean(iou_b))
    return np.mean(IOU)


def relaxed_iou(pred, gt, max_clusters=50):
    batch_size, N, K = pred.shape
    normalize = torch.nn.functional.normalize
    one = torch.ones(1).cuda()

    norms_p = torch.sum(pred, 1)
    norms_g = torch.sum(gt, 1)
    cost = []

    for b in range(batch_size):
        p = pred[b]
        g = gt[b]
        c_batch = []
        dots = p.transpose(1, 0) @ g

        for k1 in range(K):
            c = []
            for k2 in range(K):
                r_iou = dots[k1, k2]
                r_iou = r_iou / (norms_p[b, k1] + norms_g[b, k2] - dots[k1, k2] + 1e-7)
                if (r_iou < 0) or (r_iou > 1):
                    import ipdb;
                    ipdb.set_trace()
                c.append(r_iou)
            c_batch.append(c)
        cost.append(c_batch)
    return cost


def relaxed_iou_fast(pred, gt, max_clusters=50):
    batch_size, N, K = pred.shape
    normalize = torch.nn.functional.normalize
    one = torch.ones(1).cuda()

    norms_p = torch.unsqueeze(torch.sum(pred, 1), 2)
    norms_g = torch.unsqueeze(torch.sum(gt, 1), 1)
    cost = []

    for b in range(batch_size):
        p = pred[b]
        g = gt[b]
        c_batch = []
        dots = p.transpose(1, 0) @ g
        r_iou = dots
        r_iou = r_iou / (norms_p[b] + norms_g[b] - dots + 1e-7)
        cost.append(r_iou)
    cost = torch.stack(cost, 0)
    return cost
