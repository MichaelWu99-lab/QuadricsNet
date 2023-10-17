import numpy as np
from src.segment_utils import to_one_hot, matching_iou, relaxed_iou, \
    relaxed_iou_fast
import torch
from lapsolver import solve_dense
from src.utils import visualize_point_cloud
EPS = float(np.finfo(np.float32).eps)
torch.manual_seed(2)
np.random.seed(2)


def one_hot_normalization(weights):
    N, K = weights.shape
    weights = np.argmax(weights, 1)
    one_hot = to_one_hot(weights, K)
    weights = one_hot.float()
    return weights


def SIOU(target, pred_labels):

    labels_one_hot = to_one_hot(target)
    cluster_ids_one_hot = to_one_hot(pred_labels)
    cost = relaxed_iou(torch.unsqueeze(cluster_ids_one_hot, 0).double(), torch.unsqueeze(labels_one_hot, 0).double())
    cost_ = 1.0 - torch.as_tensor(cost)
    cost_ = cost_.data.cpu().numpy()
    matching = []

    for b in range(1):
        rids, cids = solve_dense(cost_[b])
        matching.append([rids, cids])

    s_iou = matching_iou(matching, np.expand_dims(pred_labels, 0), np.expand_dims(target, 0))
    return s_iou


def match(target, clustered_labels):
    labels_one_hot = to_one_hot(target)
    cluster_ids_one_hot = to_one_hot(clustered_labels)

    # cost = relaxed_iou(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())
    # cost_ = 1.0 - torch.as_tensor(cost)
    cost = relaxed_iou_fast(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())

    # cost_ = 1.0 - torch.as_tensor(cost)
    cost_ = 1.0 - cost.data.cpu().numpy()
    rids, cids = solve_dense(cost_[0])

    unique_target = np.unique(target)  # unique函数去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表
    unique_clustered = np.unique(clustered_labels)
    return rids, cids, unique_target, unique_clustered

def up_sample_all(points, normals, weights, cluster_ids, primitives, labels):

    dist = np.expand_dims(points, 1) - np.expand_dims(points, 0)
    dist = np.sum(dist ** 2, 2)
    indices = np.argsort(dist, 1)
    neighbors = points[indices[:, 0:3]]
    centers = np.mean(neighbors, 1)

    new_points = np.concatenate([points, centers])
    new_normals = np.concatenate([normals, normals])
    new_weights = np.concatenate([weights, weights], 1)

    new_primitives = np.concatenate([primitives, primitives])
    new_cluster_ids = np.concatenate([cluster_ids, cluster_ids])
    new_labels = np.concatenate([labels, labels])

    return new_points, new_normals, new_weights, new_primitives, new_cluster_ids, new_labels

def up_sample_all_torch(points, normals):

    dist = torch.unsqueeze(points, 1) - torch.unsqueeze(points, 0)
    dist = torch.sum(dist ** 2, 2)
    _, indices = torch.topk(dist, 10, 1, largest=False)
    neighbors = points[indices[:, 1:]]
    centers = torch.mean(neighbors, 1)

    new_points = torch.cat([points, centers])
    new_normals = torch.cat([normals, normals])

    return new_points, new_normals


def up_sample_points(points, times=1):

    points = points.data.cpu()
    batch_size = points.shape[0]
    points = points.permute(0, 2, 1)

    for t in range(times):
        Points = []
        for b in range(batch_size):
            dist = torch.unsqueeze(points[b], 1) - torch.unsqueeze(points[b], 0)
            dist = torch.sum(dist ** 2, 2)
            _, indices = torch.topk(dist, k=3, dim=1, largest=False)
            neighbors = points[b][indices]
            centers = torch.mean(neighbors, 1)

            new_points = torch.cat([points[b], centers])
            Points.append(new_points)
        points = torch.stack(Points, 0)
    return points.permute(0, 2, 1).cuda()


def up_sample_points_numpy(points, times=1):

    for t in range(times):
        dist = np.expand_dims(points, 1) - np.expand_dims(points, 0)
        dist = np.sum(dist ** 2, 2)
        indices = np.argsort(dist, 1)
        neighbors = points[indices[:, 0:3]]
        centers = np.mean(neighbors, 1)
        points = np.concatenate([points, centers])
    return points


def up_sample_points_torch(points, times=1):

    for t in range(times):
        dist = torch.unsqueeze(points, 1) - torch.unsqueeze(points, 0)
        dist = torch.sum(dist ** 2, 2)
        _, indices = torch.topk(dist, 5, 1, largest=False)
        neighbors = points[indices[:, 1:]]
        centers = torch.mean(neighbors, 1)
        points = torch.cat([points, centers])
    return points

def up_sample_points_torch_memory_efficient(points, times=1):

    for t in range(times):
        # dist = torch.unsqueeze(points, 1) - torch.unsqueeze(points, 0)
        # dist = torch.sum(dist ** 2, 2)
        indices = []
        N = min(points.shape[0], 100)
        for i in range(points.shape[0] // N):
            diff_ = torch.sum((torch.unsqueeze(points[i * N:(i + 1) * N], 1) - torch.unsqueeze(points, 0)) ** 2, 2)
            _, diff_indices = torch.topk(diff_, 5, 1, largest=False)
            indices.append(diff_indices)
        indices = torch.cat(indices, 0)
        # dist = dist_memory_efficient(points, points)
        # _, indices = torch.topk(dist, 5, 1, largest=False)
        neighbors = points[indices[:, 0:]]
        centers = torch.mean(neighbors, 1)
        points = torch.cat([points, centers])
    return points


def dist_memory_efficient(p, q):
    diff = []
    for i in range(p.shape[0]):
        diff.append(torch.sum((torch.unsqueeze(p[i:i + 1], 1) - torch.unsqueeze(q, 0)) ** 2, 2).data.cpu().numpy())
    diff = np.concantenate(diff, 0)
    # diff = torch.sqrt(diff)

    return diff


def up_sample_points_in_range(points, weights, a_min, a_max):
    N = points.shape[0]
    if N > a_max:
        L = np.random.choice(np.arange(N), a_max, replace=False)
        points = points[L]
        weights = weights[L]
        return points, weights
    else:
        while True:
            points = up_sample_points_torch(points)
            weights = torch.cat([weights, weights], 0)
            if points.shape[0] >= a_max:
                break
    N = points.shape[0]
    L = np.random.choice(np.arange(N), a_max, replace=False)
    points = points[L]
    weights = weights[L]
    return points, weights

def up_sample_all_in_range(points, normals, a_max):
    N = points.shape[0]
    if N > a_max:
        L = np.random.choice(np.arange(N), a_max, replace=False)
        points = points[L]
        normals = normals[L]
        return points,normals
    else:
        while True:
            points,normals = up_sample_all_torch(points, normals)
            if points.shape[0] >= a_max:
                break
    N = points.shape[0]
    L = np.random.choice(np.arange(N), a_max, replace=False)
    points = points[L]
    normals = normals[L]
    return points, normals

def up_sample_points_torch_in_range(points, a_min, a_max):
    N = points.shape[0]
    if N > a_max:
        N = points.shape[0]
        L = np.random.choice(np.arange(N), a_max, replace=False)
        points = points[L]
        return points
    else:
        while True:
            points = up_sample_points_torch(points)
            if points.shape[0] >= a_max:
                break
    N = points.shape[0]
    L = np.random.choice(np.arange(N), a_max, replace=False)
    points = points[L]
    return points

def remove_outliers(points):
    pcd = visualize_point_cloud(points)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                             std_ratio=0.5)
    return np.array(cl.points),ind
