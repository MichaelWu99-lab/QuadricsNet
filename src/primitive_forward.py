from cProfile import label
from calendar import c
import numpy as np
import torch
from torch import linalg as LA


from src.fitting_utils import remove_outliers, up_sample_all_in_range
EPS = np.finfo(np.float32).eps
torch.manual_seed(2)
np.random.seed(2)


def initialize_ellipsoid_model(config):
    from src.net_fitting import DGCNNQ_T

    config.shape = "ellipsoid"
    config.Q_size = 10
    
    decoder = DGCNNQ_T(num_Q=config.Q_size, num_points=10, config=config)

    if torch.cuda.device_count() > 1:
        alt_gpu = 1
    else:
        alt_gpu = 0
    path = config.ellipsoid_path + "_singleGPU.pth"

    decoder.cuda(alt_gpu)

    decoder.load_state_dict(torch.load(path))

    decoder.eval()
    return decoder

def initialize_sphere_model(config):
    from src.net_fitting import DGCNNQ_T

    config.shape = "sphere"
    config.Q_size = 10
    config.if_normals = config.if_fitting_normals[0]
    
    decoder = DGCNNQ_T(num_Q=config.Q_size, num_points=10, config=config)

    if torch.cuda.device_count() > 1:
        alt_gpu = 1
    else:
        alt_gpu = 0
    path = config.sphere_path + "_singleGPU.pth"

    decoder.cuda(alt_gpu)

    decoder.load_state_dict(torch.load(path))

    decoder.eval()
    return decoder

def initialize_plane_model(config):
    from src.net_fitting import DGCNNQ_T

    config.shape = "plane"
    config.Q_size = 10
    config.if_normals = config.if_fitting_normals[1]
    
    decoder = DGCNNQ_T(num_Q=config.Q_size, num_points=10, config=config)

    if torch.cuda.device_count() > 1:
        alt_gpu = 1
    else:
        alt_gpu = 0
    path = config.plane_path + "_singleGPU.pth"

    decoder.cuda(alt_gpu)

    decoder.load_state_dict(torch.load(path))

    decoder.eval()
    return decoder

def initialize_cylinder_model(config):
    from src.net_fitting import DGCNNQ_T

    config.shape = "cylinder"
    config.Q_size = 10
    config.if_normals = config.if_fitting_normals[2]

    decoder = DGCNNQ_T(num_Q=config.Q_size, num_points=10, config=config)

    if torch.cuda.device_count() > 1:
        alt_gpu = 1
    else:
        alt_gpu = 0
    path = config.cylinder_path + "_singleGPU.pth"

    decoder.cuda(alt_gpu)

    decoder.load_state_dict(torch.load(path))

    decoder.eval()
    return decoder

def initialize_cone_model(config):
    from src.net_fitting import DGCNNQ_T

    config.shape = "cone"
    config.Q_size = 10
    config.if_normals = config.if_fitting_normals[3]

    decoder = DGCNNQ_T(num_Q=config.Q_size, num_points=10, config=config)

    if torch.cuda.device_count() > 1:
        alt_gpu = 1
    else:
        alt_gpu = 0
    path = config.cone_path + "_singleGPU.pth"

    decoder.cuda(alt_gpu)

    decoder.load_state_dict(torch.load(path))

    decoder.eval()
    return decoder

def fit_one_shape_torch(data, fitter, weights, eval=False,if_fitting_normals=[0,0,0,0]):
    """
    Fits primitives
    """
    clustered_points = {}

    clustered_points_input = {}
    clustered_normals_input = {}

    clustered_Ts = {}
    clustered_primitives = {}
    clustered_primitives_gt = {}
    clustered_labels_gt = {}

    fitter.parameters = {}

    for _, d in enumerate(data):
        points, normals,_, primitives, segment_indices, part_index, primitives_gt,label_gt= d

        part_index, label_index = part_index

        clustered_points[label_index] = points

        if not eval:

            weight = weights

            if points.shape[0] < 200:

                clustered_points_input[label_index] = None
                clustered_normals_input[label_index] = None

                fitter.parameters[label_index] = None
                clustered_Ts[label_index] = None
                clustered_primitives[label_index] = None
                clustered_primitives_gt[label_index] = None
                clustered_labels_gt[label_index] = None
                continue

            points_input = points
            normals_input = normals
        else:
            weight = weights

            if points.shape[0] < 200:

                clustered_points_input[label_index] = None
                clustered_normals_input[label_index] = None

                fitter.parameters[label_index] = None
                clustered_Ts[label_index] = None
                clustered_primitives[label_index] = None
                clustered_primitives_gt[label_index] = None
                clustered_labels_gt[label_index] = None
                continue

            points_input = points
            normals_input = normals

            if primitives == 2:
                _,_,_,points_input_cropped,normals_input_copped = estimate_cylinder_properties_torch(points_input,normals_input)
                if points_input_cropped.shape[0] >= 100:
                    points_input = points_input_cropped
                    normals_input = normals_input_copped

            _,index = remove_outliers(points_input.data.cpu().numpy())
            points_input = points_input[index]
            normals_input = normals_input[index]

            num_points_input = 1100
            points_input, normals_input = up_sample_all_in_range(points_input, normals_input, num_points_input)

        points_mean = points_input.mean(dim=0)
        points_input = points_input - points_mean

        # {"Sphere":0,"Plane":1,"Cylinder":2,"Cone":3}
        S, U = pca_torch(points_input)
        S_sorted, index_sorted = torch.sort(S,descending=True)
        U_sorted = U[:, index_sorted]
        
        if primitives == 1:
            smallest_ev = U_sorted[:,2]
            points_rotation = rotation_matrix_a_to_b_torch(smallest_ev, torch.tensor([1, 1, 1]).float().cuda(smallest_ev.device))
        elif primitives == 3:
            axis_shape = U_sorted[:,pca_judgment_torch(S_sorted,primitives)]
            points_rotation = rotation_matrix_a_to_b_torch(axis_shape, torch.tensor([1, 1, 1]).float().cuda(axis_shape.device))
        else:
            axis_shape = U_sorted[:,pca_judgment_torch(S_sorted,primitives)]
            points_rotation = rotation_matrix_a_to_b_torch(axis_shape, torch.tensor([1, 1, 1]).float().cuda(axis_shape.device))

        points_input = (points_rotation @ points_input.T).T
                        
        points_std = torch.max(torch.sqrt(torch.sum(points_input ** 2, axis = 1)))
        points_input = points_input / (points_std + EPS)

        T = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0])).cuda(points_input.device)
        T_d = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0])).cuda(points_input.device)
        T_d[0:3, 3] = -points_mean
        T = torch.matmul(T_d, T)
        T_r = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0])).cuda(points_input.device)
        T_r[0:3,0:3] = points_rotation
        T = torch.matmul(T_r, T)
        T_s = torch.diag(torch.tensor([1 / (points_std + EPS), 1 / (points_std + EPS), 1 / (points_std + EPS), 1.0])).cuda(points_input.device)            
        T = torch.matmul(T_s, T)

        normals_input = torch.matmul(torch.inverse(T[0:3,0:3]).T , normals_input.T).T
        normals_input = torch.divide(normals_input,torch.unsqueeze(torch.norm(normals_input,dim=1),axis=1))
        for index_normals in range(normals_input.shape[0]):
            if normals_input[index_normals][0] < 0:
                normals_input[index_normals] = normals_input[index_normals] * (-1)
            
        # sphere
        if primitives == 0:
            _,_,_ = fitter.forward_pass_sphere(points_input,normals_input, ids=label_index, weights = weight,if_fitting_normals=if_fitting_normals[0])
        # plane
        if primitives == 1:
            _,_,_ = fitter.forward_pass_plane(points_input,normals_input,  ids=label_index, weights = weight,if_fitting_normals=if_fitting_normals[1])
        # cylinder
        if primitives == 2:
            _,_,_ = fitter.forward_pass_cylinder(points_input,normals_input,  ids=label_index, weights = weight,if_fitting_normals=if_fitting_normals[2])
        # cone
        if primitives == 3:
            _,_,_ = fitter.forward_pass_cone(points_input,normals_input,  ids=label_index, weights = weight,if_fitting_normals=if_fitting_normals[3])

        clustered_points_input[label_index] = points_input
        clustered_normals_input[label_index] = normals_input
        clustered_Ts[label_index] = T
        clustered_primitives[label_index] = primitives
        clustered_primitives_gt[label_index] = primitives_gt
        clustered_labels_gt[label_index] = label_gt

    return clustered_points,clustered_points_input,clustered_normals_input,clustered_Ts,clustered_primitives,clustered_primitives_gt,clustered_labels_gt

def forward_sphere(input_points_,input_normals_, decoder, weights=None,if_fitting_normals=0):

    points = input_points_.permute(0, 2, 1)
    normals = input_normals_.permute(0, 2, 1)
    if if_fitting_normals:
        output,trans_inv,C = decoder(torch.cat([points,normals],1),weights)
    else:
        output,trans_inv,C = decoder(points,weights)
    return output,trans_inv,C

def forward_plane(input_points_,input_normals_, decoder, weights=None,if_fitting_normals=0):

    points = input_points_.permute(0, 2, 1)
    normals = input_normals_.permute(0, 2, 1)
    if if_fitting_normals:
        output,trans_inv,C = decoder(torch.cat([points,normals],1),weights)
    else:
        output,trans_inv,C = decoder(points,weights)
    return output,trans_inv,C

def forward_cylinder(input_points_,input_normals_, decoder,weights=None,if_fitting_normals=0):

    points = input_points_.permute(0, 2, 1)
    normals = input_normals_.permute(0, 2, 1)
    if if_fitting_normals:
        output,trans_inv,C = decoder(torch.cat([points,normals],1),weights)
    else:
        output,trans_inv,C = decoder(points,weights)
    return output,trans_inv,C

def forward_cone(input_points_,input_normals_, decoder,weights=None,if_fitting_normals=0):

    points = input_points_.permute(0, 2, 1)
    normals = input_normals_.permute(0, 2, 1)
    if if_fitting_normals:
        output,trans_inv,C = decoder(torch.cat([points,normals],1),weights)
    else:
        output,trans_inv,C = decoder(points,weights)
    return output,trans_inv,C

def pca_torch(X):
    covariance = torch.transpose(X, 1, 0) @ X
    S, U = torch.eig(covariance, eigenvectors=True)
    return S[:,0], U

def rotation_matrix_a_to_b_torch(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    A = torch.divide(A,torch.norm(A))
    B = torch.divide(B,torch.norm(B))
    
    cos = torch.dot(A, B)
    sin = torch.norm(torch.cross(B, A))
    u = A
    v = B - torch.dot(A, B) * A
    v = v / (torch.norm(v) + EPS)
    w = torch.cross(B, A)
    w = w / (torch.norm(w) + EPS)
    F = torch.stack([u, v, w], 1)
    G = torch.tensor([[cos, -sin, 0],
                      [sin, cos, 0],
                      [0, 0, 1]]).cuda(cos.device)

    # B = R @ A
    try:
        R = F @ G @ torch.inverse(F)
    except:
        R = torch.eye(3, dtype=torch.float32).cuda(cos.device)
    return R

def pca_judgment_torch(S,primitives):

    x = S[1] / S[0]
    y = S[2] / S[0]

    margin = 0.1
    if abs(x-1)<margin:
        shape_axis_index = 2
    elif abs(y-1)<margin:
        shape_axis_index = 1
    elif abs(x-y)<margin:
        shape_axis_index = 0
    else:
        if primitives == 3:
            # cone
            shape_axis_index = 2
        else:
            # cylinder and sphere
            shape_axis_index = 0
    return shape_axis_index

def estimate_cylinder_properties_torch(points,normals,k=4):
    points_mean = torch.mean(points, dim=0)
    points_centered = points - points_mean

    # PCA
    u, s, v = torch.pca_lowrank(points_centered, q=3)
    axis_direction = v[:, 0]

    projected_points = torch.matmul(points_centered, axis_direction)
    height = torch.max(projected_points) - torch.min(projected_points)

    distances = LA.norm(points_centered - torch.ger(projected_points, axis_direction), dim=1)
    radius = torch.mean(distances)

    max_height = k * radius
    if height > max_height:
        height_limit = max_height
        valid_indices = (projected_points >= torch.min(projected_points) + (height - height_limit) / 2) & \
                        (projected_points <= torch.max(projected_points) - (height - height_limit) / 2)
        points_cropped = points[valid_indices]
        normals_cropped = normals[valid_indices]
    else:
        points_cropped = points
        normals_cropped = normals

    return axis_direction, height, radius, points_cropped, normals_cropped
