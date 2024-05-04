import imp
import re
import numpy as np
from sklearn.decomposition import PCA
from skimage import measure
import sys
import open3d as o3d
from scipy.spatial import KDTree


try:
    import quadrics2points
except:
    print("Please add MATLAB runtime path to the environment variable!")
    sys.exit()
import os

class utils_vis:
    def __init__(self,DOWN_SAMPLE_NUM=40000):
        import numpy as np
        self.DOWN_SAMPLE_NUM = DOWN_SAMPLE_NUM

    def bound_box(self,points):
        x_max = np.max(points[:, 0])
        x_min = np.min(points[:, 0])
        y_max = np.max(points[:, 1])
        y_min = np.min(points[:, 1])
        z_max = np.max(points[:, 2])
        z_min = np.min(points[:, 2])
        bound_box_size = np.array([[x_min, x_max],
                                [y_min, y_max],
                                [z_min, z_max]])
        return bound_box_size

    def quadrics2Q(self,q):
        Q = np.array([[q[0], q[3], q[4],q[6]],
                    [q[3], q[1], q[5], q[7]],
                    [q[4], q[5], q[2], q[8]],
                    [q[6], q[7], q[8], q[9]]])
        return Q

    def Q2quadrics(self,Q):
        q = np.array([Q[0, 0],Q[1, 1],Q[2, 2],Q[0, 1],Q[0, 2],Q[1, 2],Q[0, 3],Q[1, 3],Q[2, 3],Q[3, 3]])
        return q

    def quadrics2points_py(self,q, mesh_size, res, error):
        xlow = mesh_size[0, 0]
        xhigh = mesh_size[0, 1]
        ylow = mesh_size[1, 0]
        yhigh = mesh_size[1, 1]
        zlow = mesh_size[2, 0]
        zhigh = mesh_size[2, 1]
        x = np.arange(xlow, xhigh + res[0], res[0])
        y = np.arange(ylow, yhigh + res[1], res[1])
        z = np.arange(zlow, zhigh + res[2], res[2])
        X, Y, Z = np.meshgrid(x, y, z)
        I = np.ones(X.shape)
        
        F = (q[0] * X**2 + q[1] * Y**2 + q[2] * Z**2 + \
            q[3] * 2 * X * Y + q[4] * 2 * X * Z + q[5] * 2 * Y * Z + \
            q[6] * 2 * X + q[7] * 2 * Y + q[8] * 2 * Z + q[9] * I <= error)
        points_reconstruction,_,_,_ = measure.marching_cubes_lewiner(F)
        points_reconstruction = points_reconstruction * res+np.array([xlow,ylow,zlow])

        return points_reconstruction

    def quadrics2points_by_matlab(self,q, mesh_size, res, error):
        # generate points_reconstruction from quadrics by matlab function
        try:
            np.savetxt('temp.txt', np.hstack((q,mesh_size.reshape(-1),res,error)))
            quadrics2points_matlab = quadrics2points.initialize()
            points_reconstruction = np.array(quadrics2points_matlab.quadrics2points('temp.txt')['vertices'])
            quadrics2points_matlab.terminate()
            
            if os.path.exists('temp.txt'):
                os.remove('temp.txt')

            return points_reconstruction
        except:
            print("Installation error in the MATLAB environment!")
            sys.exit()

    def plane_trim(self,points_gt, q_pre, mesh_size, res, error,shape_index,epsilon=0.1,if_points_trim="1"):
        Q_pre = self.quadrics2Q(q_pre)
        
        # Determine the plane normal vector
        A_pre = Q_pre[:3, :3]
        value_pre ,vector_pre = np.linalg.eig(A_pre)
        idx_pre = np.argsort(value_pre)[::-1]
        vector_pre = vector_pre[:, idx_pre[0]]
        
        q_pre = self.Q2quadrics(Q_pre)

        points_reconstruction = self.quadrics2points_py(q_pre, mesh_size, res, error)

        # points_reconstruction_temp = np.expand_dims(points_reconstruction,1)
        # points_gt_temp = np.expand_dims(points_gt,0)
        points_reconstruction_temp = self.down_sample(points_reconstruction,self.DOWN_SAMPLE_NUM)
        points_reconstruction_temp_ex = np.expand_dims(points_reconstruction_temp,1)
        points_gt_temp = self.down_sample(points_gt,self.DOWN_SAMPLE_NUM)
        points_gt_temp_ex = np.expand_dims(points_gt_temp,0)
        diff = points_reconstruction_temp_ex - points_gt_temp_ex
        diff = np.sum(diff ** 2,2)
        idx = np.argmin(np.min(diff,1))
        point_nearest = points_reconstruction_temp[idx]

        t = (vector_pre[0]*point_nearest[0] + vector_pre[1]*point_nearest[1] + vector_pre[2]*point_nearest[2]) - \
    (vector_pre[0]*points_gt[:,0] + vector_pre[1]*points_gt[:,1] + vector_pre[2]*points_gt[:,2])

        points_trim = np.column_stack((points_gt[:,0] + vector_pre[0]*t,
                               points_gt[:,1] + vector_pre[1]*t,
                               points_gt[:,2] + vector_pre[2]*t))

        if if_points_trim == "1":
            # trim: nearest and within a certain distance
            points_trim_temp = self.down_sample(points_trim,self.DOWN_SAMPLE_NUM)
            points_trim_temp_ex = np.expand_dims(points_trim_temp,1)
            points_gt_temp = self.down_sample(points_gt,self.DOWN_SAMPLE_NUM)
            points_gt_temp_ex = np.expand_dims(points_gt_temp,0)
            diff = points_trim_temp_ex - points_gt_temp_ex
            diff = np.sum(diff ** 2,2)

            trim_index = np.argmin(diff,0)
            nearest_diff = np.min(diff,0)
            points_trim = points_trim_temp[trim_index[nearest_diff<(np.square(epsilon))]]
            # print("Index:",shape_index,"prmi:",1,"res_max:",np.sqrt(nearest_diff.max()),"num_trimmed",points_trim_temp.shape[0],"-",points_trim.shape[0])

        return points_trim

    def others_trim(self,q_gt, points_gt, q_pre, projection, mesh_size, res, error, margin, if_aixs_trim, primitives,shape_index,epsilon=0.1,if_points_trim="1"):
        Q_gt = self.quadrics2Q(q_gt)
        Q_pre = self.quadrics2Q(q_pre)

        A_gt = np.array([[Q_gt[0, 0], Q_gt[0, 1], Q_gt[0, 2]],
                        [Q_gt[1, 0], Q_gt[1, 1], Q_gt[1, 2]],
                        [Q_gt[2, 0], Q_gt[2, 1], Q_gt[2, 2]]])
        value_gt, vector_gt = np.linalg.eig(A_gt)
        idx_gt = np.argsort(value_gt)[::-1]
        vector_gt = vector_gt[:, idx_gt]
        value_gt = value_gt[idx_gt]

        A_pre = np.array([[Q_pre[0, 0], Q_pre[0, 1], Q_pre[0, 2]],
                        [Q_pre[1, 0], Q_pre[1, 1], Q_pre[1, 2]],
                        [Q_pre[2, 0], Q_pre[2, 1], Q_pre[2, 2]]])
        value_pre, vector_pre = np.linalg.eig(A_pre)
        idx_pre = np.argsort(value_pre)[::-1]
        vector_pre = vector_pre[:, idx_pre]
        value_pre = value_pre[idx_pre]

        _, Ir_gt, _ = self.judgment(value_gt)
        _, Ir_pre, _ = self.judgment(value_pre)

        if projection == "0":
            axis_projection = vector_gt @ np.diag(Ir_pre)
        elif projection == "1":
            axis_projection = vector_pre @ np.diag(Ir_gt)

        # If the axis cannot be found from the Quadrics coefficient, then find it from the points
        if np.sum(axis_projection) == 0:
            pca = PCA(n_components=3)
            pca.fit(points_gt)
            U_sorted = pca.components_
            S_sorted = pca.singular_values_
            if primitives == 1:
                # plane
                axis_projection_temp = U_sorted[:, 2]
            else:
                axis_projection_temp = U_sorted[:, self.judgment_pca(S_sorted, primitives)]
            axis_projection[:, 0] = axis_projection_temp

        max_projection_gt = np.max(points_gt @ axis_projection, axis=0)
        min_projection_gt = np.min(points_gt @ axis_projection, axis=0)

        q_pre = self.Q2quadrics(Q_pre)

        points_reconstruction = self.quadrics2points_by_matlab(q_pre, mesh_size, res, error)

        if if_aixs_trim == "0":
            points_trim = points_reconstruction
        else:
            margin_value = np.abs(max_projection_gt) * margin
            points_trim = self.trim(points_reconstruction, max_projection_gt + margin_value, min_projection_gt - margin_value, axis_projection)

        if if_points_trim == "1":
            # trim: nearest and within a certain distance
            points_trim_temp = self.down_sample(points_trim,self.DOWN_SAMPLE_NUM)
            points_trim_temp_ex = np.expand_dims(points_trim_temp,1)
            points_gt_temp = self.down_sample(points_gt,self.DOWN_SAMPLE_NUM)
            points_gt_temp_ex = np.expand_dims(points_gt_temp,0)
            diff = points_trim_temp_ex - points_gt_temp_ex
            diff = np.sum(diff ** 2,2)

            trim_index = np.argmin(diff,0)
            nearest_diff = np.min(diff,0)
            points_trim = points_trim_temp[trim_index[nearest_diff<(np.square(epsilon))]]
            # print("Index:",shape_index,"prmi:",primitives,"res_max:",np.sqrt(nearest_diff.max()),"num_trimmed",points_trim_temp.shape[0],"-",points_trim.shape[0])

        return points_trim

    def find_nearest_within_epsilon(self,points_trim_temp, points_gt_temp, epsilon):

        # 使用KD树查找最近点
        tree = KDTree(points_gt_temp)
        distances, _ = tree.query(points_trim_temp)

        # 筛选出距离小于 epsilon 的点
        within_epsilon = distances < epsilon
        return points_trim_temp[within_epsilon]

    def judgment(self,d):
        d = np.sort(d)[::-1]

        x = d[1] / d[0]
        y = d[2] / d[0]

        # translation degeneration
        It = np.abs(d) > 1e-3

        # scale degeneration
        margin = 1e-2
        Is = It.copy()
        # in case of plane [1 0 0 0]
        if np.abs(x) < margin and np.abs(y) < margin:
            Is = np.array([0, 0, 0])
        # in case of cylinder [1 1 0 -1]
        if x > margin and np.abs(y) < margin:
            Is = np.array([1, 1, 0])
        # in case of cone [1 1 -1 0]
        if x > margin and y < -margin:
            Is = np.array([1, 1, 0])

        # rotation degeneration
        Ir = np.ones(3)
        if np.abs(x - 1) < margin:
            Ir[1] = 0
            Ir[0] = 0
        if np.abs(x - y) < margin:
            Ir[1] = 0
            Ir[2] = 0

        return Is, Ir, It

    def judgment_pca(self,d, primitives):
        d = np.sort(d)[::-1]

        x = d[1] / d[0]
        y = d[2] / d[0]

        margin = 0.1
        if np.abs(x - 1) < margin:
            shape_axis_index = 2
        elif np.abs(y - 1) < margin:
            shape_axis_index = 1
        elif np.abs(x - y) < margin:
            shape_axis_index = 0
        else:
            if primitives == "cone":
                # cone
                shape_axis_index = 2
            else:
                # cylinder and sphere
                shape_axis_index = 0

        return shape_axis_index

    def trim(self,points_pre, max_projection, min_projection, vector):
        points_trim = []
        for index in range(points_pre.shape[0]):
            if (np.all(points_pre[index] @ vector <= max_projection) and
                np.all(points_pre[index] @ vector >= min_projection)):
                points_trim.append(points_pre[index])
            else:
                continue
        points_trim = np.array(points_trim)
        return points_trim
    
    def down_sample(self,points, num_sample=40000):
        # To adapt to memory size, downsampling point clouds can lead to a decrease in res
        if points.shape[0] > num_sample:
            choice = np.random.choice(points.shape[0], num_sample, replace=False)
            points_sample = points[choice, :]
        else:
            points_sample = points
        return points_sample

    def res_efficient(self,points_reconstruction,points_gt,down_sample_num=40000):

        if isinstance(points_reconstruction, o3d.geometry.PointCloud):
            points_reconstruction = np.asarray(points_reconstruction.points)
        if isinstance(points_gt, o3d.geometry.PointCloud):
            points_gt = np.asarray(points_gt.points)

        points_reconstruction_temp = self.down_sample(points_reconstruction,down_sample_num)
        points_gt_temp = self.down_sample(points_gt,down_sample_num)

        points_reconstruction_temp = np.expand_dims(points_reconstruction_temp,1)
        points_gt_temp = np.expand_dims(points_gt_temp,0)
        # diff: [num_reconstruction_points, num_gt_points, 3]
        diff = points_reconstruction_temp - points_gt_temp
        # diff: [num_reconstruction_points, num_gt_points]
        diff = np.sqrt(np.sum(diff ** 2,2))

        # distance_0 = np.mean(np.min(diff,1))
        distance_1_all = np.min(diff,0)
        p_cover_1 = np.mean(distance_1_all<0.01)
        p_cover_2 = np.mean(distance_1_all<0.02)
        distance_1 = np.mean(distance_1_all)
        # res = np.mean([distance_0,distance_1])
        res = distance_1
        return res,p_cover_1,p_cover_2
    
    def res_knn(self,points_reconstruction,points_gt):

        # 当points_reconstruction是open3d的点云时
        if isinstance(points_reconstruction, o3d.geometry.PointCloud):
            pcd1 = points_reconstruction
            pcd2 = points_gt
        else:
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(points_reconstruction)
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(points_gt)

        # 将点云转换为numpy数组
        points1 = np.asarray(pcd1.points)

        # 为第二个点云创建一个KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd2)

        # 计算第一个点云中每个点到第二个点云中最近点的距离
        distances = []
        for point in points1:
            _, _, dist = kdtree.search_knn_vector_3d(point, 1)
            distances.append(np.sqrt(dist[0]))

        res = np.mean(distances)
        p_cover_1 = np.mean(np.array(distances)<0.01)
        p_cover_2 = np.mean(np.array(distances)<0.02)
        return res,p_cover_1,p_cover_2

    def reconstruction_quadrics(self,shape,if_axis_trim,if_points_trim,points_input_scaled,quadrics_pre_scaled,quadrics_gt_scaled,DOWN_SAMPLE_NUM):
        resolution = 1*1e-2
        if "plane" in shape:
            # plane
            mesh_size = self.bound_box(points_input_scaled)+0.1*np.array([[-1,1],[-1,1],[-1,1]])
            error = 1e-3
            res = resolution*(mesh_size[:,1]- mesh_size[:,0])
        elif "sphere" in shape or "ellipsoid" in shape:
            # sphere
            mesh_size = self.bound_box(points_input_scaled)+1*np.array([[-1,1],[-1,1],[-1,1]])
            res = resolution*(mesh_size[:,1]- mesh_size[:,0])
            error = 1e-3
            margin_pre = [10,10,10]
        else:
            mesh_size = self.bound_box(points_input_scaled)+1*np.array([[-1,1],[-1,1],[-1,1]])
            res = resolution*(mesh_size[:,1]- mesh_size[:,0])
            error = 1e-3
            margin_pre = [10,10,0]

        if "plane" in shape:
            # continue
            points_reconstruction_shape_temp = self.plane_trim(points_input_scaled,quadrics_pre_scaled,mesh_size,res,error,0,if_points_trim=if_points_trim)
        else:
            if "cone" in shape:
                primitive_shape_pre = 3
            elif "cylinder" in shape:
                primitive_shape_pre = 2
            elif "sphere" in shape or "ellipsoid" in shape:
                primitive_shape_pre = 0
            points_reconstruction_shape_temp = self.others_trim(quadrics_gt_scaled,points_input_scaled,quadrics_pre_scaled,"1",mesh_size,res,error,margin_pre,if_axis_trim,primitive_shape_pre,0,if_points_trim=if_points_trim)
            
        res,p_cover_1,p_cover_2 = self.res_efficient(points_reconstruction_shape_temp,points_input_scaled,DOWN_SAMPLE_NUM)

        return points_reconstruction_shape_temp,res,p_cover_1,p_cover_2
