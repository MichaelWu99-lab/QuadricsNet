"""
This script defines dataset loading for the quadrics segments.
"""

import h5py
import numpy as np
from torch.utils.data import Dataset
EPS = np.finfo(np.float32).eps


class generator_iter(Dataset):
    """This is a helper function to be used in the parallel data loading using Pytorch
    DataLoader class"""

    def __init__(self, generator, train_size):
        self.generator = generator
        self.train_size = train_size

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        return next(self.generator)


class DataSetControlPointsPoisson:
    def __init__(self, config, closed=False):
        """
        :param path: path to h5py file that stores the dataset
        :param batch_size: batch size
        :param num_points: number of 
        :param size_u:
        :param size_v:
        :param splits:
        """
        self.path = config.dataset_path
        self.batch_size = config.batch_size
        self.Q_size = config.Q_size

        with h5py.File(self.path, "r") as hf:
            points = np.array(hf.get(name="points")).astype(np.float32)
            quadrics = np.array(hf.get(name="quadrics")).astype(np.float32)
            normals = np.array(hf.get(name="normals")).astype(np.float32)

        np.random.seed(0)
        List = np.arange(points.shape[0])
        np.random.shuffle(List)
        points = points[List]
        quadrics = quadrics[List]
        normals = normals[List]

        test_on_another_dataset = config.test_on_another_dataset
        if test_on_another_dataset:
            print("test on another dataset")
            self.test_path = config.test_dataset_path

        if config.num_train and config.num_val and config.num_test:
            self.train_points = points[0 : config.num_train]
            self.train_quadrics = quadrics[0 : config.num_train]
            self.train_normals = normals[0 : config.num_train]
            if test_on_another_dataset:
                with h5py.File(self.test_path, "r") as hf:
                    points_another_dataset = np.array(hf.get(name="points")).astype(np.float32)
                    quadrics_another_dataset = np.array(hf.get(name="quadrics")).astype(np.float32)
                    normals_another_dataset = np.array(hf.get(name="normals")).astype(np.float32)
                    
                    self.val_points = points_another_dataset[0 : config.num_val]
                    self.val_quadrics = quadrics_another_dataset[0 : config.num_val]
                    self.val_normals = normals_another_dataset[0 : config.num_val]

                    self.test_points = points_another_dataset[0 : config.num_test]
                    self.test_quadrics = quadrics_another_dataset[0 : config.num_test]
                    self.test_normals = normals_another_dataset[0 : config.num_test]
            else:
                self.val_points = points[config.num_train : config.num_train + config.num_val]
                self.val_quadrics = quadrics[config.num_train : config.num_train + config.num_val]
                self.val_normals = normals[config.num_train : config.num_train + config.num_val]

                self.test_points = points[config.num_train + config.num_val : config.num_train + config.num_val + config.num_test]
                self.test_quadrics = quadrics[config.num_train + config.num_val : config.num_train + config.num_val + config.num_test]
                self.test_normals = normals[config.num_train + config.num_val : config.num_train + config.num_val + config.num_test]
        else:
            self.train_points = points[0 : int(config.rate_train * points.shape[0])]
            self.train_quadrics = quadrics[0 : int(config.rate_train * points.shape[0])]
            self.train_normals = normals[0 : int(config.rate_train * points.shape[0])]

            if test_on_another_dataset:
                with h5py.File(self.test_path, "r") as hf:
                    points_another_dataset = np.array(hf.get(name="points")).astype(np.float32)
                    quadrics_another_dataset = np.array(hf.get(name="quadrics")).astype(np.float32)
                    normals_another_dataset = np.array(hf.get(name="normals")).astype(np.float32)

                    num_test_dataset = points_another_dataset.shape[0]

                    self.val_points = points_another_dataset[0 : int(config.rate_val * num_test_dataset)]
                    self.val_quadrics = quadrics_another_dataset[0 : int(config.rate_val * num_test_dataset)]
                    self.val_normals = normals_another_dataset[0 : int(config.rate_val * num_test_dataset)]

                    self.test_points = points_another_dataset[0 : int(config.rate_test * num_test_dataset)]
                    self.test_quadrics = quadrics_another_dataset[0 : int(config.rate_test * num_test_dataset)]
                    self.test_normals = normals_another_dataset[0 : int(config.rate_test * num_test_dataset)]
            else:
                self.val_points = points[int(config.rate_train * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0])]
                self.val_quadrics = quadrics[int(config.rate_train * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0])]
                self.val_normals = normals[int(config.rate_train * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0])]

                self.test_points = points[int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) + int(config.rate_test * points.shape[0])]
                self.test_quadrics = quadrics[int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) + int(config.rate_test * points.shape[0])]    
                self.test_normals = normals[int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) + int(config.rate_test * points.shape[0])]


    def rotation_matrix_a_to_b(self, A, B):
        """
        Finds rotation matrix from vector A in 3d to vector B
        in 3d.
        B = R @ A
        """
        A = np.divide(A,np.linalg.norm(A))
        B = np.divide(B,np.linalg.norm(B))

        cos = np.dot(A, B)
        sin = np.linalg.norm(np.cross(B, A))
        u = A
        v = B - np.dot(A, B) * A
        v = v / (np.linalg.norm(v) + EPS)
        w = np.cross(B, A)
        w = w / (np.linalg.norm(w) + EPS)
        F = np.stack([u, v, w], 1)
        G = np.array([[cos, -sin, 0],
                      [sin, cos, 0],
                      [0, 0, 1]])

        # B = R @ A
        try:
            R = F @ G @ np.linalg.inv(F)
        except:
            R = np.eye(3, dtype=np.float32)
        return R

    def load_train_data(self, d_mean=True, d_scale=True, d_rotation=True,if_augment=False,shape='',noise=0):
        while True:
            for batch_id in range(self.train_points.shape[0] // self.batch_size):
                Points = []
                Quadrics = []
                T_batch = []
                Normals = []
                Scale_quadrics = []

                for i in range(self.batch_size):
                    points = self.train_points[batch_id * self.batch_size + i]
                    normals = self.train_normals[batch_id * self.batch_size + i]

                    if noise!=0:
                        # 在点云中加噪声
                        # np.random.randn用于生成服从标准正态分布（均值为0，标准差为1）的随机数
                        points_noise = normals * np.clip(np.random.randn(1, points.shape[0], 1) * noise, a_min=-noise, a_max=noise)
                        points = points + points_noise.astype(np.float32)

                    T = np.diag([1.0, 1.0, 1.0, 1.0])

                    if d_mean:
                        mean = np.mean(points, 0)
                        points = points - mean
                        T_d = np.diag([1.0, 1.0, 1.0, 1.0])
                        T_d[0:3, 3] = -mean
                        T = np.dot(T_d, T)

                    if d_rotation:
                        S, U = self.pca_numpy(points)
                        index_sorted = np.argsort(-S)
                        S_sorted = S[index_sorted]
                        U_sorted = U[:,index_sorted]

                        if shape == "plane":
                            smallest_ev = U_sorted[:,2]
                            R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 1,1]))
                        elif shape == "cone":
                            # a = self.test_quadrics[batch_id * self.batch_size + i]
                            # self.write_pcd(points, "data_save.pcd")
                            # 主轴，或者小特征值
                            axis_shape = U_sorted[:,self.pca_judgment_numpy(S_sorted,shape)]
                            smallest_ev = U[:, np.argsort(S)[0]]
                            R = self.rotation_matrix_a_to_b(axis_shape, np.array([1, 1,1]))
                        else:
                            # 主轴，或者最大特征值
                            axis_shape = U_sorted[:,self.pca_judgment_numpy(S_sorted,shape)]
                            R = self.rotation_matrix_a_to_b(axis_shape, np.array([1, 1,1]))

                        # if shape == "plane":
                        #     smallest_ev = U[:, np.argsort(S)[0]]
                        #     R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 1,1]))
                        # elif shape == "cone":
                        #     # a = self.test_quadrics[batch_id * self.batch_size + i]
                        #     # self.write_pcd(points, "data_save.pcd")
                        #     smallest_ev = U[:, np.argsort(S)[0]]
                        #     R = self.rotation_matrix_a_to_b(smallest_ev, np.array([0, 0,1]))
                        # else:
                        #     # 最大特征值
                        #     biggest_ev = U[:, np.argsort(S)[2]]
                        #     R = self.rotation_matrix_a_to_b(biggest_ev, np.array([0, 0,1]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points = (R @ points.T).T
                        T_r = np.diag([1.0, 1.0, 1.0, 1.0])
                        T_r[0:3,0:3] = R
                        T = np.dot(T_r, T)

                    if d_scale:
                        # 各个点到原点的最大欧式距离
                        std = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                        points = points / (std + EPS)
                        T_s = np.diag([1 / (std + EPS), 1 / (std + EPS), 1 / (std + EPS), 1.0])
                        T = np.dot(T_s, T)

                    # if d_scale:
                    #     std = np.abs(np.max(points, 0) - np.min(points, 0))
                    #     std = std.reshape((1, 3))
                    #     points = points / (std + EPS)
                    #     T_s = np.diag([1 / (std[0,0] + EPS), 1 / (std[0,1] + EPS), 1 / (std[0,2] + EPS), 1.0])
                    #     T = np.dot(T_s, T)
                        
                    Points.append(points)
                    normals = (np.linalg.inv(T[0:3,0:3]).T @ normals.T).T
                    # 单位化
                    normals = np.divide(normals,np.expand_dims(np.linalg.norm(normals,axis=1),axis=1))
                    # 确定法向量方向，之后网络knn计算法相量的距离是按照欧式距离
                    for index_normals in range(normals.shape[0]):
                        # if normals[index_normals][np.where(np.abs(normals[index_normals]) > 1e-8)].shape == 0:
                        # print(normals[index_normals],batch_id,i)
                        # normals[index_normals][np.where(np.abs(normals[index_normals]) > 1e-8)]
                        if normals[index_normals][np.where(np.abs(normals[index_normals]) > 1e-8)][0]<0:
                            normals[index_normals] = normals[index_normals] * -1


                    Normals.append(normals)
                    T_batch.append(T)
                    quadrics = self.train_quadrics[batch_id * self.batch_size + i]

                    if quadrics.shape == (10,1):
                        quadrics = quadrics.squeeze(1)
                    Q_temp = np.array([[quadrics[0], quadrics[3], quadrics[4], quadrics[6]],
                                    [quadrics[3], quadrics[1], quadrics[5], quadrics[7]],
                                    [quadrics[4], quadrics[5], quadrics[2], quadrics[8]],
                                    [quadrics[6], quadrics[7], quadrics[8], quadrics[9]]])

                    Q_temp = np.dot(np.dot(np.linalg.inv(T).T, Q_temp), np.linalg.inv(T))

                    # points_append = np.expand_dims(np.concatenate((points,np.ones((points.shape[0],1))),1),1)
                    # aa = np.mean(np.matmul(np.matmul(points_append,Q_temp) , points_append.transpose(0,2,1)))
                    # if aa > 1e-3:
                    #    print(aa)

                    # # eliminate scale arbitrary, quadrics归一化，此过程不影响点集
                    # if shape in ["plane","cone"]:
                    #     quadrics = np.array([[Q_temp[0, 0]], [Q_temp[1, 1]], [Q_temp[2, 2]], [Q_temp[0, 1]], [Q_temp[0, 2]], [Q_temp[1, 2]], [Q_temp[0, 3]], [Q_temp[1, 3]], [Q_temp[2, 3]],
                    #                     [Q_temp[3, 3]]])
                    #     scale_quadrics = np.linalg.norm(quadrics)
                    #     quadrics = quadrics / scale_quadrics        
                    # elif shape in ["ellipsoid","cylinder"]:
                    #     Q_temp,scale_quadrics = self.quadrics_scale_identification_numpy(Q_temp)
                    #     quadrics = np.array([[Q_temp[0, 0]], [Q_temp[1, 1]], [Q_temp[2, 2]], [Q_temp[0, 1]], [Q_temp[0, 2]], [Q_temp[1, 2]], [Q_temp[0, 3]], [Q_temp[1, 3]], [Q_temp[2, 3]],
                    #                     [Q_temp[3, 3]]])

                    quadrics = np.array([[Q_temp[0, 0]], [Q_temp[1, 1]], [Q_temp[2, 2]], [Q_temp[0, 1]], [Q_temp[0, 2]], [Q_temp[1, 2]], [Q_temp[0, 3]], [Q_temp[1, 3]], [Q_temp[2, 3]],
                                [Q_temp[3, 3]]])
                    
                    # 利用二范数归一化
                    scale_quadrics = np.linalg.norm(quadrics)
                    # # 利用特征值归一化
                    # scale_quadrics = np.linalg.norm(np.linalg.eig(Q_temp[0:3,0:3])[0])
                    
                    quadrics = quadrics / scale_quadrics     
                    Scale_quadrics.append(scale_quadrics)

                    Quadrics.append(quadrics)
                Quadrics = np.stack(Quadrics, 0)
                Points = np.stack(Points, 0)
                Normals = np.stack(Normals, 0)
                Scale_quadrics = np.stack(Scale_quadrics, 0)

                if if_augment:
                    Points = augment.augment(Points)
                    Points = Points.astype(np.float32)
                
                # 测试quadrics function是否接近于0，debug模式使用

                yield [Points,Normals,Quadrics, T_batch,Scale_quadrics]

    def load_val_data(self, d_mean=True, d_scale=True, d_rotation=True,if_augment=False,shape='',noise=0):
        while True:
            for batch_id in range(self.val_points.shape[0] // self.batch_size):
                Points = []
                Quadrics = []
                T_batch = []
                Normals = []
                Scale_quadrics = []

                for i in range(self.batch_size):
                    points = self.val_points[batch_id * self.batch_size + i]
                    normals = self.val_normals[batch_id * self.batch_size + i]

                    if noise!=0:
                        # 在点云中加噪声
                        # np.random.randn用于生成服从标准正态分布（均值为0，标准差为1）的随机数
                        points_noise = normals * np.clip(np.random.randn(1, points.shape[0], 1) * noise, a_min=-noise, a_max=noise)
                        points = points + points_noise.astype(np.float32)

                    T = np.diag([1.0, 1.0, 1.0, 1.0])

                    if d_mean:
                        mean = np.mean(points, 0)
                        points = points - mean
                        T_d = np.diag([1.0, 1.0, 1.0, 1.0])
                        T_d[0:3, 3] = -mean
                        T = np.dot(T_d, T)

                    if d_rotation:
                        S, U = self.pca_numpy(points)
                        index_sorted = np.argsort(-S)
                        S_sorted = S[index_sorted]
                        U_sorted = U[:,index_sorted]

                        if shape == "plane":
                            smallest_ev = U_sorted[:,2]
                            R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 1,1]))
                        elif shape == "cone":
                            # a = self.test_quadrics[batch_id * self.batch_size + i]
                            # self.write_pcd(points, "data_save.pcd")
                            # 主轴，或者小特征值
                            axis_shape = U_sorted[:,self.pca_judgment_numpy(S_sorted,shape)]
                            smallest_ev = U[:, np.argsort(S)[0]]
                            R = self.rotation_matrix_a_to_b(axis_shape, np.array([1, 1,1]))
                        else:
                            # 主轴，或者最大特征值
                            axis_shape = U_sorted[:,self.pca_judgment_numpy(S_sorted,shape)]
                            R = self.rotation_matrix_a_to_b(axis_shape, np.array([1, 1,1]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points = (R @ points.T).T
                        T_r = np.diag([1.0, 1.0, 1.0, 1.0])
                        T_r[0:3,0:3] = R
                        T = np.dot(T_r, T)

                    if d_scale:
                        # 各个点到原点的最大欧式距离
                        std = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                        points = points / (std + EPS)
                        T_s = np.diag([1 / (std + EPS), 1 / (std + EPS), 1 / (std + EPS), 1.0])
                        T = np.dot(T_s, T)

                    # if d_scale:
                    #     std = np.abs(np.max(points, 0) - np.min(points, 0))
                    #     std = std.reshape((1, 3))
                    #     points = points / (std + EPS)
                    #     T_s = np.diag([1 / (std[0,0] + EPS), 1 / (std[0,1] + EPS), 1 / (std[0,2] + EPS), 1.0])
                    #     T = np.dot(T_s, T)
                        
                    Points.append(points)
                    normals = (np.linalg.inv(T[0:3,0:3]).T @ normals.T).T
                    # 单位化
                    normals = np.divide(normals,np.expand_dims(np.linalg.norm(normals,axis=1),axis=1))
                    # 确定法向量方向，之后网络knn计算法相量的距离是按照欧式距离
                    for index_normals in range(normals.shape[0]):
                        if normals[index_normals][np.where(np.abs(normals[index_normals]) > 1e-8)][0]<0:
                            normals[index_normals] = normals[index_normals] * -1
                    Normals.append(normals)
                    T_batch.append(T)
                    quadrics = self.val_quadrics[batch_id * self.batch_size + i]

                    if quadrics.shape == (10,1):
                        quadrics = quadrics.squeeze(1)
                    Q_temp = np.array([[quadrics[0], quadrics[3], quadrics[4], quadrics[6]],
                                        [quadrics[3], quadrics[1], quadrics[5], quadrics[7]],
                                        [quadrics[4], quadrics[5], quadrics[2], quadrics[8]],
                                        [quadrics[6], quadrics[7], quadrics[8], quadrics[9]]])

                    Q_temp = np.dot(np.dot(np.linalg.inv(T).T, Q_temp), np.linalg.inv(T))

                    # # eliminate scale arbitrary, quadrics归一化，此过程不影响点集
                    # if shape in ["plane","cone"]:
                    #     quadrics = np.array([[Q_temp[0, 0]], [Q_temp[1, 1]], [Q_temp[2, 2]], [Q_temp[0, 1]], [Q_temp[0, 2]], [Q_temp[1, 2]], [Q_temp[0, 3]], [Q_temp[1, 3]], [Q_temp[2, 3]],
                    #                     [Q_temp[3, 3]]])
                    #     scale_quadrics = np.linalg.norm(quadrics)
                    #     quadrics = quadrics / scale_quadrics        
                    # elif shape in ["ellipsoid","cylinder"]:
                    #     Q_temp,scale_quadrics = self.quadrics_scale_identification_numpy(Q_temp)
                    #     quadrics = np.array([[Q_temp[0, 0]], [Q_temp[1, 1]], [Q_temp[2, 2]], [Q_temp[0, 1]], [Q_temp[0, 2]], [Q_temp[1, 2]], [Q_temp[0, 3]], [Q_temp[1, 3]], [Q_temp[2, 3]],
                    #                     [Q_temp[3, 3]]])
                  
                    quadrics = np.array([[Q_temp[0, 0]], [Q_temp[1, 1]], [Q_temp[2, 2]], [Q_temp[0, 1]], [Q_temp[0, 2]], [Q_temp[1, 2]], [Q_temp[0, 3]], [Q_temp[1, 3]], [Q_temp[2, 3]],
                                [Q_temp[3, 3]]])

                    # 利用二范数归一化
                    scale_quadrics = np.linalg.norm(quadrics)
                    # # 利用特征值归一化
                    # scale_quadrics = np.linalg.norm(np.linalg.eig(Q_temp[0:3,0:3])[0])

                    quadrics = quadrics / scale_quadrics      
                    Scale_quadrics.append(scale_quadrics)

                    Quadrics.append(quadrics)
                Quadrics = np.stack(Quadrics, 0)
                Points = np.stack(Points, 0)
                Normals = np.stack(Normals, 0)
                Scale_quadrics = np.stack(Scale_quadrics, 0)

                if if_augment:
                    Points = augment.augment(Points)
                    Points = Points.astype(np.float32)
                yield [Points,Normals, Quadrics, T_batch,Scale_quadrics]

    def load_test_data(self, d_mean=True, d_scale=True, d_rotation=True,if_augment=False,shape='',noise=0):
        for batch_id in range(self.test_points.shape[0] // self.batch_size):
            Points = []
            Points_ = []
            Quadrics = []
            Quadrics_ = []
            T_batch = []
            Normals = []
            Scale_quadrics = []

            for i in range(self.batch_size):
                points = self.test_points[batch_id * self.batch_size + i]
                normals = self.test_normals[batch_id * self.batch_size + i]

                if noise!=0:
                        # 在点云中加噪声
                        # np.random.randn用于生成服从标准正态分布（均值为0，标准差为1）的随机数
                        points_noise = normals * np.clip(np.random.randn(1, points.shape[0], 1) * noise, a_min=-noise, a_max=noise)
                        points = points + points_noise.astype(np.float32)
                        
                Points_.append(points)
                T = np.diag([1.0, 1.0, 1.0, 1.0])

                if d_mean:
                    mean = np.mean(points, 0)
                    points = points - mean
                    T_d = np.diag([1.0, 1.0, 1.0, 1.0])
                    T_d[0:3, 3] = -mean
                    T = np.dot(T_d, T)

                if d_rotation:
                    S, U = self.pca_numpy(points)
                    index_sorted = np.argsort(-S)
                    S_sorted = S[index_sorted]
                    U_sorted = U[:,index_sorted]

                    if shape == "plane":
                        smallest_ev = U_sorted[:,2]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 1,1]))
                    elif shape == "cone":
                        # a = self.test_quadrics[batch_id * self.batch_size + i]
                        # self.write_pcd(points, "data_save.pcd")
                        # 主轴，或者小特征值
                        axis_shape = U_sorted[:,self.pca_judgment_numpy(S_sorted,shape)]
                        smallest_ev = U[:, np.argsort(S)[0]]
                        R = self.rotation_matrix_a_to_b(axis_shape, np.array([1, 1,1]))
                    else:
                        # 主轴，或者最大特征值
                        axis_shape = U_sorted[:,self.pca_judgment_numpy(S_sorted,shape)]
                        R = self.rotation_matrix_a_to_b(axis_shape, np.array([1, 1,1]))
                    # rotate input points such that the minor principal
                    # axis aligns with x axis.
                    points = (R @ points.T).T
                    T_r = np.diag([1.0, 1.0, 1.0, 1.0])
                    T_r[0:3,0:3] = R
                    T = np.dot(T_r, T)

                if d_scale:
                    # 各个点到原点的最大欧式距离
                    std = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                    points = points / (std + EPS)
                    T_s = np.diag([1 / (std + EPS), 1 / (std + EPS), 1 / (std + EPS), 1.0])
                    T = np.dot(T_s, T)

                # if d_scale:
                #     std = np.abs(np.max(points, 0) - np.min(points, 0))
                #     std = std.reshape((1, 3))
                #     points = points / (std + EPS)
                #     T_s = np.diag([1 / (std[0,0] + EPS), 1 / (std[0,1] + EPS), 1 / (std[0,2] + EPS), 1.0])
                #     T = np.dot(T_s, T)
                    
                Points.append(points)
                
                normals = (np.linalg.inv(T[0:3,0:3]).T @ normals.T).T
                # normals = (T[0:3,0:3] @ normals.T).T

                # 单位化
                normals = np.divide(normals,np.expand_dims(np.linalg.norm(normals,axis=1),axis=1))
                # 确定法向量方向，之后网络knn计算法相量的距离是按照欧式距离
                for index_normals in range(normals.shape[0]):
                    if normals[index_normals][np.where(np.abs(normals[index_normals]) > 1e-8)][0]<0:
                        normals[index_normals] = normals[index_normals] * -1
                Normals.append(normals)
                T_batch.append(T)
                quadrics = self.test_quadrics[batch_id * self.batch_size + i]
                Quadrics_.append(quadrics)

                if quadrics.shape == (10,1):
                    quadrics = quadrics.squeeze(1)
                Q_temp = np.array([[quadrics[0], quadrics[3], quadrics[4], quadrics[6]],
                                   [quadrics[3], quadrics[1], quadrics[5], quadrics[7]],
                                   [quadrics[4], quadrics[5], quadrics[2], quadrics[8]],
                                   [quadrics[6], quadrics[7], quadrics[8], quadrics[9]]])

                Q_temp = np.dot(np.dot(np.linalg.inv(T).T, Q_temp), np.linalg.inv(T))

                # if shape in ["plane","cone"]:
                #     quadrics = np.array([[Q_temp[0, 0]], [Q_temp[1, 1]], [Q_temp[2, 2]], [Q_temp[0, 1]], [Q_temp[0, 2]], [Q_temp[1, 2]], [Q_temp[0, 3]], [Q_temp[1, 3]], [Q_temp[2, 3]],
                #                     [Q_temp[3, 3]]])
                #     scale_quadrics = np.linalg.norm(quadrics)
                #     quadrics = quadrics / scale_quadrics        
                # elif shape in ["ellipsoid","cylinder"]:
                #     Q_temp,scale_quadrics = self.quadrics_scale_identification_numpy(Q_temp)
                #     quadrics = np.array([[Q_temp[0, 0]], [Q_temp[1, 1]], [Q_temp[2, 2]], [Q_temp[0, 1]], [Q_temp[0, 2]], [Q_temp[1, 2]], [Q_temp[0, 3]], [Q_temp[1, 3]], [Q_temp[2, 3]],
                #                     [Q_temp[3, 3]]])

                quadrics = np.array([[Q_temp[0, 0]], [Q_temp[1, 1]], [Q_temp[2, 2]], [Q_temp[0, 1]], [Q_temp[0, 2]], [Q_temp[1, 2]], [Q_temp[0, 3]], [Q_temp[1, 3]], [Q_temp[2, 3]],
                                [Q_temp[3, 3]]])

                # 利用二范数归一化
                scale_quadrics = np.linalg.norm(quadrics)
                # # 利用特征值归一化
                # scale_quadrics = np.linalg.norm(np.linalg.eig(Q_temp[0:3,0:3])[0])
                
                quadrics = quadrics / scale_quadrics        
                Scale_quadrics.append(scale_quadrics)

                Quadrics.append(quadrics)

            Quadrics = np.stack(Quadrics, 0)
            Quadrics_ = np.stack(Quadrics_, 0)
            Points = np.stack(Points, 0)
            Points_ = np.stack(Points_, 0)
            Normals = np.stack(Normals, 0)
            Scale_quadrics = np.stack(Scale_quadrics, 0)

            if if_augment:
                Points = augment.augment(Points)
                Points = Points.astype(np.float32)
            yield [Points,Points_,Normals, Quadrics,Quadrics_,T_batch,Scale_quadrics]

    def pca_torch(self, X):
        covariance = torch.transpose(X, 1, 0) @ X
        S, U = torch.eig(covariance, eigenvectors=True)
        return S, U

    def pca_numpy(self, X):
        S, U = np.linalg.eig(X.T @ X)
        return S, U

    def quadrics_scale_identification_numpy(self,Q):
        eigenvalue_Q,_ = np.linalg.eig(Q)

        # 筛选出特征值绝对占比大于1%的特征值
        eigenvalue_Q_sum = np.sum(np.abs(eigenvalue_Q))
        eigenvalue_Q = eigenvalue_Q[np.where(np.abs(eigenvalue_Q) > (eigenvalue_Q_sum * 0.001))]

        scale_Q = np.array([1])
        for i in eigenvalue_Q:
            scale_Q = scale_Q * i

        eigenvalue_E,_ = np.linalg.eig(Q[0:3,0:3])

        # 筛选出特征值绝对占比大于5%的特征值
        eigenvalue_E_sum = np.sum(np.abs(eigenvalue_E))
        eigenvalue_E = eigenvalue_E[np.where(np.abs(eigenvalue_E) > (eigenvalue_E_sum * 0.001))]
        scale_E = np.array([1])

        for i in eigenvalue_E:
            scale_E = scale_E * i
        
        scale_identification = np.abs(scale_E / scale_Q)
        Q = scale_identification * Q
        
        # 这里要返回scale_identification的倒数，因为之后的反归一化时，是乘法
        return Q,np.squeeze(1/scale_identification)

    def write_pcd(self,points, save_pcd_path):
        HEADER = '''\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii
        '''
        with open(save_pcd_path, 'w') as f:
            f.write(HEADER.format(len(points), len(points)) + '\n')
            np.savetxt(f, points, delimiter = ' ', fmt = '%f %f %f')

    def pca_judgment_numpy(self,S,shape):

        x = S[1] / S[0]
        y = S[2] / S[0]

        margin = 0.1
        # 寻找主轴，即特征值不同于其他两个轴
        if abs(x-1)<margin:
            shape_axis_index = 2
        elif abs(y-1)<margin:
            shape_axis_index = 1
        elif abs(x-y)<margin:
            shape_axis_index = 0
        else:
            if shape == "cone":
                # cone
                shape_axis_index = 2
            else:
                # cylinder and sphere
                shape_axis_index = 0
        return shape_axis_index