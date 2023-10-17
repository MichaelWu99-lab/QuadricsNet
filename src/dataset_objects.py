"""
This script defines dataset loading for the quadrics objects.
"""

from random import shuffle
import h5py
import numpy as np

EPS = np.finfo(np.float32).eps

class Dataset:
    def __init__(self,
                 config,shuffle=True):

        self.path = config.dataset_path
        self.batch_size = config.batch_size

        with h5py.File(self.path, "r") as hf:
            points = np.array(hf.get("points")).astype(np.float32)
            quadrics = np.array(hf.get("quadrics")).astype(np.float32)
            labels = np.array(hf.get("labels")).astype(np.int8)
            primitives = np.array(hf.get("prims")).astype(np.int8)
            normals = np.array(hf.get(name="normals")).astype(np.float32)

        if shuffle:
            np.random.seed(0)
            List = np.arange(points.shape[0])
            np.random.shuffle(List)
            points = points[List]
            quadrics = quadrics[List]
            labels = labels[List]
            primitives = primitives[List]
            normals = normals[List]
        else:
            List = np.arange(points.shape[0])

        if config.num_train and config.num_val and config.config.test:
            
            self.train_points = points[0 : config.num_train]
            self.val_points = points[config.num_train : config.num_train + config.num_val]
            self.test_points = points[config.num_train + config.num_val : config.num_train + config.num_val + config.num_test]
            
            self.train_quadrics = quadrics[0 : config.num_train]
            self.val_quadrics = quadrics[config.num_train : config.num_train + config.num_val]
            self.test_quadrics = quadrics[config.num_train + config.num_val : config.num_train + config.num_val + config.num_test]
            
            self.train_labels = labels[0 : config.num_train]
            self.val_labels = labels[config.num_train : config.num_train + config.num_val]
            self.test_labels = labels[config.num_train + config.num_val : config.num_train + config.num_val + config.num_test]
            
            self.train_primitives = primitives[0 : config.num_train]
            self.val_primitives = primitives[config.num_train : config.num_train + config.num_val]
            self.test_primitives = primitives[config.num_train + config.num_val : config.num_train + config.num_val + config.num_test]
            
            self.train_normals = normals[0 : config.num_train]
            self.val_normals = normals[config.num_train : config.num_train + config.num_val]
            self.test_normals = normals[config.num_train + config.num_val : config.num_train + config.num_val + config.num_test]

            # Save index for test
            self.test_list = List[config.num_train + config.num_val : config.num_train + config.num_val + config.num_test]
        else:
            self.train_points = points[0 : int(config.rate_train * points.shape[0])]
            self.val_points = points[int(config.rate_train * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0])]
            self.test_points = points[int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) + int(config.rate_test * points.shape[0])]
            
            self.train_quadrics = quadrics[0 : int(config.rate_train * points.shape[0])]
            self.val_quadrics = quadrics[int(config.rate_train * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0])]
            self.test_quadrics = quadrics[int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) + int(config.rate_test * points.shape[0])]
            
            self.train_labels = labels[0 : int(config.rate_train * points.shape[0])]
            self.val_labels = labels[int(config.rate_train * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0])]
            self.test_labels = labels[int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) + int(config.rate_test * points.shape[0])]
            
            self.train_primitives = primitives[0 : int(config.rate_train * points.shape[0])]
            self.val_primitives = primitives[int(config.rate_train * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0])]
            self.test_primitives = primitives[int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) + int(config.rate_test * points.shape[0])]

            self.train_normals = normals[0 : int(config.rate_train * points.shape[0])]
            self.val_normals = normals[int(config.rate_train * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0])]
            self.test_normals = normals[int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) + int(config.rate_test * points.shape[0])]

            # Save index for test
            self.test_list = List[int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) : int(config.rate_train * points.shape[0]) + int(config.rate_val * points.shape[0]) + int(config.rate_test * points.shape[0])]


    def get_train(self, d_mean=False, d_scale=False,noise=0):
        while True:
            for batch_id in range(self.train_points.shape[0] // self.batch_size):
                Points = []
                Quadrics = []
                T_batch = []
                Normals = []
                # Scale_quadrics = []

                for i in range(self.batch_size):
                    points = self.train_points[batch_id * self.batch_size + i]
                    normals = self.train_normals[batch_id * self.batch_size + i]
                    
                    if noise!=0:
                        points_noise = normals * np.clip(np.random.randn(1, points.shape[0], 1) * noise, a_min=-noise, a_max=noise)
                        points = points + points_noise.astype(np.float32)
                    
                    T = np.diag([1.0, 1.0, 1.0, 1.0])

                    if d_mean:
                        mean = np.mean(points, 0)
                        points = points - mean
                        T_d = np.diag([1.0, 1.0, 1.0, 1.0])
                        T_d[0:3, 3] = -mean
                        T = np.dot(T_d, T)

                    if d_scale:
                        std = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                        points = points / (std + EPS)
                        T_s = np.diag([1 / (std + EPS), 1 / (std + EPS), 1 / (std + EPS), 1.0])
                        T = np.dot(T_s, T)

                    T_batch.append(T)
                    Points.append(points)
                    normals = (np.linalg.inv(T[0:3,0:3]).T @ normals.T).T
                    normals = np.divide(normals,np.expand_dims(np.linalg.norm(normals,axis=1),axis=1))
                    for index_normals in range(normals.shape[0]):
                        if normals[index_normals][np.where(np.abs(normals[index_normals]) > 1e-8)][0]<0:
                            normals[index_normals] = normals[index_normals] * -1                   
                    Normals.append(normals)

                    quadrics = self.train_quadrics[batch_id * self.batch_size + i]

                    Q_temp = np.array([[quadrics[:,0], quadrics[:,3], quadrics[:,4], quadrics[:,6]],
                               [quadrics[:,3], quadrics[:,1], quadrics[:,5], quadrics[:,7]],
                               [quadrics[:,4], quadrics[:,5], quadrics[:,2], quadrics[:,8]],
                               [quadrics[:,6], quadrics[:,7], quadrics[:,8], quadrics[:,9]]]).transpose(2,0,1)

                    T_inv = np.expand_dims(np.linalg.inv(T),0)

                    Q_temp = np.matmul(np.matmul(T_inv.transpose(0,2,1),Q_temp),T_inv)

                    quadrics = np.array([[Q_temp[:,0,0]],[Q_temp[:,1,1]],[Q_temp[:,2,2]],[Q_temp[:,0,1]],[Q_temp[:,0,2]]
                                            , [Q_temp[:,1,2]],[Q_temp[:,0,3]],[Q_temp[:,1,3]],[Q_temp[:,2,3]],[Q_temp[:,3,3]]]).transpose(2,0,1).squeeze(2)

                    Quadrics.append(quadrics)

                Labels = self.train_labels[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]

                Primitives = self.train_primitives[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]

                Quadrics = np.stack(Quadrics, 0)
                Points = np.stack(Points, 0)
                Normals = np.stack(Normals, 0)

                yield [Points, Normals,Quadrics, T_batch,Labels, Primitives]

    def get_val(self, d_mean=False, d_scale=False,noise=0):
        while True:
            for batch_id in range(self.val_points.shape[0] // self.batch_size):
                Points = []
                Quadrics = []
                T_batch = []
                Normals = []
                # Scale_quadrics = []

                for i in range(self.batch_size):
                    points = self.val_points[batch_id * self.batch_size + i]
                    normals = self.val_normals[batch_id * self.batch_size + i]

                    if noise!=0:
                        points_noise = normals * np.clip(np.random.randn(1, points.shape[0], 1) * noise, a_min=-noise, a_max=noise)
                        points = points + points_noise.astype(np.float32)
                    
                    T = np.diag([1.0, 1.0, 1.0, 1.0])

                    if d_mean:
                        mean = np.mean(points, 0)
                        points = points - mean
                        T_d = np.diag([1.0, 1.0, 1.0, 1.0])
                        T_d[0:3, 3] = -mean
                        T = np.dot(T_d, T)

                    if d_scale:
                        std = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                        points = points / (std + EPS)
                        T_s = np.diag([1 / (std + EPS), 1 / (std + EPS), 1 / (std + EPS), 1.0])
                        T = np.dot(T_s, T)

                    T_batch.append(T)
                    Points.append(points)
                    normals = (np.linalg.inv(T[0:3,0:3]).T @ normals.T).T
                    normals = np.divide(normals,np.expand_dims(np.linalg.norm(normals,axis=1),axis=1))
                    for index_normals in range(normals.shape[0]):
                        if normals[index_normals][np.where(np.abs(normals[index_normals]) > 1e-8)][0]<0:
                            normals[index_normals] = normals[index_normals] * -1                   
                    Normals.append(normals)

                    quadrics = self.val_quadrics[batch_id * self.batch_size + i]

                    Q_temp = np.array([[quadrics[:,0], quadrics[:,3], quadrics[:,4], quadrics[:,6]],
                               [quadrics[:,3], quadrics[:,1], quadrics[:,5], quadrics[:,7]],
                               [quadrics[:,4], quadrics[:,5], quadrics[:,2], quadrics[:,8]],
                               [quadrics[:,6], quadrics[:,7], quadrics[:,8], quadrics[:,9]]]).transpose(2,0,1)

                    T_inv = np.expand_dims(np.linalg.inv(T),0)

                    Q_temp = np.matmul(np.matmul(T_inv.transpose(0,2,1),Q_temp),T_inv)

                    quadrics = np.array([[Q_temp[:,0,0]],[Q_temp[:,1,1]],[Q_temp[:,2,2]],[Q_temp[:,0,1]],[Q_temp[:,0,2]]
                                            , [Q_temp[:,1,2]],[Q_temp[:,0,3]],[Q_temp[:,1,3]],[Q_temp[:,2,3]],[Q_temp[:,3,3]]]).transpose(2,0,1).squeeze(2)

                    Quadrics.append(quadrics)

                Labels = self.val_labels[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]

                Primitives = self.val_primitives[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]

                Quadrics = np.stack(Quadrics, 0)
                Points = np.stack(Points, 0)
                Normals = np.stack(Normals, 0)

                yield [Points, Normals,Quadrics, T_batch,Labels, Primitives]

    def get_test(self, d_mean=False, d_scale=False,noise=0):
        while True:
            for batch_id in range(self.test_points.shape[0] // self.batch_size):
                Points = []
                Points_ = []
                Quadrics = []
                Quadrics_ = []
                T_batch = []
                Normals = []

                for i in range(self.batch_size):
                    points = self.test_points[batch_id * self.batch_size + i]
                    normals = self.test_normals[batch_id * self.batch_size + i]

                    if noise!=0:
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

                    if d_scale:
                        std = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                        points = points / (std + EPS)
                        T_s = np.diag([1 / (std + EPS), 1 / (std + EPS), 1 / (std + EPS), 1.0])
                        T = np.dot(T_s, T)

                    T_batch.append(T)
                    Points.append(points)
                    normals = (np.linalg.inv(T[0:3,0:3]).T @ normals.T).T
                    normals = np.divide(normals,np.expand_dims(np.linalg.norm(normals,axis=1),axis=1))
                    for index_normals in range(normals.shape[0]):
                        if normals[index_normals][np.where(np.abs(normals[index_normals]) > 1e-8)][0]<0:
                            normals[index_normals] = normals[index_normals] * -1
                    Normals.append(normals)
                                 
                    quadrics = self.test_quadrics[batch_id * self.batch_size + i]
                    Quadrics_.append(quadrics)

                    Q_temp = np.array([[quadrics[:,0], quadrics[:,3], quadrics[:,4], quadrics[:,6]],
                               [quadrics[:,3], quadrics[:,1], quadrics[:,5], quadrics[:,7]],
                               [quadrics[:,4], quadrics[:,5], quadrics[:,2], quadrics[:,8]],
                               [quadrics[:,6], quadrics[:,7], quadrics[:,8], quadrics[:,9]]]).transpose(2,0,1)

                    T_inv = np.expand_dims(np.linalg.inv(T),0)

                    Q_temp = np.matmul(np.matmul(T_inv.transpose(0,2,1),Q_temp),T_inv)

                    quadrics = np.array([[Q_temp[:,0,0]],[Q_temp[:,1,1]],[Q_temp[:,2,2]],[Q_temp[:,0,1]],[Q_temp[:,0,2]]
                                            , [Q_temp[:,1,2]],[Q_temp[:,0,3]],[Q_temp[:,1,3]],[Q_temp[:,2,3]],[Q_temp[:,3,3]]]).transpose(2,0,1).squeeze(2)

                    Quadrics.append(quadrics)

                Labels = self.test_labels[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]

                Primitives = self.test_primitives[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]

                Quadrics = np.stack(Quadrics, 0)
                Quadrics_ = np.stack(Quadrics_, 0)
                Points = np.stack(Points, 0)
                Points_ = np.stack(Points_, 0)
                Normals = np.stack(Normals, 0)
                
                test_data_index = self.test_list[batch_id * self.batch_size]

                yield [Points,Points_, Normals,Quadrics,Quadrics_, T_batch,Labels, Primitives,test_data_index]

    def normalize_points(self, points, normals, anisotropic=False):
        points = points - np.mean(points, 0, keepdims=True)
        noise = normals * np.clip(np.random.randn(points.shape[0], 1) * 0.01, a_min=-0.01, a_max=0.01)
        points = points + noise.astype(np.float32)

        S, U = self.pca_numpy(points)
        smallest_ev = U[:, np.argmin(S)]
        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
        # rotate input points such that the minor principal
        # axis aligns with x axis.
        points = (R @ points.T).T
        normals = (R @ normals.T).T
        std = np.max(points, 0) - np.min(points, 0)
        if anisotropic:
            points = points / (std.reshape((1, 3)) + EPS)
        else:
            points = points / (np.max(std) + EPS)
        return points.astype(np.float32), normals.astype(np.float32)

    def rotation_matrix_a_to_b(self, A, B):
        """
        Finds rotation matrix from vector A in 3d to vector B
        in 3d.
        B = R @ A
        """
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

    def pca_numpy(self, X):
        S, U = np.linalg.eig(X.T @ X)
        return S, U

    def quadrics_scale_identification_numpy(self,Q):
        eigenvalue_Q,_ = np.linalg.eig(Q)

        eigenvalue_Q_sum = np.sum(np.abs(eigenvalue_Q))
        eigenvalue_Q = eigenvalue_Q[np.where(np.abs(eigenvalue_Q) > (eigenvalue_Q_sum * 0.01))]

        scale_Q = np.array([1])
        for i in eigenvalue_Q:
            scale_Q = scale_Q * i

        eigenvalue_E,_ = np.linalg.eig(Q[0:3,0:3])

        eigenvalue_E_sum = np.sum(np.abs(eigenvalue_E))
        eigenvalue_E = eigenvalue_E[np.where(np.abs(eigenvalue_E) > (eigenvalue_E_sum * 0.05))]
        scale_E = np.array([1])

        for i in eigenvalue_E:
            scale_E = scale_E * i
        
        scale_identification = np.abs(scale_E / scale_Q)
        Q = scale_identification * Q
        
        return Q,1/scale_identification