from turtle import shape
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


EPS = np.finfo(np.float32).eps


def knn(x, k):
    batch_size = x.shape[0]
    indices = np.arange(0, k)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        idx = distances.topk(k=k, dim=-1)[1][:, :, indices]
    return idx

def knn_points_normals(x, k):

    batch_size = x.shape[0]
    indices = np.arange(0, k)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            p = x[b: b + 1, 0:3]
            n = x[b: b + 1, 3:6]

            inner = 2 * torch.matmul(p.transpose(2, 1), p)
            xx = torch.sum(p ** 2, dim=1, keepdim=True)
            p_pairwise_distance = xx - inner + xx.transpose(2, 1)

            inner = 2 * torch.matmul(n.transpose(2, 1), n)
            n_pairwise_distance = 2 - inner

            # This pays less attention to normals
            pairwise_distance = p_pairwise_distance * (1 + n_pairwise_distance)

            # This pays more attention to normals
            # pairwise_distance = p_pairwise_distance * torch.exp(n_pairwise_distance)

            # pays too much attention to normals
            # pairwise_distance = p_pairwise_distance + n_pairwise_distance

            distances.append(-pairwise_distance)

        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.contiguous()
    x = x.view(batch_size, -1, num_points).contiguous()
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx_base = idx_base.cuda(x.device)
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature

def get_graph_feature_with_normals(x, k=20, idx=None):

    batch_size = x.size(0)
    num_points = x.size(2)
    x=x.contiguous()
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn_points_normals(x, k=k)

    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx_base = idx_base.cuda(x.device)

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature

class DGCNNQ_T(nn.Module):
    def __init__(self, num_Q=10, num_points=40,config=""):

        super(DGCNNQ_T, self).__init__()
        self.k = num_points
        self.mode = config.mode
        self.if_normals = config.if_normals
        self.trans = TRANS()
        if self.mode == 0:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm1d(1024)
            self.drop = 0.0
            if self.if_normals:
                self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.LeakyReLU(negative_slope=0.2))
            else:
                self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                        self.bn1,
                        nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.num_Q = num_Q
            self.conv6 = torch.nn.Conv1d(1024, 1024, 1)
            self.conv7 = torch.nn.Conv1d(1024, 1024, 1)

            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(1024, 1, 1)

            self.bn6 = nn.BatchNorm1d(1024)
            self.bn7 = nn.BatchNorm1d(1024)

        if self.mode == 1:
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm1d(1024)
            self.drop = 0.0

            if self.if_normals:
                self.conv1 = nn.Sequential(nn.Conv2d(12, 128, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.LeakyReLU(negative_slope=0.2))
            else:
                self.conv1 = nn.Sequential(nn.Conv2d(6, 128, kernel_size=1, bias=False),
                        self.bn1,
                        nn.LeakyReLU(negative_slope=0.2))

            self.conv2 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv3 = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv4 = nn.Sequential(nn.Conv2d(256 * 2, 512, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv5 = nn.Sequential(nn.Conv1d(1024 + 128, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.num_Q = num_Q
            self.conv6 = torch.nn.Conv1d(1024, 1024, 1)
            self.conv7 = torch.nn.Conv1d(1024, 1024, 1)

            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(1024, 1, 1)
            self.bn6 = nn.BatchNorm1d(1024)
            self.bn7 = nn.BatchNorm1d(1024)

        self.shape = SHAPE(config)

    def forward(self, x, weights=None):
        """
        :param weights: weights of size B x N
        """
        batch_size = x.size(0)

        if self.if_normals:
            x = get_graph_feature_with_normals(x, k=self.k)
        else:
            x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x1 = torch.unsqueeze(x1, 2)

        x = F.dropout(F.relu(self.bn6(self.conv6(x1))), self.drop)
        x = F.dropout(F.relu(self.bn7(self.conv7(x))), self.drop)

        trans_inv = self.trans(x)

        C = self.shape(x)
        
        Q = torch.bmm(torch.bmm(trans_inv.transpose(2,1), C),trans_inv)

        q = torch.stack((Q[:,0,0],Q[:,1,1],Q[:,2,2],Q[:,0,1],Q[:,0,2],Q[:,1, 2],Q[:,0,3],Q[:,1,3],Q[:,2,3],Q[:,3,3])).permute(1,0)

        # return q,trans_inv,C
        return q,trans_inv,C

class TRANS(nn.Module):
    def __init__(self, T_size=12):
        super(TRANS, self).__init__()
        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, T_size)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.T_size = T_size

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,0,1,0,0,0,0,1,0]).flatten().astype(np.float32)))\
            .view(1, self.T_size).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda(x.device)
        x = x + iden

        append = Variable(torch.from_numpy(np.array([0,0,0,1]).flatten().astype(np.float32)))\
            .view(1, 4).repeat(batch_size, 1)
        if x.is_cuda:
            append = append.cuda(x.device)

        x = torch.cat([x,append],1)

        x = x.view(-1, 4, 4)
        return x

class SHAPE(nn.Module):
    def __init__(self,config):
        super(SHAPE, self).__init__()

        if "plane" in config.shape:
            self.num_scale = 1
            self.shape = "plane"
        elif "ellipsoid" in config.shape:
            self.num_scale = 4
            self.shape = "ellipsoid"
        elif "sphere" in config.shape:
            self.num_scale = 2
            self.shape = "sphere"
        elif "elliptic_cylinder" in config.shape:
            self.num_scale = 3
            self.shape = "elliptic_cylinder"
        elif "cylinder" in config.shape:
            self.num_scale = 2
            self.shape = "cylinder"
        elif "elliptic_cone" in config.shape:
            self.num_scale = 3
            self.shape = "elliptic_cone"
        elif "cone" in config.shape:
            self.num_scale = 2
            self.shape = "cone"

        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_scale)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.last = nn.LeakyReLU(negative_slope=1)

    def forward(self, x):
        batch_size = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.last(self.fc3(x))
        
        if self.shape == "plane":
            x_00 = x ** 2

            x_00 = x_00[:, 0].unsqueeze(1)

            append_1 = Variable(torch.from_numpy(np.array([0,0,0]).flatten().astype(np.float32)))\
                .view(1, 3).repeat(batch_size, 1)

            if x.is_cuda:
                append_1 = append_1.cuda(x.device)

            x = torch.cat([x_00,append_1],1)
        elif self.shape == "ellipsoid":
            x_00_11_22_33 = x ** 2
    
            x_00 = x_00_11_22_33[:, 0].unsqueeze(1)
            x_11 = x_00_11_22_33[:, 1].unsqueeze(1)
            x_22 = x_00_11_22_33[:, 2].unsqueeze(1)
            x_33 = x_00_11_22_33[:, 3].unsqueeze(1)
    
            x = torch.cat([x_00,x_11],1)
            x = torch.cat([x,x_22],1)
            x = torch.cat([x,-x_33],1)
        elif self.shape == "sphere":
            x_00_33 = x ** 2
    
            x_00 = x_00_33[:, 0].unsqueeze(1)
            x_33 = x_00_33[:, 1].unsqueeze(1)

            x = torch.cat([x_00,x_00],1)
            x = torch.cat([x,x_00],1)
            x = torch.cat([x,-x_33],1)
        elif self.shape == "elliptic_cylinder":
            x_00_11_33 = x ** 2
            
            x_00 = x_00_11_33[:, 0].unsqueeze(1)
            x_11 = x_00_11_33[:, 1].unsqueeze(1)
            x_33 = x_00_11_33[:, 2].unsqueeze(1)
            
            append_1 = Variable(torch.from_numpy(np.array([0]).flatten().astype(np.float32)))\
                .view(1, 1).repeat(batch_size, 1)
            if x.is_cuda:
                append_1 = append_1.cuda(x.device)
    
            x = torch.cat([x_00,x_11],1)
            x = torch.cat([x,append_1],1)
            x = torch.cat([x,-x_33],1)
        elif self.shape == "cylinder":
            x_00_33 = x ** 2
            
            x_00 = x_00_33[:, 0].unsqueeze(1)
            x_33 = x_00_33[:, 1].unsqueeze(1)
            
            append_1 = Variable(torch.from_numpy(np.array([0]).flatten().astype(np.float32)))\
                .view(1, 1).repeat(batch_size, 1)
            if x.is_cuda:
                append_1 = append_1.cuda(x.device)
    
            x = torch.cat([x_00,x_00],1)
            x = torch.cat([x,append_1],1)
            x = torch.cat([x,-x_33],1)
        elif self.shape == "elliptic_cone":
            x_00_11_22 = x ** 2
    
            x_00 = x_00_11_22[:, 0].unsqueeze(1)
            x_11 = x_00_11_22[:, 1].unsqueeze(1)
            x_22 = x_00_11_22[:, 2].unsqueeze(1)
            
            append_1 = Variable(torch.from_numpy(np.array([0]).flatten().astype(np.float32)))\
                .view(1, 1).repeat(batch_size, 1)
            if x.is_cuda:
                append_1 = append_1.cuda(x.device)
    
            x = torch.cat([x_00,x_11],1)
            x = torch.cat([x,-x_22],1)
            x = torch.cat([x,append_1],1)
        elif self.shape == "cone":
            x_00_22 = x ** 2
    
            x_00 = x_00_22[:, 0].unsqueeze(1)
            x_22 = x_00_22[:, 1].unsqueeze(1)
            
            append_1 = Variable(torch.from_numpy(np.array([0]).flatten().astype(np.float32)))\
                .view(1, 1).repeat(batch_size, 1)
            if x.is_cuda:
                append_1 = append_1.cuda(x.device)
    
            x = torch.cat([x_00,x_00],1)
            x = torch.cat([x,-x_22],1)
            x = torch.cat([x,append_1],1)

        x = torch.diag_embed(x)

        return x