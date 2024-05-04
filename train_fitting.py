import logging
import os
import sys
from shutil import copyfile

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import numpy as np

from src.read_config_quadricsFitting import Config
from src.dataset_segments import Dataset
from src.dataset_segments import generator_iter
from src.loss import quadrics_reg_loss,quadrics_function_loss,quadrics_decomposition_loss,normals_deviation_loss,Taubin_distance_loss
from src.net_fitting import DGCNNQ_T
from src.utils import rescale_input_outputs
import torch.optim as optim
import torch.utils.data
from tensorboard_logger import Logger
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

np.set_printoptions(precision=4)

config = Config(sys.argv[1])
config.shape = sys.argv[2]
config.lqf_weight = float(sys.argv[3])
config.lqr_weight = float(sys.argv[4])
config.ldr_weight = float(sys.argv[5])
config.lds_weight = float(sys.argv[6])
config.ldt_weight = float(sys.argv[7])
config.lnd_weight = float(sys.argv[8])
config.r_regularization = float(sys.argv[9])
config.lr = float(sys.argv[10])

for data_file in os.listdir(config.dataset_dir):
    if config.shape in data_file:
        config.dataset_path = config.dataset_dir + data_file
        break

if config.test_on_another_dataset:
    for data_file in os.listdir(config.dataset_test_dir):
        if config.shape in data_file:
            config.test_dataset_path = config.dataset_test_dir + data_file
            break

model_name = config.model_path.format(
    config.mode,
    config.num_points,
    int(config.if_normals),
    config.lqf_weight,
    config.lqr_weight,
    config.ldr_weight,
    config.lds_weight,
    config.ldt_weight,
    config.lnd_weight,
    config.r_regularization,
    config.batch_size,
    config.lr,
    config.shape,
    config.more
)

print("Model name: ", model_name)

os.makedirs(
    "logs/tensorboard/train_fitting/{}/train".format(model_name),
    exist_ok=True,
)
os.makedirs(
    "logs/tensorboard/train_fitting/{}/val".format(model_name),
    exist_ok=True,
)
os.makedirs(
    "logs/logs".format(model_name),
    exist_ok=True,
)
logger_train = Logger("logs/tensorboard/train_fitting/{}/train".format(model_name), flush_secs=15)
logger_val = Logger("logs/tensorboard/train_fitting/{}/val".format(model_name), flush_secs=15)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(
    "logs/logs/{}.log".format(model_name), mode="w"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(handler)

os.makedirs(
    "logs/trained_models/{}/".format(model_name),
    exist_ok=True,
)
os.makedirs(
    "logs/scripts/{}/".format(model_name),
    exist_ok=True,
)
source_file = __file__
destination_file = "logs/scripts/{}/{}".format(
    model_name, __file__.split("/")[-1]
)
copyfile(source_file, destination_file)
# config备份
source_file_config = sys.argv[1]
destination_file_config = "logs/scripts/{}/{}".format(
    model_name, sys.argv[1].split("/")[-1]
)
copyfile(source_file_config, destination_file_config)

Q_decoder = DGCNNQ_T(num_Q=config.Q_size, num_points=10, config=config)

# use pretrained model
pretrain = False
print("pretrain: ", pretrain)
if pretrain:
    if torch.cuda.device_count() > 1:
        Q_decoder = torch.nn.DataParallel(Q_decoder)
        Q_decoder.load_state_dict(
        torch.load("logs/pretrained_models/quadrics_fitting/cylinder/" + "train_loss_multGPU.pth")
        )
    else:
        Q_decoder.load_state_dict(
        torch.load("logs/pretrained_models/quadrics_fitting/cylinder/" + "train_loss_singleGPU.pth")
        )
else:
    if torch.cuda.device_count() > 1:
        Q_decoder = torch.nn.DataParallel(Q_decoder)

Q_decoder.cuda()

if_augment = False

dataset = Dataset(config)
print('num_train: {}, num_val: {}'.format(dataset.train_points.shape[0],dataset.val_points.shape[0]))

get_train_data = dataset.load_train_data(d_mean=config.d_mean, d_scale=config.d_scale, d_rotation=config.d_rotation,if_augment=if_augment,shape=config.shape)

get_val_data = dataset.load_val_data(d_mean = config.d_mean, d_scale=config.d_scale, d_rotation=config.d_rotation, if_augment=if_augment,shape=config.shape)

loader = generator_iter(get_train_data, int(1e10))
get_train_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=16,
        pin_memory=True,
    )
)

loader = generator_iter(get_val_data, int(1e10))
get_val_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=16,
        pin_memory=True,
    )
)

# sss
optimizer = optim.Adam(Q_decoder.parameters(), lr=config.lr)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=3e-5
)

prev_loss_value_train = 1e8
prev_loss_value_val = 1e8
for e in range(config.epochs):

    train_loss = []
    train_quadrics_function = []
    train_quadrics_reg = []
    train_decomposition_r = []
    train_decomposition_s = []
    train_decomposition_t = []
    train_normals_deviation = []
    train_r_regularization = []

    Q_decoder.train()
    for train_b_id in range(dataset.train_points.shape[0] // config.batch_size):
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        points_, normals_,quadrics, T_batch,_ = next(get_train_data)[0]
        quadrics = torch.from_numpy(quadrics.astype(np.float32)).cuda()

        points = torch.from_numpy(points_.astype(np.float32)).cuda()
        points = points.permute(0, 2, 1)
        normals = torch.from_numpy(normals_.astype(np.float32)).cuda()
        normals = normals.permute(0, 2, 1)

        rand_num_points = config.num_points

        if config.if_normals:
            output,trans_inv,C = Q_decoder(torch.cat([points[:, :, 0:rand_num_points],normals[:, :, 0:rand_num_points]],1))
            
        else:
            output,trans_inv,C = Q_decoder(points[:, :, 0:rand_num_points])
            # loss_normals_deviation = 0

        loss_decomposition_r, loss_decomposition_s,loss_decomposition_t = quadrics_decomposition_loss(output, config, quadrics,trans_inv,C,mode="train"
        )

        loss_quadrics_function = quadrics_function_loss(
            output, points, config, quadrics
        )

        loss_quadrics_reg = quadrics_reg_loss(
            output, quadrics,config
        )

        loss_normals_deviation = normals_deviation_loss(output,points,normals,config,quadrics)

        trans_r = trans_inv[:,0:3,0:3]
        regularization_trans_r = torch.mean((torch.matmul(trans_r,trans_r.transpose(2,1)) - torch.eye(3).repeat(config.batch_size,1,1).cuda(trans_inv.device))**2)

        loss = loss_quadrics_function * config.lqf_weight + loss_quadrics_reg * config.lqr_weight + loss_decomposition_r * config.ldr_weight + loss_decomposition_s * config.lds_weight +loss_decomposition_t * config.ldt_weight+  loss_normals_deviation * config.lnd_weight + regularization_trans_r * config.r_regularization

        loss.backward()

        optimizer.step()

        train_loss.append(loss.data.cpu().numpy())
        train_quadrics_function.append(loss_quadrics_function.data.cpu().numpy())
        train_quadrics_reg.append(loss_quadrics_reg.data.cpu().numpy())
        train_decomposition_r.append(loss_decomposition_r.data.cpu().numpy())
        train_decomposition_s.append(loss_decomposition_s.data.cpu().numpy())
        train_decomposition_t.append(loss_decomposition_t.data.cpu().numpy())
        train_normals_deviation.append(loss_normals_deviation.data.cpu().numpy())
        train_r_regularization.append(regularization_trans_r.data.cpu().numpy())

        print(
            "\rEpoch: {} iter: {}, loss: {}, lqf {} | {}, lqr {} | {}, ldr {} | {}, lds {} | {}, ldt {} | {}, lnd {} | {}, reguRo {} | {}".format(
                e, train_b_id, loss.item(), loss_quadrics_function.item(), config.lqf_weight, loss_quadrics_reg.item(), config.lqr_weight, loss_decomposition_r.item(), config.ldr_weight, loss_decomposition_s.item(), config.lds_weight, loss_decomposition_t.item(), config.ldt_weight, loss_normals_deviation.item(), config.lnd_weight,regularization_trans_r,config.r_regularization
            ),
            end="",
        )

    del loss, loss_quadrics_function, loss_quadrics_reg, loss_decomposition_r, loss_decomposition_s,loss_decomposition_t,loss_normals_deviation,regularization_trans_r

    val_loss = []
    val_quadrics_function = []
    val_quadrics_reg = []
    val_decomposition_r = []
    val_decomposition_s = []
    val_decomposition_t = []
    val_normals_deviation = []
    val_r_regularization = []
    Q_decoder.eval()

    for val_b_id in range(dataset.val_points.shape[0] // config.batch_size):
        torch.cuda.empty_cache()
        points_, normals_,quadrics, T_batch,_ = next(get_val_data)[0]

        quadrics = torch.from_numpy(quadrics.astype(np.float32)).cuda()
        points = torch.from_numpy(points_.astype(np.float32)).cuda()
        points = points.permute(0, 2, 1)
        normals = torch.from_numpy(normals_.astype(np.float32)).cuda()
        normals = normals.permute(0, 2, 1)

        with torch.no_grad():
            if config.if_normals:
                output,trans_inv,C = Q_decoder(torch.cat([points[:, :, 0:rand_num_points],normals[:, :, 0:rand_num_points]],1))
            else:
                output,trans_inv,C = Q_decoder(points[:, :, 0:rand_num_points])

        loss_decomposition_r, loss_decomposition_s,loss_decomposition_t = quadrics_decomposition_loss(output, config, quadrics,trans_inv,C,mode="eval"
        )

        loss_quadrics_function = Taubin_distance_loss(
            output, points, config, quadrics
        )

        loss_quadrics_reg = quadrics_reg_loss(
            output, quadrics,config
        )
        
        loss_normals_deviation = normals_deviation_loss(output,points,normals,config,quadrics)

        trans_r = trans_inv[:,0:3,0:3]
        regularization_trans_r = torch.mean((torch.matmul(trans_r,trans_r.transpose(2,1)) - torch.eye(3).repeat(config.batch_size,1,1).cuda(trans_inv.device))**2)

        loss = loss_quadrics_function * config.lqf_weight + loss_quadrics_reg * config.lqr_weight + loss_decomposition_r * config.ldr_weight + loss_decomposition_s * config.lds_weight + loss_decomposition_t * config.ldt_weight + loss_normals_deviation * config.lnd_weight + regularization_trans_r * config.r_regularization

        val_loss.append(loss.data.cpu().numpy())
        val_quadrics_function.append(loss_quadrics_function.data.cpu().numpy())
        val_quadrics_reg.append(loss_quadrics_reg.data.cpu().numpy())
        val_decomposition_r.append(loss_decomposition_r.data.cpu().numpy())
        val_decomposition_s.append(loss_decomposition_s.data.cpu().numpy())
        val_decomposition_t.append(loss_decomposition_t.data.cpu().numpy())
        val_normals_deviation.append(loss_normals_deviation.data.cpu().numpy())
        val_r_regularization.append(regularization_trans_r.data.cpu().numpy())


    logger.info(
        "\nEpoch: {}/{} => Train: {}, Val: {}, Tr lqf: {}, Val lqf: {}, Tr lqr: {}, Val lqr: {}, Tr ldr: {}, Val ldr: {}, Tr lds: {}, Val lds: {}, Tr ldt: {}, Val ldt: {}, Tr lnd: {}, Val lnd: {}, Tr reguRo: {}, Val reguRo: {}".format(
            e,
            config.epochs,
            np.mean(train_loss),
            np.mean(val_loss),
            np.mean(train_quadrics_function),
            np.mean(val_quadrics_function),
            np.mean(train_quadrics_reg),
            np.mean(val_quadrics_reg),
            np.mean(train_decomposition_r),
            np.mean(val_decomposition_r),
            np.mean(train_decomposition_s),
            np.mean(val_decomposition_s),
            np.mean(train_decomposition_t),
            np.mean(val_decomposition_t),
            np.mean(train_normals_deviation),
            np.mean(val_normals_deviation),
            np.mean(train_r_regularization),
            np.mean(val_r_regularization)
        )
    )

    logger_train.log_value("loss", np.mean(train_loss), e)
    logger_val.log_value("loss", np.mean(val_loss), e)
    logger_train.log_value("loss_quadrics_function", np.mean(train_quadrics_function), e)
    logger_val.log_value("loss_quadrics_function", np.mean(val_quadrics_function), e)
    logger_train.log_value("loss_quadrics_reg", np.mean(train_quadrics_reg), e)
    logger_val.log_value("loss_quadrics_reg", np.mean(val_quadrics_reg), e)
    logger_train.log_value("loss_decomposition_rotation", np.mean(train_decomposition_r), e)
    logger_val.log_value("loss_decomposition_rotation", np.mean(val_decomposition_r), e)
    logger_train.log_value("loss_decomposition_scale", np.mean(train_decomposition_s), e)
    logger_val.log_value("loss_decomposition_scale", np.mean(val_decomposition_s), e)
    logger_train.log_value("loss_decomposition_translation", np.mean(train_decomposition_t), e)
    logger_val.log_value("loss_decomposition_translation", np.mean(val_decomposition_t), e)
    logger_train.log_value("loss_normals_deviation", np.mean(train_normals_deviation), e)
    logger_val.log_value("loss_normals_deviation", np.mean(val_normals_deviation), e)
    logger_train.log_value("regularization_trans_r", np.mean(train_r_regularization), e)
    logger_val.log_value("regularization_trans_r", np.mean(val_r_regularization), e)
    
    scheduler.step(np.mean(val_loss))

    # 1.
    monitor_loss = train_loss
    loss_name = "train_loss"
    if prev_loss_value_train > np.mean(monitor_loss):
        logger.info("{} improvement, saving model at epoch: {}".format(loss_name,e))
        prev_loss_value_train = np.mean(monitor_loss)
        torch.save(
            Q_decoder.module.state_dict(),
            "logs/trained_models/{}/{}_singleGPU.pth".format(model_name,loss_name),
        )

        torch.save(
            Q_decoder.state_dict(),
            "logs/trained_models/{}/{}_multGPU.pth".format(model_name,loss_name),
        )
    # 2.
    monitor_loss = val_loss
    loss_name = "val_loss"
    if prev_loss_value_val > np.mean(monitor_loss):
        logger.info("{} improvement, saving model at epoch: {}".format(loss_name,e))
        prev_loss_value_val = np.mean(monitor_loss)
        torch.save(
            Q_decoder.module.state_dict(),
            "logs/trained_models/{}/{}_singleGPU.pth".format(model_name,loss_name),
        )

        torch.save(
            Q_decoder.state_dict(),
            "logs/trained_models/{}/{}_multGPU.pth".format(model_name,loss_name),
        )
