import gc
import json
import logging
import os
import sys
import traceback
from shutil import copyfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np

from src.read_config_feature import Config
from src.net_dection import PrimitivesEmbeddingDGCNGn
from src.dataset_segments import generator_iter
from src.dataset_objects import Dataset
from src.residual_utils import Evaluation
from src.loss import (
    EmbeddingLoss,
    primitive_loss,
)
from src.utils import grad_norm
import torch.optim as optim
import torch.utils.data
from tensorboard_logger import Logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


np.set_printoptions(precision=3)

config = Config(sys.argv[1])

model_name = config.model_path.format(
    config.mode,
    int(config.if_normals),
    config.batch_size,
    config.lr,
    config.knn,
    config.knn_step,
    config.more
)

print("Model name: ", model_name)

logger_train = Logger("logs/tensorboard/train_feature/{}/train".format(model_name), flush_secs=15)
logger_val = Logger("logs/tensorboard/train_feature/{}/val".format(model_name), flush_secs=15)

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
source_file_config = sys.argv[1]
destination_file_config = "logs/scripts/{}/{}".format(
    model_name, sys.argv[1].split("/")[-1]
)
copyfile(source_file_config, destination_file_config)

Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)

model = PrimitivesEmbeddingDGCNGn(
    embedding=True,
    emb_size=128,
    primitives=True,
    num_primitives=config.num_primitives,
    loss_function=Loss.triplet_loss,
    mode=config.mode,
    if_normals=config.if_normals,
    knn=config.knn,
    knn_step=config.knn_step
)

model = torch.nn.DataParallel(model)

# model.load_state_dict(
#     torch.load("logs/pretrained_models/" + config.pretrain_model_path)
# )
model.cuda()

# Do not train the encoder weights to save gpu memory.
for key, values in model.named_parameters():
    if key.startswith("module.encoder"):
        values.requires_grad = True
    else:
        values.requires_grad = True

dataset = Dataset(
    config
)

d_mean = config.d_mean
d_scale = config.d_scale
if_augment = False

get_train_data = dataset.get_train(d_mean=True, d_scale=d_scale)

get_val_data = dataset.get_val(d_mean=True, d_scale=d_scale)

optimizer = optim.Adam(model.parameters(), lr=config.lr)

# optimizer.load_state_dict(torch.load("logs/pretrained_models/" +
#                                      config.pretrain_model_path.split(".")[0] + "_optimizer.pth"))

loader = generator_iter(get_train_data, int(1e10))
get_train_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=8,
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
        num_workers=8,
        pin_memory=True,
    )
)

# Reduce learning rate when a metric has stopped improving.
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=1e-4
)

# model_bkp.triplet_loss = Loss.triplet_loss
prev_loss_value_train = 1e4
prev_loss_value_val = 1e4

print("started training!")

# no updates to the bn

for e in range(config.epochs):
    torch.cuda.empty_cache()

    train_loss = []
    train_prim_loss = []
    train_emb_loss = []

    model.train()
    for train_b_id in range(dataset.train_points.shape[0] // config.batch_size):
        optimizer.zero_grad() # 每个batch清一次
        while True:
            points_, normals_,_, T_batch, labels_, primitives_ = next(get_train_data)[0]
            break 
            if np.unique(labels).shape[0] < 3:
                continue
            else:
                break

        rand_num_points = config.num_points

        points = torch.from_numpy(points_.astype(np.float32)).cuda()[:,0:rand_num_points,:]
        normals = torch.from_numpy(normals_.astype(np.float32)).cuda()[:,0:rand_num_points,:]

        primitives = torch.from_numpy(primitives_.astype(np.int64)).cuda()[:,0:rand_num_points]
        labels = labels_[:,0:rand_num_points] # 之后有用到np.unique，所以还不能转换为tensor
        
        if config.if_normals:
            embedding, primitives_log_prob, embed_loss = model(torch.cat([points.permute(0, 2, 1),normals.permute(0, 2, 1)],1), torch.from_numpy(labels).cuda(), True
            )
        else:
            embedding, primitives_log_prob, embed_loss = model(
                points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
            )

        embed_loss = torch.mean(embed_loss)
        prim_loss = primitive_loss(primitives_log_prob, primitives)

        loss = embed_loss + prim_loss

        loss.backward()
        optimizer.step()

        train_loss.append(loss.data.cpu().numpy())
        train_prim_loss.append(prim_loss.data.cpu().numpy())
        train_emb_loss.append(embed_loss.data.cpu().numpy())

        # 
        print(
            "\rEpoch: {} iter: {}, loss: {:.4} = el: {:.4} + pl: {:.4}".format(
                e, train_b_id, loss.item(), embed_loss.item(), prim_loss.item()
            ),
            end="",
        )

    del loss, embed_loss, prim_loss

    val_loss = []
    val_prim_loss = []
    val_emb_loss = []

    torch.cuda.empty_cache()
    model.eval() # 不启用Batch Normalization和Dropout 

    # 这里据说要 - 1
    for val_b_id in range(dataset.val_points.shape[0] // config.batch_size):
        
        points_, normals_,_, T_batch,labels_, primitives_ = next(get_val_data)[0]
 
        points = torch.from_numpy(points_.astype(np.float32)).cuda()[:,0:rand_num_points,:]
        normals = torch.from_numpy(normals_.astype(np.float32)).cuda()[:,0:rand_num_points,:]

        primitives = torch.from_numpy(primitives_.astype(np.int64)).cuda()[:,0:rand_num_points]
        labels = labels_[:,0:rand_num_points] # 之后有用到np.unique，所以还不能转换为tensor

        with torch.no_grad():
            if config.if_normals:
                embedding, primitives_log_prob, embed_loss = model(torch.cat([points.permute(0, 2, 1),normals.permute(0, 2, 1)],1), torch.from_numpy(labels).cuda(), True
                )
            else:
                embedding, primitives_log_prob, embed_loss = model(
                    points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )
            prim_loss = primitive_loss(primitives_log_prob, primitives)
        
        embed_loss = torch.mean(embed_loss)
        loss = embed_loss + prim_loss

        val_loss.append(loss.data.cpu().numpy())
        val_prim_loss.append(prim_loss.data.cpu().numpy())
        val_emb_loss.append(embed_loss.data.cpu().numpy())

    logger.info(
        "\nEpoch: {}/{} =>\nTrain: loss: {:.4} = el: {:.4} + pl: {:.4}\nVal: loss: {:.4} = el: {:.4} + pl: {:.4}".format(
            e,
            config.epochs,
            np.mean(train_loss),
            np.mean(train_emb_loss),
            np.mean(train_prim_loss),
            # val
            np.mean(val_loss),
            np.mean(val_emb_loss),
            np.mean(val_prim_loss),
        )
    )

    logger_train.log_value("loss/loss", np.mean(train_loss), e)
    logger_val.log_value("loss/loss", np.mean(val_loss), e)

    logger_train.log_value("loss/emb_loss", np.mean(train_emb_loss), e)
    logger_val.log_value("loss/emb_loss", np.mean(val_emb_loss), e)

    logger_train.log_value("loss/prim_loss", np.mean(train_prim_loss), e)
    logger_val.log_value("loss/prim_loss", np.mean(val_prim_loss), e)

    scheduler.step(np.mean(val_loss))

    # 1.
    monitor_loss = train_loss
    loss_name = "train_loss"
    if prev_loss_value_train > np.mean(monitor_loss):
        logger.info("{} improvement, saving model at epoch: {}".format(loss_name,e))
        prev_loss_value_train = np.mean(monitor_loss)
        
        torch.save(
            model.module.state_dict(),
            "logs/trained_models/{}/{}_singleGPU.pth".format(model_name,loss_name),
        )

        torch.save(
            model.state_dict(),
            "logs/trained_models/{}/{}_multGPU.pth".format(model_name,loss_name),
        )

        torch.save(
            optimizer.state_dict(),
            "logs/trained_models/{}/{}_optimizer.pth".format(model_name,loss_name),
        )
    # 2.
    monitor_loss = val_loss
    loss_name = "val_loss"
    if prev_loss_value_val > np.mean(monitor_loss):
        logger.info("{} improvement, saving model at epoch: {}".format(loss_name,e))
        prev_loss_value_val = np.mean(monitor_loss)

        torch.save(
            model.module.state_dict(),
            "logs/trained_models/{}/{}_singleGPU.pth".format(model_name,loss_name),
        )

        torch.save(
            model.state_dict(),
            "logs/trained_models/{}/{}_multGPU.pth".format(model_name,loss_name),
        )

        torch.save(
            optimizer.state_dict(),
            "logs/trained_models/{}/{}_optimizer.pth".format(model_name,loss_name),
        )

# os.system('cp logs/trained_models/{}/val_loss_singleGPU.pth logs/pretrained_models/quadrics_feature/if_normals_{}/'.format(model_name,int(config.if_normals)))
# os.system('cp logs/trained_models/{}/val_loss_multGPU.pth logs/pretrained_models/quadrics_feature/if_normals_{}/'.format(model_name,int(config.if_normals)))
# os.system('cp logs/trained_models/{}/val_loss_optimizer.pth logs/pretrained_models/quadrics_feature/if_normals_{}/'.format(model_name,int(config.if_normals)))