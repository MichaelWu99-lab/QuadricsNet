import gc
import json
import logging
import os
import sys
import traceback
from shutil import copyfile

import numpy as np

from src.read_config_e2e import Config
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
    "".join(list(map(str, config.if_fitting_normals))),
    int(config.if_detection_normals),
    config.lamb_0_0,
    config.lamb_0_1,
    config.lamb_0_2,
    config.lamb_0_3,
    config.lamb_0_4,
    config.lamb_0_5,
    config.lamb_0_6,
    config.lamb_1,
    config.cluster_iterations,
    config.batch_size,
    config.lr,
    config.knn,
    config.knn_step,
    config.more
)

print("Model name: ", model_name)

logger_train = Logger("logs/tensorboard/train_e2e/{}/train".format(model_name), flush_secs=15)
logger_val = Logger("logs/tensorboard/train_e2e/{}/val".format(model_name), flush_secs=15)

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
    if_normals=config.if_detection_normals,
    knn=config.knn,
    knn_step=config.knn_step
)

# device = torch.device("cuda:0")

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    model.load_state_dict(
    torch.load(config.detection_model_path+"if_normals_{}/".format(int(config.if_detection_normals)) + "val_loss_multGPU.pth")
    )
else:
    model.load_state_dict(
    torch.load(config.detection_model_path+"if_normals_{}/".format(int(config.if_detection_normals)) + "val_loss_singleGPU.pth")
    )

model.cuda()

# Do not train the encoder weights to save gpu memory.
for key, values in model.named_parameters():
    if key.startswith("module.encoder"):
        values.requires_grad = True
    else:
        values.requires_grad = True

evaluation = Evaluation(config)

dataset = Dataset(
    config
)


get_train_data = dataset.get_train(d_mean=config.d_mean, d_scale=config.d_scale)

get_val_data = dataset.get_val(d_mean=config.d_mean, d_scale=config.d_scale)

optimizer = optim.Adam(model.parameters(), lr=config.lr)

# optimizer.load_state_dict(torch.load(config.pretrain_model_path + "_optimizer.pth"))

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

prev_loss_value_train = 1e4
prev_loss_value_val = 1e4
print("started training!")

if torch.cuda.device_count() > 1:
    alt_gpu = 1
else:
    alt_gpu = 0

lamb_0 = [config.lamb_0_0, config.lamb_0_1, config.lamb_0_2, config.lamb_0_3,config.lamb_0_4,config.lamb_0_5,config.lamb_0_6]
lamb_1 = config.lamb_1
# no updates to the bn

for e in range(config.epochs):
    torch.cuda.empty_cache()
    
    train_iou = []
    train_seg_iou = []
    train_loss = []
    train_prim_loss = []
    train_emb_loss = []
    train_res_loss = []
    train_quadrics_reg_loss = []
    train_quadrics_function_loss = []
    train_quadrics_decomposition_r = []
    train_quadrics_decomposition_s = []
    train_quadrics_decomposition_t = []
    train_normals_deviation = []
    train_quadrics_r_regularization = []

    model.train()
    for train_b_id in range(dataset.train_points.shape[0] // config.batch_size):
        optimizer.zero_grad() # 每个batch清一次
        while True:
            points_, normals_,quadrics_, T_batch,labels_, primitives_ = next(get_train_data)[0]
            # Take only training dataset with no. segments more than 2
            break
            if np.unique(labels).shape[0] < 3:
                continue
            else:
                break

        points = torch.from_numpy(points_.astype(np.float32)).cuda()
        normals = torch.from_numpy(normals_.astype(np.float32)).cuda()

        quadrics = torch.from_numpy(quadrics_.astype(np.float32)).cuda()
        primitives = torch.from_numpy(primitives_.astype(np.int64)).cuda()

        if config.if_detection_normals:
            embedding, primitives_log_prob, embed_loss = model(torch.cat([points.permute(0, 2, 1),normals.permute(0, 2, 1)],1), torch.from_numpy(labels_).cuda(), True
            )
        else:
            embedding, primitives_log_prob, embed_loss = model(
                points.permute(0, 2, 1), torch.from_numpy(labels_).cuda(), True
            )

        embed_loss = torch.mean(embed_loss)
        prim_loss = primitive_loss(primitives_log_prob, primitives)
        torch.cuda.empty_cache()

        metric, _,_,_,_,_,_,_,_,_,_ = evaluation.fitting_loss(
                embedding.permute(0, 2, 1).to(torch.device("cuda:{}".format(alt_gpu))),
                points.to(torch.device("cuda:{}".format(alt_gpu))),
                normals.to(torch.device("cuda:{}".format(alt_gpu))),
                quadrics,
                labels_,
                primitives.to(torch.device("cuda:{}".format(alt_gpu))),
                primitives_log_prob.to(torch.device("cuda:{}".format(alt_gpu))),
                quantile=0.025,
                eval=False,
                iterations=config.cluster_iterations,
                lamb=lamb_0,
                if_fitting_normals = config.if_fitting_normals
            )

        # metric组成：[res_loss, quadrics_reg_loss, quadrics_function_loss, quadrics_decomposition_r,quadrics_decomposition_s, seg_iou, iou]
        res_loss = metric[0].to(torch.device("cuda:0"))

        seg_iou, iou = metric[8:]

        loss = embed_loss + prim_loss + lamb_1 * res_loss

        loss.backward()
        optimizer.step()

        train_iou.append(iou)
        train_seg_iou.append(seg_iou)
        train_loss.append(loss.data.cpu().numpy())
        train_prim_loss.append(prim_loss.data.cpu().numpy())
        train_emb_loss.append(embed_loss.data.cpu().numpy())
        train_res_loss.append(res_loss.data.cpu().numpy())
        train_quadrics_reg_loss.append(metric[1].data.cpu().numpy())
        train_quadrics_function_loss.append(metric[2].data.cpu().numpy())
        train_quadrics_decomposition_r.append(metric[3].data.cpu().numpy())
        train_quadrics_decomposition_s.append(metric[4].data.cpu().numpy())
        train_quadrics_decomposition_t.append(metric[5].data.cpu().numpy())
        train_normals_deviation.append(metric[6].data.cpu().numpy())
        train_quadrics_r_regularization.append(metric[7].data.cpu().numpy())

        # 
        print(
            "\rEpoch: {} iter: {}, loss: {:.4} = el: {:.4} + pl: {:.4} + rl: {:.4}*{}(=f.reg.rot.s.t.n.reguRo-{}-{}-{}-{}-{}-{}-{}: {:.4}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}), seg_iou: {:.4}, iou: {:.4}".format(
            e, train_b_id, loss.item(), embed_loss.item(), prim_loss.item(), res_loss.item(),lamb_1,lamb_0[0],lamb_0[1],lamb_0[2],lamb_0[3],lamb_0[4],lamb_0[5],lamb_0[6], metric[2].item(),metric[1].item(),metric[3].item(),metric[4].item(),metric[5].item(),metric[6].item(),metric[7].item(),seg_iou.item(),iou.item()
            ),
            end="",
        )

    del res_loss, loss, embed_loss, prim_loss, seg_iou, iou,metric

    val_iou = []
    val_seg_iou = []
    val_loss = []
    val_prim_loss = []
    val_emb_loss = []
    val_res_loss = []
    val_quadrics_reg_loss = []
    val_quadrics_function_loss = []
    val_quadrics_decomposition_r = []
    val_quadrics_decomposition_s = []
    val_quadrics_decomposition_t = []
    val_normals_deviation = []
    val_quadrics_r_regularization = []
   

    torch.cuda.empty_cache()
    model.eval()

    for val_b_id in range(dataset.val_points.shape[0] // config.batch_size):

        points_,normals_, quadrics_, T_batch,labels_, primitives_ = next(get_val_data)[0]
 
        points = torch.from_numpy(points_.astype(np.float32)).cuda()
        normals = torch.from_numpy(normals_.astype(np.float32)).cuda()

        quadrics = torch.from_numpy(quadrics_.astype(np.float32)).cuda()
        primitives = torch.from_numpy(primitives_.astype(np.int64)).cuda()

        with torch.no_grad():
            if config.if_feature_normals:
                embedding, primitives_log_prob, embed_loss = model(torch.cat([points.permute(0, 2, 1),normals.permute(0, 2, 1)],1), torch.from_numpy(labels_).cuda(), True
                )
            else:
                embedding, primitives_log_prob, embed_loss = model(
                    points.permute(0, 2, 1), torch.from_numpy(labels_).cuda(), True
                )

            prim_loss = primitive_loss(primitives_log_prob, primitives)
            embed_loss = torch.mean(embed_loss)
            
            metric, _,_,_,_,_,_,_,_,_,_= evaluation.fitting_loss(
                embedding.permute(0, 2, 1).to(torch.device("cuda:{}".format(alt_gpu))),
                points.to(torch.device("cuda:{}".format(alt_gpu))),
                normals.to(torch.device("cuda:{}".format(alt_gpu))),
                quadrics,
                labels_,
                primitives.to(torch.device("cuda:{}".format(alt_gpu))),
                primitives_log_prob.to(torch.device("cuda:{}".format(alt_gpu))),
                quantile=0.025,
                iterations=config.cluster_iterations,
                lamb=lamb_0,
                eval=True,
                if_fitting_normals = config.if_fitting_normals
            )

       # metric：[res_loss, quadrics_reg_loss, quadrics_function_loss, seg_iou, iou]
        res_loss = metric[0].to(torch.device("cuda:0"))

        seg_iou, iou = metric[8:]

        loss = embed_loss + prim_loss + lamb_1 * res_loss

        val_iou.append(iou)
        val_seg_iou.append(seg_iou)
        val_loss.append(loss.data.cpu().numpy())
        val_prim_loss.append(prim_loss.data.cpu().numpy())
        val_emb_loss.append(embed_loss.data.cpu().numpy())
        val_res_loss.append(res_loss.data.cpu().numpy())
        val_quadrics_reg_loss.append(metric[1].data.cpu().numpy())
        val_quadrics_function_loss.append(metric[2].data.cpu().numpy())
        val_quadrics_decomposition_r.append(metric[3].data.cpu().numpy())
        val_quadrics_decomposition_s.append(metric[4].data.cpu().numpy())
        val_quadrics_decomposition_t.append(metric[5].data.cpu().numpy())
        val_normals_deviation.append(metric[6].data.cpu().numpy())
        val_quadrics_r_regularization.append(metric[7].data.cpu().numpy())

    logger.info(
        "\nEpoch: {}/{} =>\nTrain: loss: {:.4} = el: {:.4} + pl: {:.4} + rl: {:.4}*{}(=f.reg.rot.s.t.n.reguR0-{}-{}-{}-{}-{}-{}-{}: {:.4}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}), seg_iou: {:.4}, iou: {:.4}\nVal: loss: {:.4} = el: {:.4} + pl: {:.4} + rl: {:.4}*{}(=f.reg.rot.s.t.n.reguR0-{}-{}-{}-{}-{}-{}-{}: {:.4}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}), seg_iou: {:.4}, iou: {:.4}".format(
            e,
            config.epochs,
            np.mean(train_loss),
            np.mean(train_emb_loss),
            np.mean(train_prim_loss),
            np.mean(train_res_loss),
            lamb_1,
            lamb_0[0],
            lamb_0[1],
            lamb_0[2],
            lamb_0[3],
            lamb_0[4],
            lamb_0[5],
            lamb_0[6],
            np.mean(train_quadrics_function_loss),
            np.mean(train_quadrics_reg_loss),
            np.mean(train_quadrics_decomposition_r),
            np.mean(train_quadrics_decomposition_s),
            np.mean(train_quadrics_decomposition_t),
            np.mean(train_normals_deviation),
            np.mean(train_quadrics_r_regularization),
            np.mean(train_seg_iou),
            np.mean(train_iou),
            # val
            np.mean(val_loss),
            np.mean(val_emb_loss),
            np.mean(val_prim_loss),
            np.mean(val_res_loss),
            lamb_1,
            lamb_0[0],
            lamb_0[1],
            lamb_0[2],
            lamb_0[3],
            lamb_0[4],
            lamb_0[5],
            lamb_0[6],
            np.mean(val_quadrics_function_loss),
            np.mean(val_quadrics_reg_loss),
            np.mean(val_quadrics_decomposition_r),
            np.mean(val_quadrics_decomposition_s),
            np.mean(val_quadrics_decomposition_t),
            np.mean(val_normals_deviation),
            np.mean(val_quadrics_r_regularization),
            np.mean(val_seg_iou),
            np.mean(val_iou),
        )
    )

    logger_train.log_value("loss/loss", np.mean(train_loss), e)
    logger_val.log_value("loss/loss", np.mean(val_loss), e)

    logger_train.log_value("loss/emb_loss", np.mean(train_emb_loss), e)
    logger_val.log_value("loss/emb_loss", np.mean(val_emb_loss), e)

    logger_train.log_value("loss/prim_loss", np.mean(train_prim_loss), e)
    logger_val.log_value("loss/prim_loss", np.mean(val_prim_loss), e)

    logger_train.log_value("loss/res_loss", np.mean(train_res_loss), e)
    logger_val.log_value("loss/res_loss", np.mean(val_res_loss), e)

    logger_train.log_value("res_loss/quadrics_reg_loss", np.mean(train_quadrics_reg_loss), e)
    logger_val.log_value("res_loss/quadrics_reg_loss", np.mean(val_quadrics_reg_loss), e)

    logger_train.log_value("res_loss/quadrics_function_loss", np.mean(train_quadrics_function_loss), e)
    logger_val.log_value("res_loss/quadrics_function_loss", np.mean(val_quadrics_function_loss), e)

    logger_train.log_value("res_loss/quadrics_decomposition_r", np.mean(train_quadrics_decomposition_r), e)
    logger_val.log_value("res_loss/quadrics_decomposition_r", np.mean(val_quadrics_decomposition_r), e)

    logger_train.log_value("res_loss/quadrics_decomposition_s", np.mean(train_quadrics_decomposition_s), e)
    logger_val.log_value("res_loss/quadrics_decomposition_s", np.mean(val_quadrics_decomposition_s), e)

    logger_train.log_value("res_loss/quadrics_decomposition_t", np.mean(train_quadrics_decomposition_t), e)
    logger_val.log_value("res_loss/quadrics_decomposition_t", np.mean(val_quadrics_decomposition_t), e)

    logger_train.log_value("res_loss/normals_deviation", np.mean(train_normals_deviation), e)
    logger_val.log_value("res_loss/normals_deviation", np.mean(val_normals_deviation), e)

    logger_train.log_value("res_loss/quadrics_r_regularization", np.mean(train_quadrics_r_regularization), e)
    logger_val.log_value("res_loss/quadrics_r_regularization", np.mean(val_quadrics_r_regularization), e)

    logger_train.log_value("iou/seg_iou", np.mean(train_seg_iou), e)
    logger_val.log_value("iou/seg_iou", np.mean(val_seg_iou), e)

    logger_train.log_value("iou/iou", np.mean(train_iou), e)
    logger_val.log_value("iou/iou", np.mean(val_iou), e)

    scheduler.step(np.mean(val_res_loss))

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
