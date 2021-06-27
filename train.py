import os

import torch
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

import pickle as pkl
import argparse
import pdb

from dataset import ContourDataset
from config import cfg
from config import update_config
from function import train
from function import validate
from model import GCNModel, SAGEModel, GATModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args


def main():
    # basic configurations
    args = parse_args()
    update_config(cfg, args)

    try:
        assert os.path.exists(cfg.OUTPUT_DIR)
    except AssertionError:
        os.system('mkdir ' + cfg.OUTPUT_DIR)

    try:
        assert os.path.exists(cfg.LOG_DIR)
    except AssertionError:
        os.system('mkdir ' + cfg.LOG_DIR)

    final_output_dir = cfg.OUTPUT_DIR

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # model
    assert config.MODEL.GNN_TYPE in ['GCN', 'SAGE', 'GAT']
    with torch.cuda.device(cfg.GPUS[0]):
        if config.MODEL.GNN_TYPE == 'GCN':
            with open('edge.pkl', 'rb') as edge:
                std_edge = torch.from_numpy(pkl.load(edge)).long().cuda()

            model = GCNModel(num_nodes=cfg.MODEL.NUM_NODES,
                            node_feature_dim=cfg.MODEL.NODE_FEATURE_DIM,
                            std_edge=std_edge)
            model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

        if config.MODEL.GNN_TYPE == 'SAGE':
            with open('edge.pkl', 'rb') as edge:
                std_edge = torch.from_numpy(pkl.load(edge)).long().cuda()

            model = SAGEModel(num_nodes=cfg.MODEL.NUM_NODES,
                            node_feature_dim=cfg.MODEL.NODE_FEATURE_DIM,
                            std_edge=std_edge)
            model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

        if config.MODEL.GNN_TYPE == 'GAT':
            with open('edge.pkl', 'rb') as edge:
                std_edge = torch.from_numpy(pkl.load(edge)).long().cuda()

            model = GATModel(num_nodes=cfg.MODEL.NUM_NODES,
                            node_feature_dim=cfg.MODEL.NODE_FEATURE_DIM,
                            std_edge=std_edge)
            model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()


    # summary writer
    writer_dict = {
        'writer': SummaryWriter(log_dir=cfg.LOG_DIR),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # dataset
    training_set = ContourDataset(cfg, is_train=True)
    valid_set = ContourDataset(cfg, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True)

    # statistics
    last_epoch = -1
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')

    # optimizer
    optimizer = optim.Adam(params=model.parameters(),
                           lr=cfg.TRAIN.LR,
                           weight_decay=cfg.TRAIN.WD)

    # auto resume
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    # lr scheduler
    warmup_milestone = cfg.TRAIN.END_EPOCH // 20
    lrstep_milestone = list(
        map(int, [
            cfg.TRAIN.END_EPOCH * 0.4, cfg.TRAIN.END_EPOCH * 0.6,
            cfg.TRAIN.END_EPOCH * 0.8
        ]))
    warmup_lr = lambda epoch: (epoch + 1) / 20 * cfg.TRAIN.LR if epoch <= warmup_milestone else \
        cfg.TRAIN.LR * cfg.TRAIN.LR_FACTOR ** len([m for m in lrstep_milestone if m <= epoch])

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=warmup_lr,
                                                     last_epoch=last_epoch)

    try:
        os.mkdir(os.path.join(final_output_dir, 'image'))
        os.mkdir(os.path.join(final_output_dir, 'heatmap'))
    except FileExistsError:
        pass

    train_perf_name = cfg.PERF_FNAME + '.pkl'
    val_perf_name = 'val' + cfg.PERF_FNAME[5:] + '.pkl'

    image_train = training_set.size
    training_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    batches_per_train = image_train // training_size + 1

    image_val = valid_set.size
    validation_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    batches_per_val = image_val // validation_size + 1

    try:
        train_perf = pkl.load(open(train_perf_name, 'rb'))
        val_perf = pkl.load(open(val_perf_name, 'rb'))
        assert begin_epoch * batches_per_train == len(train_perf['pck'])
        assert begin_epoch * batches_per_val == len(val_perf['pck'])
    except AssertionError:
        train_perf = {'pck': [], 'pckh': [], 'oks': [], 'loss': []}
        val_perf = {'pck': [], 'pckh': [], 'oks': [], 'loss': []}
    except FileNotFoundError:
        train_perf = {'pck': [], 'pckh': [], 'oks': [], 'loss': []}
        val_perf = {'pck': [], 'pckh': [], 'oks': [], 'loss': []}

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        epoch_pck, epoch_pckh, epoch_oks, epoch_loss = train(
            cfg, train_loader, model, optimizer, epoch, final_output_dir,
            writer_dict)

        for i in range(len(epoch_pck)):
            train_perf['pck'].append(epoch_pck[i])
            train_perf['pckh'].append(epoch_pckh[i])
            train_perf['oks'].append(epoch_oks[i])
            train_perf['loss'].append(epoch_loss[i])
        with open(train_perf_name, 'wb') as f:
            pkl.dump(train_perf, f)

        lr_scheduler.step()

        # evaluate on validation set
        epoch_pck, epoch_pckh, epoch_oks, epoch_loss = validate(
            cfg, valid_loader, model, epoch, final_output_dir, writer_dict)

        for i in range(len(epoch_pck)):
            val_perf['pck'].append(epoch_pck[i])
            val_perf['pckh'].append(epoch_pckh[i])
            val_perf['oks'].append(epoch_oks[i])
            val_perf['loss'].append(epoch_loss[i])
        with open(val_perf_name, 'wb') as f:
            pkl.dump(val_perf, f)

        # save training status
        print('=> saving checkpoint to {}'.format(final_output_dir))
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_stae_dict': model.state_dict(),
            'perf': max(epoch_pck),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(final_output_dir, 'checkpoint.pth'))

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    print('=> saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
