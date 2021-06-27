from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import random
import numpy as np
import torch

from utils import save_sample_image, oks

def train(config, train_loader, model, pretrained_model, optimizer, epoch, output_dir,
          writer_dict):
    model.train()
    torch.autograd.set_detect_anomaly(True)

    # Statistics
    samples_viewed = 0
    total_oks = 0
    total_full_score = 0
    gained_score = 0
    gained_score_h = 0

    im_w = config.MODEL.IMAGE_SIZE[1]
    im_h = config.MODEL.IMAGE_SIZE[0]
    num_points = config.MODEL.NUM_JOINTS

    perf_loss = []
    perf_pck = []
    perf_pckh = []
    perf_oks = []

    with torch.cuda.device(config.GPUS[0]):
        for i, (inputs, target_coords,
                target_weight) in enumerate(train_loader):
            # [+] 输入位置
            inputs = inputs.reshape(-1, 3, im_h, im_w)
            target_coords = target_coords.reshape(-1, num_points, 2)
            target_weight = target_weight.reshape(-1, num_points, 1)

            batch_size = inputs.shape[0]

            inputs = inputs.float().cuda()
            target_coords = target_coords.float()
            target_weight = target_weight.float().cuda()

            # [+] 此步骤根据归一化gt坐标target_coords生成绝对gt坐标targets
            targets = target_coords.numpy().copy().reshape(-1, 2)
            im_X = im_w * targets[:, 0]
            im_Y = im_h * targets[:, 1]
            targets[:, 0] = np.rint(im_X).astype(np.int32)
            targets[:, 1] = np.rint(im_Y).astype(np.int32)
            targets = targets.reshape(batch_size, num_points, 2)

            # [+] 生成pck指标阈值
            pck_thresh = np.sqrt(im_w * im_w + im_h * im_h) * 0.1
            pckh_thresh = np.sqrt(
                np.sum(np.square(targets[:, 0] - targets[:, 1]), axis=1)) * 0.5

            # [+] 去除不需要回归的顶点（头顶、下巴）
            targets = targets[:, 2:].astype(np.int32)
            target_coords = target_coords[:, 2:]
            target_weight = target_weight[:, 2:]

            # [+] 提取图片feature，选取中间层的方法未有定论，可在model_res文件中修改
            with torch.no_grad():
                _, x1, x2, x3, x4 = pretrained_model(inputs)
                up2 = torch.nn.Upsample(scale_factor=2)
                # up4 = torch.nn.Upsample(scale_factor=4)
                # up8 = torch.nn.Upsample(scale_factor=8)
                x1 = up2(x1)
                x2 = up2(x2)
                net_inp = torch.cat([x1, x2, x3, x4], dim=1)

            # [+] 模型输出
            output = model(net_inp)
            output = torch.reshape(output, (-1, 63, 2))

            if not output.is_cuda:
                output = output.cuda(non_blocking=True)
            if not target_coords.is_cuda:
                target_coords = target_coords.cuda()

            # [+] loss计算
            mse_items = torch.sum(torch.abs(
                torch.add(output, -target_coords)),
                                        axis=2)
            mse_items = mse_items * target_weight[:, :, 0]
            loss = torch.mean(mse_items, axis=(0, 1))


            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            perf_loss.append(loss.item())
            fine_preds = output.detach().cpu().numpy()
            fine_preds[:, :, 0] = fine_preds[:, :, 0] * im_w
            fine_preds[:, :, 1] = fine_preds[:, :, 1] * im_h
            fine_preds = fine_preds.astype(np.int32)

            batch_score = 0
            batch_score_h = 0
            samples_viewed += output.shape[0]
            total_oks += oks(fine_preds.copy(), im_h, im_w, targets,
                             target_weight.detach().cpu().numpy())
            batch_oks = total_oks / samples_viewed

            for ii in range(fine_preds.shape[0]):
                eu_dist = np.sqrt(
                    np.sum(np.square(fine_preds[ii] - targets[ii]), axis=1))
                batch_score += torch.sum(
                    target_weight[ii][eu_dist <= pck_thresh])
                batch_score_h += torch.sum(
                    target_weight[ii][eu_dist <= pckh_thresh[ii]])

            full_score = torch.sum(target_weight)
            gained_score += batch_score
            gained_score_h += batch_score_h
            batch_acc = (batch_score / full_score).item()
            batch_acc_h = (batch_score_h / full_score).item()
            total_full_score += full_score
            current_avg_acc = (gained_score / total_full_score).item()
            current_avg_acc_h = (gained_score_h / total_full_score).item()

            perf_pck.append(batch_acc)
            perf_pckh.append(batch_acc_h)
            perf_oks.append(batch_oks)

            # [+] 打印epoch信息
            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}]\n' \
                    'Train: [Batch {1}/{2}]\t' \
                    'Loss {loss:.5f} \tOKS {batch_oks:.4f}\n' \
                    'PCK10: batch {batch_acc:.3f} ({batch_score:.2f}/{full_score:.2f}); average {current_avg_acc:.3f}\n' \
                    'PCKh50: batch {batch_acc_h:.3f} ({batch_score_h:.2f}/{full_score:.2f}); average {current_avg_acc_h:.3f}'.format(
                        epoch, i, len(train_loader),
                        loss=loss.item(), batch_oks=batch_oks,
                        batch_acc=batch_acc, batch_score=batch_score,
                        full_score=full_score, current_avg_acc=current_avg_acc,
                        batch_acc_h=batch_acc_h, batch_score_h=batch_score_h,
                        current_avg_acc_h=current_avg_acc_h)

                print(msg)

                prefix_img = '{}_epoch{}_batch{}'.format(
                    os.path.join(output_dir, 'image', 'train'), epoch, i)
                batch_size = inputs.shape[0]
                show_id = random.randint(0, batch_size - 1)

                target_coords = target_coords.detach().cpu()
                save_sample_image(inputs[show_id], target_coords[show_id],
                                  fine_preds[show_id], target_weight[show_id], prefix_img)

    return perf_pck, perf_pckh, perf_oks, perf_loss


def validate(config, val_loader, model, pretrained_model, epoch, output_dir, writer_dict=None):
    model.eval()

    with torch.no_grad():
        samples_viewed = 0
        total_oks = 0
        total_full_score = 0
        gained_score = 0
        gained_score_h = 0

        im_w = config.MODEL.IMAGE_SIZE[1]
        im_h = config.MODEL.IMAGE_SIZE[0]
        num_points = config.MODEL.NUM_JOINTS

        perf_loss = []
        perf_pck = []
        perf_pckh = []
        perf_oks = []

        with torch.cuda.device(config.GPUS[0]):
            for i, (inputs, target_coords,
                    target_weight) in enumerate(val_loader):
                # [+] 输入位置
                inputs = inputs.reshape(-1, 3, im_h, im_w)
                target_coords = target_coords.reshape(-1, num_points, 2)
                target_weight = target_weight.reshape(-1, num_points, 1)

                batch_size = inputs.shape[0]

                inputs = inputs.float().cuda()
                target_coords = target_coords.float()
                target_weight = target_weight.float().cuda()

                # [+] 此步骤根据归一化gt坐标target_coords生成绝对gt坐标targets
                targets = target_coords.numpy().copy().reshape(-1, 2)
                im_X = im_w * targets[:, 0]
                im_Y = im_h * targets[:, 1]
                targets[:, 0] = np.rint(im_X).astype(np.int32)
                targets[:, 1] = np.rint(im_Y).astype(np.int32)
                targets = targets.reshape(batch_size, num_points, 2)
                
                # [+] 生成pck指标阈值
                pck_thresh = np.sqrt(im_w * im_w + im_h * im_h) * 0.1
                pckh_thresh = np.sqrt(
                    np.sum(np.square(targets[:, 0] - targets[:, 1]),
                           axis=1)) * 0.5

                # [+] 去除不需要回归的顶点（头顶、下巴）
                targets = targets[:, 2:].astype(np.int32)
                target_coords = target_coords[:, 2:]
                target_weight = target_weight[:, 2:]

                # [+] 提取图片feature，选取中间层的方法未有定论，可在model_res文件中修改
                _, x1, x2, x3, x4 = pretrained_model(inputs)
                up2 = torch.nn.Upsample(scale_factor=2)
                # up4 = torch.nn.Upsample(scale_factor=4)
                # up8 = torch.nn.Upsample(scale_factor=8)
                x2 = up2(x2)
                x1 = up2(x1)
                # x4 = up8(x4)
                net_input = torch.cat([
                    x1, x2, x3, x4], dim=1)

                # [+] 模型输出
                output = model(net_input)
                output = torch.reshape(output, (-1, 63, 2))

                if not output.is_cuda:
                    output = output.cuda(non_blocking=True)
                if not target_coords.is_cuda:
                    target_coords = target_coords.cuda()

                # [+] loss计算
                mse_items = torch.sum(torch.abs(
                    torch.add(output, -target_coords)),
                                            axis=2)
                mse_items = mse_items * target_weight[:, :, 0]
                loss = torch.mean(mse_items, axis=(0, 1))
                perf_loss.append(loss.item())

                fine_preds = output.detach().cpu().numpy()
                fine_preds[:, :, 0] = fine_preds[:, :, 0] * im_w
                fine_preds[:, :, 1] = fine_preds[:, :, 1] * im_h
                fine_preds = fine_preds.astype(np.int32)

                batch_score = 0
                batch_score_h = 0
                samples_viewed += output.shape[0]
                total_oks += oks(fine_preds.copy(), im_h, im_w, targets,
                                 target_weight.detach().cpu().numpy())
                batch_oks = total_oks / samples_viewed

                for ii in range(fine_preds.shape[0]):
                    eu_dist = np.sqrt(
                        np.sum(np.square(fine_preds[ii] - targets[ii]),
                               axis=1))
                    batch_score += torch.sum(
                        target_weight[ii][eu_dist <= pck_thresh])
                    batch_score_h += torch.sum(
                        target_weight[ii][eu_dist <= pckh_thresh[ii]])

                full_score = torch.sum(target_weight)
                gained_score += batch_score
                gained_score_h += batch_score_h
                batch_acc = (batch_score / full_score).item()
                batch_acc_h = (batch_score_h / full_score).item()
                total_full_score += full_score
                current_avg_acc = (gained_score / total_full_score).item()
                current_avg_acc_h = (gained_score_h / total_full_score).item()

                perf_pck.append(batch_acc)
                perf_pckh.append(batch_acc_h)
                perf_oks.append(batch_oks)

                # [+] 打印epoch信息
                if i % config.PRINT_FREQ == 0:
                    msg = 'Test: [Batch {0}/{1}]\t' \
                        'Loss {loss:.5f} \tOKS {batch_oks:.4f}\n' \
                        'PCK10: batch {batch_acc:.3f} ({batch_score:.2f}/{full_score:.2f}); average {current_avg_acc:.3f}\n' \
                        'PCKh50: batch {batch_acc_h:.3f} ({batch_score_h:.2f}/{full_score:.2f}); average {current_avg_acc_h:.3f}'.format(
                            i, len(val_loader),
                            loss=loss.item(), batch_acc=batch_acc, batch_oks=batch_oks,
                            batch_score=batch_score, full_score=full_score,
                            current_avg_acc=current_avg_acc,
                            batch_acc_h=batch_acc_h, batch_score_h=batch_score_h,
                            current_avg_acc_h=current_avg_acc_h)

                    print(msg)

                    prefix_img = '{}_epoch{}_batch{}'.format(
                        os.path.join(output_dir, 'image', 'val'), epoch, i)
                    batch_size = inputs.shape[0]
                    show_id = 0  # random.randint(0, batch_size - 1)

                    target_coords = target_coords.detach().cpu()
                    save_sample_image(inputs[show_id], target_coords[show_id],
                                      fine_preds[show_id], target_weight[show_id], prefix_img)

    return perf_pck, perf_pckh, perf_oks, perf_loss
