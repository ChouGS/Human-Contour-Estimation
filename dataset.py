import torch
import numpy as np
import cv2

import json
import os
import pdb

from augmentation import augmentation


class ContourDataset():
    def __init__(self, cfg, is_train):
        self.is_train = is_train
        self.convert = np.array([1., 1., 0.])
        self.cfg = cfg

        if is_train:
            self.size = self.cfg.TRAIN.SIZE
            self.imagelist = []
            self.annolist = []
            for i, iname in enumerate(
                    os.listdir(os.path.join(cfg.DATASET.ROOT, 'images/train/'))):
                if len(self.imagelist) == self.size:
                    break
                aname = iname.rstrip('.jpg') + '.json'
                try:
                    assert os.path.exists(os.path.join(cfg.DATASET.ROOT, 'annotations/train/') +
                                          aname)
                except AssertionError:
                    continue
                self.imagelist.append(os.path.join(cfg.DATASET.ROOT, 'images/train/') + iname)
                self.annolist.append(os.path.join(cfg.DATASET.ROOT, 'annotations/train/') + aname)

        else:
            self.size = self.cfg.TEST.SIZE
            self.imagelist = []
            self.annolist = []
            for i, iname in enumerate(
                    os.listdir(os.path.join(cfg.DATASET.ROOT, 'images/val/'))):
                if len(self.imagelist) == self.size:
                    break
                aname = iname.rstrip('.jpg') + '.json'
                try:
                    assert os.path.exists(os.path.join(cfg.DATASET.ROOT, 'annotations/val/') +
                                          aname)
                except AssertionError:
                    continue
                self.imagelist.append(os.path.join(cfg.DATASET.ROOT, 'images/val/') + iname)
                self.annolist.append(os.path.join(cfg.DATASET.ROOT, 'annotations/val/') + aname)

    def __getitem__(self, index):
        # if self.is_train:
        #     inputs = torch.zeros(1, 3, self.cfg.MODEL.IMAGE_SIZE[0],
        #                          self.cfg.MODEL.IMAGE_SIZE[1])
        #     target_coords = torch.zeros(1, self.cfg.MODEL.NUM_JOINTS, 2)
        #     target_weight = torch.zeros(1, self.cfg.MODEL.NUM_JOINTS, 1)

        #     img = cv2.imread(self.imagelist[index])
        #     anno = json.load(open(self.annolist[index], 'r'))

        #     flag = np.array(anno['flag'].split(' '))

        #     weight_np = np.expand_dims(self.convert[flag.astype(int)], 1)

        #     coordinate = np.array(anno['coordinate'].split(' '))
        #     coords_np = coordinate.astype(float).reshape(
        #         self.cfg.MODEL.NUM_JOINTS, 2)

        #     bbox = anno['human_box']

        #     image, target, weight = augmentation(img,
        #                                          coords_np,
        #                                          weight_np,
        #                                          bbox,
        #                                          aug=False)
        #     target_weight[0] = torch.from_numpy(weight)
        #     target_coords[0] = torch.from_numpy(target)
        #     inputs[0] = torch.from_numpy(image.transpose((2, 0, 1)) / 255)

        #     # image, target, weight = augmentation(img, coords_np, weight_np,
        #     #                                      bbox)
        #     # target_weight[1] = torch.from_numpy(weight)
        #     # target_coords[1] = torch.from_numpy(target)
        #     # inputs[1] = torch.from_numpy(image.transpose((2, 0, 1)) / 255)

        #     # image, target, weight = augmentation(img, coords_np, weight_np,
        #     #                                      bbox)
        #     # target_weight[2] = torch.from_numpy(weight)
        #     # target_coords[2] = torch.from_numpy(target)
        #     # inputs[2] = torch.from_numpy(image.transpose((2, 0, 1)) / 255)

        #     # image, target, weight = augmentation(img, coords_np, weight_np,
        #     #                                      bbox)
        #     # target_weight[3] = torch.from_numpy(weight)
        #     # target_coords[3] = torch.from_numpy(target)
        #     # inputs[3] = torch.from_numpy(image.transpose((2, 0, 1)) / 255)
        #     # if target_weight[0].sum() == 0 or target_weight[1].sum() == 0 or \
        #     #    target_weight[2].sum() == 0 or target_weight[3].sum() == 0:
        #     #     pdb.set_trace()

        # else:
        inputs = torch.zeros(1, 3, self.cfg.MODEL.IMAGE_SIZE[0],
                             self.cfg.MODEL.IMAGE_SIZE[1])
        target_coords = torch.zeros(1, self.cfg.MODEL.NUM_JOINTS, 2)
        target_weight = torch.zeros(1, self.cfg.MODEL.NUM_JOINTS, 1)

        img = cv2.imread(self.imagelist[index])
        anno = json.load(open(self.annolist[index], 'r'))

        flag = np.array(anno['flag'].split(' '))

        weight_np = np.expand_dims(self.convert[flag.astype(int)], 1)

        coordinate = np.array(anno['coordinate'].split(' '))
        coords_np = coordinate.astype(float).reshape(
            self.cfg.MODEL.NUM_JOINTS, 2)

        bbox = anno['human_box']

        image, target, weight = augmentation(img,
                                             coords_np,
                                             weight_np,
                                             bbox,
                                             aug=False)
        target_weight[0] = torch.from_numpy(weight)
        target_coords[0] = torch.from_numpy(target)
        inputs[0] = torch.from_numpy(image.transpose((2, 0, 1)) / 255)

        return inputs, target_coords, target_weight

    def __len__(self):
        return self.size
