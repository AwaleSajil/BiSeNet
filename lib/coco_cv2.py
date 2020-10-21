#!/usr/bin/python
# -*- encoding: utf-8 -*-


#mean and std for /gdrive/My Drive/COCO_small_datasets/smallCoco_v3_full.zip dataset is 
# Mean:  tensor([0.3507, 0.3343, 0.3084])
# Std:  tensor([0.3043, 0.2962, 0.2929])

import sys
import os
import os.path as osp
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.base_dataset import BaseDataset, TransformationTrain, TransformationVal


labels_info = [{'name': 'unlabeled', 'ignoreInEval': False, 'id': 0, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'person', 'ignoreInEval': False, 'id': 1, 'trainId': 0, 'color': (215, 0, 0)}, {'name': 'bicycle', 'ignoreInEval': False, 'id': 2, 'trainId': 1, 'color': (140, 60, 255)}, {'name': 'car', 'ignoreInEval': False, 'id': 3, 'trainId': 2, 'color': (2, 136, 0)}, {'name': 'motorcycle', 'ignoreInEval': False, 'id': 4, 'trainId': 3, 'color': (0, 172, 199)}, {'name': 'airplane', 'ignoreInEval': False, 'id': 5, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'bus', 'ignoreInEval': False, 'id': 6, 'trainId': 4, 'color': (152, 255, 0)}, {'name': 'train', 'ignoreInEval': False, 'id': 7, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'truck', 'ignoreInEval': False, 'id': 8, 'trainId': 5, 'color': (255, 127, 209)}, {'name': 'boat', 'ignoreInEval': False, 'id': 9, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'traffic', 'ignoreInEval': False, 'id': 10, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'fire', 'ignoreInEval': False, 'id': 11, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'street', 'ignoreInEval': False, 'id': 12, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'stop', 'ignoreInEval': False, 'id': 13, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'parking', 'ignoreInEval': False, 'id': 14, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'bench', 'ignoreInEval': False, 'id': 15, 'trainId': 6, 'color': (108, 0, 79)}, {'name': 'bird', 'ignoreInEval': False, 'id': 16, 'trainId': 7, 'color': (255, 165, 48)}, {'name': 'cat', 'ignoreInEval': False, 'id': 17, 'trainId': 8, 'color': (0, 0, 157)}, {'name': 'dog', 'ignoreInEval': False, 'id': 18, 'trainId': 9, 'color': (134, 112, 104)}, {'name': 'horse', 'ignoreInEval': False, 'id': 19, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'sheep', 'ignoreInEval': False, 'id': 20, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'cow', 'ignoreInEval': False, 'id': 21, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'elephant', 'ignoreInEval': False, 'id': 22, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'bear', 'ignoreInEval': False, 'id': 23, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'zebra', 'ignoreInEval': False, 'id': 24, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'giraffe', 'ignoreInEval': False, 'id': 25, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'hat', 'ignoreInEval': False, 'id': 26, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'backpack', 'ignoreInEval': False, 'id': 27, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'umbrella', 'ignoreInEval': False, 'id': 28, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'shoe', 'ignoreInEval': False, 'id': 29, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'eye', 'ignoreInEval': False, 'id': 30, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'handbag', 'ignoreInEval': False, 'id': 31, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'tie', 'ignoreInEval': False, 'id': 32, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'suitcase', 'ignoreInEval': False, 'id': 33, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'frisbee', 'ignoreInEval': False, 'id': 34, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'skis', 'ignoreInEval': False, 'id': 35, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'snowboard', 'ignoreInEval': False, 'id': 36, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'sports', 'ignoreInEval': False, 'id': 37, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'kite', 'ignoreInEval': False, 'id': 38, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'baseball', 'ignoreInEval': False, 'id': 39, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'baseball', 'ignoreInEval': False, 'id': 40, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'skateboard', 'ignoreInEval': False, 'id': 41, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'surfboard', 'ignoreInEval': False, 'id': 42, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'tennis', 'ignoreInEval': False, 'id': 43, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'bottle', 'ignoreInEval': False, 'id': 44, 'trainId': 12, 'color': (0, 253, 207)}, {'name': 'plate', 'ignoreInEval': False, 'id': 45, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'wine', 'ignoreInEval': False, 'id': 46, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'cup', 'ignoreInEval': False, 'id': 47, 'trainId': 13, 'color': (188, 183, 255)}, {'name': 'fork', 'ignoreInEval': False, 'id': 48, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'knife', 'ignoreInEval': False, 'id': 49, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'spoon', 'ignoreInEval': False, 'id': 50, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'bowl', 'ignoreInEval': False, 'id': 51, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'banana', 'ignoreInEval': False, 'id': 52, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'apple', 'ignoreInEval': False, 'id': 53, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'sandwich', 'ignoreInEval': False, 'id': 54, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'orange', 'ignoreInEval': False, 'id': 55, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'broccoli', 'ignoreInEval': False, 'id': 56, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'carrot', 'ignoreInEval': False, 'id': 57, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'hot', 'ignoreInEval': False, 'id': 58, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'pizza', 'ignoreInEval': False, 'id': 59, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'donut', 'ignoreInEval': False, 'id': 60, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'cake', 'ignoreInEval': False, 'id': 61, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'chair', 'ignoreInEval': False, 'id': 62, 'trainId': 10, 'color': (0, 73, 66)}, {'name': 'couch', 'ignoreInEval': False, 'id': 63, 'trainId': 18, 'color': (220, 179, 175)}, {'name': 'potted', 'ignoreInEval': False, 'id': 64, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'bed', 'ignoreInEval': False, 'id': 65, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'mirror', 'ignoreInEval': False, 'id': 66, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'dining', 'ignoreInEval': False, 'id': 67, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'window', 'ignoreInEval': False, 'id': 68, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'desk', 'ignoreInEval': False, 'id': 69, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'toilet', 'ignoreInEval': False, 'id': 70, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'door', 'ignoreInEval': False, 'id': 71, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'tv', 'ignoreInEval': False, 'id': 72, 'trainId': 11, 'color': (79, 42, 0)}, {'name': 'laptop', 'ignoreInEval': False, 'id': 73, 'trainId': 14, 'color': (149, 180, 122)}, {'name': 'mouse', 'ignoreInEval': False, 'id': 74, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'remote', 'ignoreInEval': False, 'id': 75, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'keyboard', 'ignoreInEval': False, 'id': 76, 'trainId': 15, 'color': (192, 4, 185)}, {'name': 'cell', 'ignoreInEval': False, 'id': 77, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'microwave', 'ignoreInEval': False, 'id': 78, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'oven', 'ignoreInEval': False, 'id': 79, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'toaster', 'ignoreInEval': False, 'id': 80, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'sink', 'ignoreInEval': False, 'id': 81, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'refrigerator', 'ignoreInEval': False, 'id': 82, 'trainId': 16, 'color': (37, 102, 162)}, {'name': 'blender', 'ignoreInEval': False, 'id': 83, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'book', 'ignoreInEval': False, 'id': 84, 'trainId': 17, 'color': (40, 0, 65)}, {'name': 'clock', 'ignoreInEval': False, 'id': 85, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'vase', 'ignoreInEval': False, 'id': 86, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'scissors', 'ignoreInEval': False, 'id': 87, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'teddy', 'ignoreInEval': False, 'id': 88, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'hair', 'ignoreInEval': False, 'id': 89, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'toothbrush', 'ignoreInEval': False, 'id': 90, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'hair', 'ignoreInEval': False, 'id': 91, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'banner', 'ignoreInEval': False, 'id': 92, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'blanket', 'ignoreInEval': False, 'id': 93, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'branch', 'ignoreInEval': False, 'id': 94, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'bridge', 'ignoreInEval': False, 'id': 95, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'building', 'ignoreInEval': False, 'id': 96, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'bush', 'ignoreInEval': False, 'id': 97, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'cabinet', 'ignoreInEval': False, 'id': 98, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'cage', 'ignoreInEval': False, 'id': 99, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'cardboard', 'ignoreInEval': False, 'id': 100, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'carpet', 'ignoreInEval': False, 'id': 101, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'ceiling', 'ignoreInEval': False, 'id': 102, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'ceiling', 'ignoreInEval': False, 'id': 103, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'cloth', 'ignoreInEval': False, 'id': 104, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'clothes', 'ignoreInEval': False, 'id': 105, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'clouds', 'ignoreInEval': False, 'id': 106, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'counter', 'ignoreInEval': False, 'id': 107, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'cupboard', 'ignoreInEval': False, 'id': 108, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'curtain', 'ignoreInEval': False, 'id': 109, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'desk', 'ignoreInEval': False, 'id': 110, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'dirt', 'ignoreInEval': False, 'id': 111, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'door', 'ignoreInEval': False, 'id': 112, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'fence', 'ignoreInEval': False, 'id': 113, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'floor', 'ignoreInEval': False, 'id': 114, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'floor', 'ignoreInEval': False, 'id': 115, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'floor', 'ignoreInEval': False, 'id': 116, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'floor', 'ignoreInEval': False, 'id': 117, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'floor', 'ignoreInEval': False, 'id': 118, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'flower', 'ignoreInEval': False, 'id': 119, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'fog', 'ignoreInEval': False, 'id': 120, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'food', 'ignoreInEval': False, 'id': 121, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'fruit', 'ignoreInEval': False, 'id': 122, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'furniture', 'ignoreInEval': False, 'id': 123, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'grass', 'ignoreInEval': False, 'id': 124, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'gravel', 'ignoreInEval': False, 'id': 125, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'ground', 'ignoreInEval': False, 'id': 126, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'hill', 'ignoreInEval': False, 'id': 127, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'house', 'ignoreInEval': False, 'id': 128, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'leaves', 'ignoreInEval': False, 'id': 129, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'light', 'ignoreInEval': False, 'id': 130, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'mat', 'ignoreInEval': False, 'id': 131, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'metal', 'ignoreInEval': False, 'id': 132, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'mirror', 'ignoreInEval': False, 'id': 133, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'moss', 'ignoreInEval': False, 'id': 134, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'mountain', 'ignoreInEval': False, 'id': 135, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'mud', 'ignoreInEval': False, 'id': 136, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'napkin', 'ignoreInEval': False, 'id': 137, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'net', 'ignoreInEval': False, 'id': 138, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'paper', 'ignoreInEval': False, 'id': 139, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'pavement', 'ignoreInEval': False, 'id': 140, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'pillow', 'ignoreInEval': False, 'id': 141, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'plant', 'ignoreInEval': False, 'id': 142, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'plastic', 'ignoreInEval': False, 'id': 143, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'platform', 'ignoreInEval': False, 'id': 144, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'playingfield', 'ignoreInEval': False, 'id': 145, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'railing', 'ignoreInEval': False, 'id': 146, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'railroad', 'ignoreInEval': False, 'id': 147, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'river', 'ignoreInEval': False, 'id': 148, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'road', 'ignoreInEval': False, 'id': 149, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'rock', 'ignoreInEval': False, 'id': 150, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'roof', 'ignoreInEval': False, 'id': 151, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'rug', 'ignoreInEval': False, 'id': 152, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'salad', 'ignoreInEval': False, 'id': 153, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'sand', 'ignoreInEval': False, 'id': 154, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'sea', 'ignoreInEval': False, 'id': 155, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'shelf', 'ignoreInEval': False, 'id': 156, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'sky', 'ignoreInEval': False, 'id': 157, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'skyscraper', 'ignoreInEval': False, 'id': 158, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'snow', 'ignoreInEval': False, 'id': 159, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'solid', 'ignoreInEval': False, 'id': 160, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'stairs', 'ignoreInEval': False, 'id': 161, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'stone', 'ignoreInEval': False, 'id': 162, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'straw', 'ignoreInEval': False, 'id': 163, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'structural', 'ignoreInEval': False, 'id': 164, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'table', 'ignoreInEval': False, 'id': 165, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'tent', 'ignoreInEval': False, 'id': 166, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'textile', 'ignoreInEval': False, 'id': 167, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'towel', 'ignoreInEval': False, 'id': 168, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'tree', 'ignoreInEval': False, 'id': 169, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'vegetable', 'ignoreInEval': False, 'id': 170, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'wall', 'ignoreInEval': False, 'id': 171, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'wall', 'ignoreInEval': False, 'id': 172, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'wall', 'ignoreInEval': False, 'id': 173, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'wall', 'ignoreInEval': False, 'id': 174, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'wall', 'ignoreInEval': False, 'id': 175, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'wall', 'ignoreInEval': False, 'id': 176, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'wall', 'ignoreInEval': False, 'id': 177, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'water', 'ignoreInEval': False, 'id': 178, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'waterdrops', 'ignoreInEval': False, 'id': 179, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'window', 'ignoreInEval': False, 'id': 180, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'window', 'ignoreInEval': False, 'id': 181, 'trainId': 255, 'color': [0, 0, 0]}, {'name': 'wood', 'ignoreInEval': False, 'id': 182, 'trainId': 255, 'color': [0, 0, 0]}]




class Coco(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Coco, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 19
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

        # self.to_tensor = T.ToTensor(
        #     mean=(0.3257, 0.3690, 0.3223), # city, rgb
        #     std=(0.2112, 0.2148, 0.2115),
        # )

        self.to_tensor = T.ToTensor(
            mean=(0.3507, 0.3343, 0.3084), # coco-customdataset, rgb
            std=(0.3043, 0.2962, 0.2929),
        )
        


def get_mean_std(loader):
    #var[x] = E[x**2] - E[x]**2
    channels_sum, channels_squared_sum, num_batches = 0,0,0
    for data, _ in loader:
        print("Data Shape:", data.size())
        channels_sum += torch.mean(data, dim = [0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    print("Mean: ", mean)
    print("Std: ", std)
    return mean, std

def get_data_loader(datapth, annpath, ims_per_gpu, scales, cropsize, max_iter=None, mode='train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(scales, cropsize)
        batchsize = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal(cropsize)
        batchsize = ims_per_gpu
        shuffle = False
        drop_last = False

    ds = Coco(datapth, annpath, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = ims_per_gpu * dist.get_world_size() * max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl



if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = Coco('./data/', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
