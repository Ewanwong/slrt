import os
import cv2
import logging
import torch
import numpy as np
import random
import pickle
from utils.img2vec import video2vector
from utils.augmentation import *


class Reader:
    def __init__(self, prefix, data_path, mode, gloss_dict, batch_size=2, distortion=True, do_shuffle=True):
        self.prefix = prefix
        self.mode = mode
        dict_path = f'{data_path}.pkl'
        with open(dict_path, 'rb') as f:
            mode_dict = pickle.load(f)
        self.data = mode_dict[mode]
        with open(gloss_dict, 'rb') as f:
            self.gloss_dict = pickle.load(f)
        self.num_classes = len(self.gloss_dict.keys())
        self.num_instances = len(self.data.keys())
        self.batch_size = batch_size
        self.distortion = distortion
        self.do_shuffle = do_shuffle




        # TODO: 取样求特征值分解
        # folders =
        # px = self.imgs[sorted(np.random.choice(799006, 1000, replace=False))]
        # px = px.reshape((-1, 3)) / 255.
        # px -= px.mean(axis=0)
        # self.eig_value, self.eig_vector = np.linalg.eig(np.cov(px.T))


        self.data_aug = self.transform()

    def get_num_instances(self):
        return self.num_instances

    def iterate(self):
        index = list(range(self.num_instances))
        if self.do_shuffle:
            random.shuffle(index)

        for i in range(0, self.num_instances, self.batch_size):
            indices = index[i: i + self.batch_size]
            batch_size = len(indices)  # for end of the iteration

            video_paths = [os.path.join(self.prefix, self.data[k]['paths']) for k in indices]
            labels = [self.data[k]['label'] for k in indices]

            videos = []
            outputs = []
            for video_path, label in zip(video_paths, labels):
                video = video2vector(video_path)
                video, label = self.data_aug(video, label, self.num_classes)

                video = video.float() / 127.5 - 1

                videos.append(video)
                outputs.append(label)

            max_len = max([len(video) for video in videos])
            # mask = []
            # TODO:不用mask，用长度即可
            mask_len = [max_len - len(video) for video in videos]
            valid_len = torch.Tensor([len(video) for video in videos])
            for i in range(batch_size):
                videos[i] = torch.concat([videos[i], torch.zeros((mask_len[i], 224, 224))])
                # mask.append(torch.concat([torch.ones(len(videos[i]), 1), torch.zeros(mask_len[i]), 1]))

            videos = torch.stack(videos, dim=0)

            # mask = torch.stack(mask, dim=0)

            valid_output_len = torch.Tensor([len(output) for output in outputs])
            max_output_len = max(valid_output_len)
            for i in range(batch_size):
                outputs[i] = torch.concat(
                    [outputs[i], torch.zeros((max_output_len - valid_output_len[i], self.num_classes, 1))])
            outputs = torch.stack(outputs, dim=0)

            # TODO: 改为输出有效长度，对output对齐
            yields = [videos, valid_len, outputs, valid_output_len]
            # videos: tensor, batch_size, max_len, C, W, H
            # len: list
            # outputs: batch, len, num_classes
            yield yields

            # video = 读图片+noise
            # 转成torch.tensor(with temporal scaling)
            # 加mask

    def transform(self):
        if self.mode == 'train':
            print("Apply training transform.")
            return Compose([
                ImageAugmentation(trans=0.1, color_dev=0.1, distortion=True),
                ToTensor(),
                TemporalRescale(temp_scaling=0.2)
            ])
        else:
            print("Apply test transform")
            return Compose([
                ToTensor()
            ])

