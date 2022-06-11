import glob
import os
import torch
import sys
import cv2
import time
import tqdm
import numpy as np
import pandas
import pickle
from PIL import Image
from utils import video_augmentation
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


# 读所有id 读label 读video

class CSL_dataset(Dataset):
    def __init__(self, prefix='phoenix2014_data', gloss_dict='gloss_dict.pkl', mode='train', input_type='fullFrame-210x260px', transform_mode=True):
        super(CSL_dataset, self).__init__()
        self.prefix = prefix
        self.mode = mode
        self.transform_mode = transform_mode
        with open(gloss_dict, 'rb') as f:
            self.gloss_dict = pickle.load(f)
        self.data_aug = self.transform()
        self.dataset = []
        self.feature_path = f'{prefix}/features/{input_type}/{mode}/'
        self.label_path = f'{prefix}/annotations/manual/{mode}.corpus.csv'
        self.len_labels = 0
        self.len_features = 0
        self.data = defaultdict(dict)
        self._get_labels()
        self._get_features()
        self.normalize()

    def __len__(self):
        return self.len_labels, self.len_features  # 目前用于sample data

    def __getitem__(self, item):
        return self.dataset[item]

    def _get_labels(self):
        annotations = pandas.read_csv(self.label_path, header=0, names=['data'])
        self.len_labels = len(annotations['data'])
        for i in range(self.len_labels):
            idx, file_id, signer, glosses = annotations['data'][i].split('|')
            folder = file_id.split('/')[1]
            if not self.data[idx]:
                self.data[idx] = defaultdict(dict)
            if not self.data[idx][folder]:
                self.data[idx][folder] = defaultdict(dict)
            self.data[idx][folder]['label'] = glosses.split(' ')  # 是否int
            self.data[idx][folder]['signer'] = signer

    def _get_features(self):
        idx_paths = glob.glob(os.path.join(self.feature_path, '*', '*'))
        self.len_features = len(idx_paths)
        for idx_path in idx_paths:
            idx, folder, video = self._get_video(idx_path)
            self.data[idx][folder]['features'] = video

    def _get_video(self, video_path):
        path = os.path.join(video_path, '*')
        img_list = sorted(glob.glob(path))
        video = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
        idx, folder = video_path.split('\\')[-2], video_path.split('\\')[-1]
        return idx, folder, video

    def normalize(self):
        for idx, v1 in self.data.items():
            for folder, v2 in v1.items():
                video, label, signer = self.data[idx][folder]['features'], self.data[idx][folder]['label'], self.data[idx][folder]['signer']
                video, label = self.data_aug(video, label, None)
                print(label)
                video = video.float() / 127.5 - 1
                self.data[idx][folder]['features'], self.data[idx][folder]['label'] = video, label
                self.dataset.append((video, torch.LongTensor(label), idx, folder, signer))

    def transform(self):
        if self.transform_mode:
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                # video_augmentation.Resize(0.5),
                video_augmentation.ToTensor(),
            ])

    def save_to_path(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for i in self.dataset:
                pickle.dump(i, f)




a = CSL_dataset("sample_data")
a.save_to_path('data.pk')