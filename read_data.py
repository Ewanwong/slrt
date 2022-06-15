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
from sklearn import preprocessing
import argparse


# 读所有id 读label 读video

class CSL_dataset(Dataset):
    def __init__(self, prefix='phoenix2014_data', gloss_dict='gloss_dict.pkl', mode='train',
                 input_type='fullFrame-210x260px'):
        super(CSL_dataset, self).__init__()
        self.prefix = prefix
        self.mode = mode
        with open(gloss_dict, 'rb') as f:
            self.gloss_dict = pickle.load(f)

        #self.data_aug = self.transform()
        self.dataset = defaultdict()
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
            labels = glosses.split(' ')  # 是否int
            labels = [self.gloss_dict[gloss] for gloss in labels if len(gloss) > 0]

            self.data[idx][folder]['label'] = labels
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
        video = [img_path for img_path in img_list]
        #video = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]

        idx, folder = video_path.split('/')[-2], video_path.split('/')[-1]
        return idx, folder, video

    def normalize(self):
        id = 0
        for idx, v1 in self.data.items():
            for folder, v2 in v1.items():
                video, label, signer = self.data[idx][folder]['features'], self.data[idx][folder]['label'], \
                                        self.data[idx][folder]['signer']
                self.dataset['ids'] = id
                id += 1
                self.dataset['features'] = video
                self.dataset['label'] = label
                self.dataset['signer'] = signer
                # video, label = self.data_aug(video, label, None)
                #
                # video = video.float() / 127.5 - 1
                # self.data[idx][folder]['features'], self.data[idx][folder]['label'] = video, label
                

    def transform(self):
        if self.mode=='train':
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
        with open(path, 'wb') as f:
            
            pickle.dump(self.dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default='../phoenix2014_data')
    parser.add_argument("--gloss_dict", default='gloss_dict.pkl')
    parser.add_argument("--mode", default="train")
    parser.add_argument("--input_type", default='fullFrame-224x224px')
    parser.add_argument("--save_path", default='data.pkl')
    args = parser.parse_args()

    data_set = CSL_dataset(args.prefix, args.gloss_dict, args.mode, args.input_type)
    data_set.save_to_path(args.save_path)
