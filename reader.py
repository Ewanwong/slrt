import os
import cv2
import logging
import torch
import numpy as np
import random
import pickle


class Reader:
    def __init__(self, prefix, mode):
        self.prefix = prefix
        self.mode = mode
        dict_path = f'{mode}.pkl'
        with open(dict_path, 'rb') as f:
            self.list = pickle.load(f)
        print(self.list)


a = Reader('', 'train')
