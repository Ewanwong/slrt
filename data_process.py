import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import glob
from typing import List
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as KNN # classification models
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix # metrics
import seaborn as sn # plot

import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

loader = transforms.Compose([
    transforms.ToTensor()
])

train_path = "hand_gestures/train"
test_path = "hand_gestures/test"


def image_to_tensor(file_path: str):
    """

    :param file_path: image directory
    :return: image shape: 3 * 480 * 640
    """
    image = Image.open(file_path).convert("RGB")
    image = loader(image)
    return image


def get_images_path(file_path: str):
    images_path = []
    sub_folders = os.listdir(file_path)
    for sub_folder in sub_folders:
        path = os.path.join(file_path, sub_folder)
        for image in os.listdir(path):
            images_path.append(os.path.join(path, image))
    return images_path


def get_dataset(file_path: str):
    data = []
    labels = []
    images_path = get_images_path(file_path)
    for image_path in images_path:
        data.append(image_to_tensor(image_path))
        labels.append(image_path.split('\\')[-2])
    return data, labels


data, labels = get_dataset(train_path)
print(len(data))
print(len(labels))
print(data[0])
print(data[0].shape)