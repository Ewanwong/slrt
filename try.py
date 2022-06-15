import glob
import os
import cv2
import pandas
import pickle

path = 'dev.pkl'

with open(path, 'rb') as f:
    diction = pickle.load(f)

print(diction.keys())
print(diction['features'][0])





















