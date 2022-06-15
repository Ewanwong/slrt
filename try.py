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



















>>>>>>> b8af45fcf9b2f1b07b268c6bd0eed11378d82796


