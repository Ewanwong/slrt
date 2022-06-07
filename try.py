import glob
import os
import cv2
import pandas

path = "sample_data/annotations/manual/test.corpus.csv"
a = pandas.read_csv(path, header=0, names=['data'])
print(a['data'][0].split('|'))

idx, file_id, signer, glosses = a['data'][0].split('|')
folder = file_id.split('/')[1]
print(folder)
print(glosses.split(' '))


path = 'sample_data/features/fullFrame-210x260px/dev/'


idx_paths = os.path.join(path, '*', '*')
path = glob.glob(idx_paths)[0]

img_path = os.path.join(path, '*')
print(cv2.imread(glob.glob(img_path)[0]))
print(glob.glob(img_path)[0].split('\\'))


