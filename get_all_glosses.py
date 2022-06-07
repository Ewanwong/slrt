import os
import pandas


sel = ['dev', 'test', 'train']

gloss_list = []
for i in sel:
    label_path = path = f"sample_data/annotations/manual/{i}.corpus.csv"
    f = pandas.read_csv(label_path, header=0, names=['data'])
    for entry in f['data']:
        idx, file_id, signer, glosses = entry.split('|')
        gloss_list += glosses.split(' ')

gloss_list = list(set(gloss_list))
print(len(gloss_list))