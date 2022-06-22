import glob
import os
import cv2
import pandas

from collections import defaultdict
import pickle
import torch
total_dict = {}
modes = ['train', 'dev', 'test']
count = 0
# for mode in modes:
#     total_dict[mode] = defaultdict(dict)
#     with open(f'{mode}.pkl', 'rb') as f:
#         dd = pickle.load(f)
#     for id, feature, label, signer in zip(dd['ids'], dd['features'], dd['label'], dd['signer']):
#         total_dict[mode][id]['paths'] = feature
#         total_dict[mode][id]['label'] = label
#         total_dict[mode][id]['signer'] = signer
#
# with open('data.pkl', 'wb') as f:
# #     pickle.dump(total_dict, f)

with open('data.pkl', 'rb') as f:
     dd = pickle.load(f)
print(dd['train'][1]['paths'])
# for mode in modes:
#     for id, v in dd[mode].items():
#         for path in v['paths']:
#             count += 1
# print(count)

# print(dd['train'][10]['signer'])

# a = list(range(10))
# import random
# random.shuffle(a)
# print(a[6:10])


# import torch
# a = torch.zeros((3, 3, 1))
# print(a[1, :, :])
# print(torch.concat([a, torch.zeros((2, 3, 1))]).shape)




# import torch
# a = torch.Tensor([[1,2,0], [2, 0, 0], [3, 2, 1]])  # 3, 3
# lens = torch.Tensor([2, 1, 3]) # 3, 3, 1
# a = torch.unsqueeze(a, dim=2)
# print(a.shape)
# lstm = torch.nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True,
#                             bidirectional=True)
#
# p = torch.nn.utils.rnn.pack_padded_sequence(a, lens, batch_first=True, enforce_sorted=False)
# print(p.data.shape)
# a, _ = lstm(p)
#
# print(a.data.shape)
# a, _ = torch.nn.utils.rnn.pad_packed_sequence(a, batch_first=True)
# print(a[1])





# import pickle
# with open('gloss_dict.pkl', 'rb') as f:
#     gloss_dict = pickle.load(f)
# print(gloss_dict['<BLANK>'])
#
#
# a = torch.rand((3, 6))
# m, n = torch.max(a, dim=1)
# print(m)
# print(n.shape)
#
#
# print(n[0].item())
#
#
# a = torch.rand((3, 6, 3))
# print(a[1, :3, :].shape)

# a = torch.zeros((3, 1))
# b = torch.zeros((3, 1))
# print(torch.stack([a, b], dim=0).squeeze().shape)
def get_int():
    for i in range(10):

        yield i

a = get_int()
length = len(list(a))

for i in range(20):
    if i % length == 0:
        a = get_int()
    print(next(a))


import pickle

path = 'dev.pkl'

with open(path, 'rb') as f:
    diction = pickle.load(f)

print(diction.keys())
print(diction['features'][0])



















