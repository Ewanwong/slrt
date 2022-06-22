from model import *
from utils.decode import Decoder

decoder = Decoder(1296, gloss_dict='gloss_dict.pkl', search_mode='max')
model = CSLR(1024, 1296, 512, decoder)
prefix = '../phoenix2014_data/features/fullFrame-224x224px'
train_model(model, 'train', prefix, '../data.pkl', 'gloss_dict.pkl', 10, 2, 1e-5, 1e-2, 'model.pt')
