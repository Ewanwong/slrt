from model import *

prefix = '../phoenix2014_data/features/fullFrame-224x224px'
decoder = Decoder(1296, gloss_dict='gloss_dict.pkl', search_mode='max')
model = CSLR(1024, 1296, 512, decoder)

load_model(model, '../model.pt')

evaluate(model, 'dev', prefix, '../data.pkl', 'gloss_dict.pkl', 2)