import torch
import torch.nn as nn
from utils.decode import Decoder
from utils.ctc_loss import get_ctc_loss
from utils.evaluation import *
from reader import Reader
from tqdm import tqdm


class TemporalFusion(nn.Module):
    def __init__(self, conv_k, pool_k):
        super(TemporalFusion, self).__init__()
        self.conv = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=conv_k, padding='same')
        self.pooling = nn.MaxPool1d(kernel_size=pool_k, stride=pool_k, padding=int(pool_k/2))

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.conv(x)
        x = self.pooling(x)
        return x


class CSLR(nn.Module):
    def __init__(self, spatio_dim, num_classes, hidden_dim, decoder):
        super(CSLR, self).__init__()

        self.conv2d = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.conv2d_fc = nn.Linear(1000, spatio_dim)
        self.conv1d = TemporalFusion(5, 2)
        self.conv1d_fc = nn.Linear(spatio_dim, num_classes)
        self.lstm = nn.LSTM(input_size=num_classes, hidden_size=hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        self.decoder = decoder

    def forward(self, videos, valid_lengths, phase):
        batch, max_len, C, H, W = videos.shape
        inputs = videos.reshape(batch * max_len, C, H, W)
        framewise_features = self.conv2d(inputs)
        framewise_features = self.conv2d_fc(framewise_features)

        # TODO: 要补成0
        framewise_features = framewise_features.reshape(batch, max_len, -1)
        spatio_dim = framewise_features.shape[2]
        for i in range(batch):  # 检查此处维度
            framewise_features[i, int(valid_lengths[i]):, :] = torch.zeros(((max_len - valid_lengths[i]).item(), spatio_dim))
        framewise_features = framewise_features.permute(0, 2, 1)
        spatio_temporal = self.conv1d(framewise_features)
        spatio_temporal = spatio_temporal.permute(0, 2, 1)
        spatio_temporal = self.conv1d_fc(spatio_temporal)
        # batch, len, dim

        # TODO:计算有效长度
        valid_len = torch.Tensor([vlg / 4 for vlg in valid_lengths]).type(torch.int32)

        # lstm处mask
        packed_emb = nn.utils.rnn.pack_padded_sequence(spatio_temporal, valid_len, batch_first=True,
                                                       enforce_sorted=False)
        alignments, _ = self.lstm(packed_emb)
        alignments, _ = nn.utils.rnn.pad_packed_sequence(alignments, batch_first=True)

        alignments = self.fc(alignments)  # 有mask
        # batch, len , num_classes

        if phase == "predict":
            outputs = self.decoder.max_decode(alignments, valid_len)
            return outputs  # list of tensors

        if phase == "train":
            return alignments, valid_len

        # TODO: ctc loss, alignment proposal
        if phase == "get_feature":
            pass


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)


def train_model(model, mode, prefix, data_path, gloss_dict, epochs, batch, lr, alpha, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  #

    training_data = Reader(prefix, data_path, mode, gloss_dict, batch)
    total_batch = training_data.get_num_instances() / batch

    for i in tqdm(range(epochs)):
        if i > 0 and i % total_batch == 0:
            training_data = Reader(prefix, data_path, mode, gloss_dict, batch)
        model.train()
        videos, valid_len, outputs, valid_output_len = next(training_data.iterate())
        videos, valid_len, outputs, valid_output_len = videos.to(device), valid_len.to(device), outputs.to(device), valid_output_len.to(device)
        alignments, valid_len = model(videos, valid_len, 'train')
        loss = get_ctc_loss(alignments, valid_len, outputs, valid_output_len)
        loss = loss.mean()
        # zero grad, backwards, step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

    save_model(model, path)


def evaluate(model, mode, prefix, data_path, gloss_dict, batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_data = Reader(prefix, data_path, mode, gloss_dict, batch)
    total_num = test_data.get_num_instances()
    total_distance, total_length = 0, 0
    for i in tqdm(range(int(total_num/batch))):
        videos, valid_len, labels, valid_output_len = next(test_data.iterate())
        videos, valid_len, labels, valid_output_len = videos.to(device), valid_len.to(device), labels.to(
            device), valid_output_len.to(device)
        outputs = model(videos, valid_len, 'predict')
        wer, distance, length = batch_evaluation(outputs, labels, valid_output_len)
        total_length += length
        total_distance += distance
        print(total_distance/total_length)


