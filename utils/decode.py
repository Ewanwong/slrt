import torch
import pickle

class Decoder:
    def __init__(self, num_classes, gloss_dict, search_mode, blank_token='<BLANK>'):
        self.num_classes = num_classes
        with open(gloss_dict, 'rb') as f:
            self.gloss_dict = pickle.load(f)
        self.search_mode = search_mode
        self.blank_id = self.gloss_dict[blank_token]

    def max_decode(self, alignments, valid_len):
        outputs = []
        for batch_id in range(alignments.shape[0]):
            alignment = alignments[batch_id, :valid_len[batch_id], :]

            # alignment: length * num_classes
            _, max_alignment = torch.max(alignment, dim=1)

            # max_alignment: shape = length
            output = []
            if max_alignment[0].item() != self.blank_id:
                output.append(max_alignment[0].item())
            for i in range(1, max_alignment.shape[0]):
                if max_alignment[i].item() != self.blank_id and max_alignment[i].item() != max_alignment[i-1].item():
                    output.append(max_alignment[i].item())
                else:
                    continue
            outputs.append(torch.Tensor(output))
        return outputs  # list of tensors

