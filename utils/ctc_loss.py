import torch
import torch.nn as nn


def get_ctc_loss(alignments, valid_len, outputs, valid_output_len, blank_id=1295):
    # alignments: batch, max_len, num_classes
    # outputs: batch, max_output_len, num_classes

    ctc_loss = nn.CTCLoss(blank=blank_id, reduction='none', zero_infinity=True)
    log_probs = alignments.permute(1, 0, 2).log_softmax(2).requires_grad_()  # max_len, batch, num_classes 是否要detach
    _, target = torch.max(outputs, dim=2)
    loss = ctc_loss(log_probs, target, valid_len, valid_output_len)
    return loss


def get_alignment_proposal(alignments, valid_len, outputs, valid_output_len, blank_id=1296):
    pass
