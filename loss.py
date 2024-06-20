import torch.nn as nn

class TacotronLoss():

    def __init__(self):

        self.l1_loss = nn.L1Loss()


    def __call__(self, mel_pred, lin_pred, mel_targ, lin_targ, seq_len):
        
        # prepare mask
        mel_mask = self.get_mask(mel_pred, seq_len)
        lin_mask = self.get_mask(lin_pred, seq_len)

        # masking
        mel_pred = mel_pred * mel_mask
        mel_targ = mel_targ * mel_mask

        lin_pred = lin_pred * lin_mask
        lin_targ = lin_targ * lin_mask

        # calculate l1 loss
        mel_loss = self.l1_loss(mel_pred, mel_targ)
        lin_loss = self.l1_loss(lin_pred, lin_targ)

        # conjugate loss
        return mel_loss * 0.5 + lin_loss * 0.5


    def get_mask(self, pred, seq_len):
        
        # Create mask based on sequence length
        mask = pred.new(pred.permute(0, 2, 1).size()).fill_(1)
        for idx, length in enumerate(seq_len):
            mask[idx, :, length:] = 0
        return mask.permute(0, 2, 1)