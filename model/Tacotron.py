import torch
import torch.nn as nn
from model.modules import PreNet, CBHG, AttentionRNN, Attention, DecoderRNN
from utils import mask_from_lengths


class Encoder(nn.Module):

    def __init__(self, n_phoneme, emb_dim, k, hidden_dims):

        super(Encoder, self).__init__()
        self.character_embedding = nn.Embedding(n_phoneme, emb_dim)
        self.prenet = PreNet(emb_dim)
        self.cbhg = CBHG(k, hidden_dims)


    def forward(self, x):

        # character embedding -> [B, text_len, 128]
        x = self.character_embedding(x)

        # prenet -> [B, text_len, 128]
        x = self.prenet(x)

        # cbhg -> [B, text_len, 256]
        x = self.cbhg(x)

        return x
    

class Decoder(nn.Module):
    """
    Tacotron paper - 3.3 Decoder

    input : encoder output              [B, text_len, 256]
    input : target melspectrogram       [B, seq_len, 80]
    input : text sequence length        [B, 1]

    output: predicted melspectrogram    [B, seq_len, 80]
    output: predicted spectrogram       [B, seq_len, 1025]
    output: predicted alignment         [B, text_len, seq_len]
    """

    def __init__(self, n_mels, r, k, hidden_dims):

        super(Decoder, self).__init__()
        self.max_T          = 200
        self.n_mels         = n_mels
        self.r              = r

        # layers
        self.prenet         = PreNet(n_mels * r)
        self.attention_rnn  = AttentionRNN()
        self.attention      = Attention()
        self.decoder_rnn    = DecoderRNN()
        self.postnet        = CBHG(k, hidden_dims)

        # linears
        self.pre_cbhg       = nn.Linear(80, 128, bias=False)
        self.mel_linear     = nn.Linear(256, self.n_mels * self.r)
        self.lin_linear     = nn.Linear(256, 1025)


    def forward(self, z, y, len):

        # prepare decoding
        B = z.size(0)
        self.max_T, y = self.prepare_melframes(y)
        mask = mask_from_lengths(z, len) if len is not None else None

        # initial variables
        input_frame      = z.new_zeros(B, self.n_mels * self.r)
        attn_rnn_hidden  = z.new_zeros(B, 256)
        dec_rnn_hiddens  = [z.new_zeros(B, 256) for _ in range(2)]
        attn_out         = z.new_zeros(B, 256)

        # Prediction results (container)
        pred_mel_frames = []
        pred_alignments = []
        t = 0

        while True:

            # decoder prenet -> [B, 128]
            prenet_out = self.prenet(input_frame)

            # attention rnn -> [B, 256]
            attn_rnn_hidden = self.attention_rnn(prenet_out, attn_out, attn_rnn_hidden)

            # attention -> attention context [B, 256], alignment [B, text_len]
            attn_out, alignment = self.attention(attn_rnn_hidden, z, mask)
            pred_alignments.append(alignment)

            # decoder rnn -> decoder rnn out [B, 256], dec_rnn_hiddens [2, B, 256]
            decoder_rnn_out, dec_rnn_hiddens = self.decoder_rnn(attn_out, attn_rnn_hidden, dec_rnn_hiddens)
            
            # Make reduction factor (r) number of mel frames -> [B, 80 * r]
            r_mel_frames = self.mel_linear(decoder_rnn_out)
            pred_mel_frames.append(r_mel_frames)

            t += 1

            if self.is_end_of_timestep(t, r_mel_frames):
                break

            input_frame = y[t - 1] if self.training else r_mel_frames


        # Concat all pred_alignments -> [B, seq_len // r, text_len]
        # Concat all mel frames -> [B, seq_len // r, 80 * r]
        pred_alignments = torch.stack(pred_alignments).transpose(0, 1)
        pred_mel_frames = torch.stack(pred_mel_frames).transpose(0, 1).contiguous()

        # predicted mel spectrograms -> [B, seq_len, 80]
        mel_pred = pred_mel_frames.view(B, -1, 80)

        # Post-net 처리를 거친 후 선형 레이어를 통해 최종 스펙트로그램 생성
        a = self.pre_cbhg(mel_pred)

        b = self.postnet(a)

        c = self.lin_linear(b)

        lin_pred = c

        return mel_pred, lin_pred, pred_alignments


    def is_end_of_timestep(self, t, melframes):

        # inference
        if self.training is False:
            if self.is_end_of_frame(melframes):
                return True
            elif t > self.max_T:
                print("[caution] Mel spectrogram does not seem to be converged.")
                return True
        
        # training
        elif t == self.max_T:
            return True
        
        return False

        
    def is_end_of_frame(self, z):
        return (z < 0.2).all()
    

    def prepare_melframes(self, melspectrogram):
        
        # input  : melspectrogram                       [B, seq_len, 80]
        # output : melframes (using reduction factor)   [seq_len // r, B, 80 * r]

        B, L, M = melspectrogram.size()

        if M == self.n_mels:
            y = melspectrogram.contiguous()
            y = y.view(B, L // self.r, -1)

        assert y.size(2) == M * self.r
        return y.size(1), y.transpose(0, 1)
    

class Tacotron(nn.Module):
    """
    Tacotron model
    """
    
    def __init__(self, hp):

        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.n_phonemes, hp.ch_emb_dim, hp.encoder_K, hp.encoder_projection)
        self.decoder = Decoder(hp.n_mels, hp.reduction_factor, hp.decoder_K, hp.decoder_projection)


    def forward(self, x, y=None, text_len=None):

        x = self.encoder(x)
        return self.decoder(x, y, text_len)