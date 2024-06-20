import torch
import torch.nn as nn


class Conv1DBN(nn.Module):
    """
    1D-Convolution with batch normalization and activation function.
    """
    
    def __init__(self, in_channel, out_channel, k, bias=False, activation=None):

        super(Conv1DBN, self).__init__()
        self.conv_1d = nn.Conv1d(in_channel, out_channel, k, 1, k//2, bias=bias)
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation = activation


    def forward(self, x):

        # 1d convolution
        x = self.conv_1d(x)
        
        # activation function
        if self.activation is not None:
            x = self.activation(x)

        # batch normalization
        return self.bn(x)


class Highway(nn.Module):
    """
    https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/tacotron.py
    """
    
    def __init__(self, in_size, out_size):

        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):

        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)

    
class PreNet(nn.Module):
    """
    Tacotron paper - 3.2 Encoder

    input : text embedding  [B, text_len, 256]
    output: prenet output   [B, text_len, 128]
    """
    
    def __init__(self, in_dim):

        super(PreNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))

        return x
    

class Conv1DBank(nn.Module):
    """
    Tacotron paper - 3.1 CBHG Module

    Bank of 1D Convolution: convolved with K sets of 1D convolutional filters
    where k-th set contains Ck filters of width k (k = 1, 2, ..., K).

    input : output of prenet                [B, text_len, 128]
    output: bank of 1d convolution output   [B, text_len * K, 128]
    """

    def __init__(self, K):

        super(Conv1DBank, self).__init__()
        self.convolutions = nn.ModuleList(
            [Conv1DBN(128, 128, k, False, nn.ReLU()) 
             for k in range(1, K + 1)])
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)


    def forward(self, x):

        text_len = x.size(1)

        # Transpose for convolution
        x = x.transpose(1, 2)

        # PAPER: The convolution outputs are stacked together
        x = torch.cat([conv(x)[:, :, :text_len] for conv in self.convolutions], dim=1)

        # PAPER: and further max pooled along ...
        x = self.maxpool(x)[:, :, :text_len]
        
        # Transpose for convolution
        return x.transpose(1, 2)
    

class Conv1DProjection(nn.Module):
    """
    Tacotron paper - Table 1. Network architectures

    1D-convolution projection layer.
    Note that the last hidden layer does not contains activation function.

    input:  bank of convolution output   [B, text_len * K, 128]
    output: 1D convolution layer output  [B, text_len * K, 128]
    """

    def __init__(self, K, hidden_dims):
        super(Conv1DProjection, self).__init__()

        # convolution layer dimensions
        in_dims     = [K * 128] + hidden_dims[:-1]
        out_dims    = hidden_dims[1:]
        activations = [nn.ReLU()] * len(out_dims) + [None]

        # Several convolution projection layers
        self.convolutions = nn.ModuleList(
            [Conv1DBN(in_dim, out_dim, k=3, bias=True, activation=act)
             for in_dim, out_dim, act in zip(in_dims, out_dims, activations)]
        )

        # Linear layer to match dimension
        self.linear = nn.Linear(hidden_dims[-1], 128, bias=False)


    def forward(self, x):

        # Transpose for convolution
        x = x.transpose(1, 2)

        # convolution projections
        for conv in self.convolutions:
            x = conv(x)
            
        # Transpose for convolution
        x = x.transpose(1, 2)

        # Match dimension with Highway input
        return self.linear(x)
    

class CBHG(nn.Module):
    """
    Tacotron paper - 3.1 CBHG Module

    input:  prenet output  [B, text_len, 128]
    output: encoder out    [B, text_len, 256]
    """

    def __init__(self, K, projections):

        super(CBHG, self).__init__()
        self.convolution_bank   = Conv1DBank(K)
        self.convolution_proj   = Conv1DProjection(K, projections)
        self.highways           = nn.ModuleList([Highway(128, 128) for _ in range(4)])
        self.bidirectional_gru  = nn.GRU(128, 128, batch_first=True, bidirectional=True)


    def forward(self, x):

        residual = x

        # Conv 1D bank + stacking + maxpool -> [B, text_len, K * 256]
        x = self.convolution_bank(x)

        # Convolution projection -> [B, text_len, 128]
        x = self.convolution_proj(x)

        # residual connection
        x += residual

        # highway layers -> [B, text_len, 128]
        for highway in self.highways:
            x = highway(x)

        # Bidirectional RNN -> [B, text_len, 256]
        x, _ = self.bidirectional_gru(x)
        
        return x


class AttentionRNN(nn.Module):
    """
    Tacotron paper - Table 1. Network architectures
    
    Attention RNN consists of `1 layer GRU Cell`.

    input : decoder prenet out               [B, 128]
    input : attention out (context vector)   [B, 256]
    input : attention rnn hidden             [B, 256]

    output: attention rnn hidden             [B, 256]

    """
    
    def __init__(self):

        super(AttentionRNN, self).__init__()
        self.gru = nn.GRUCell(128 + 256, 256)


    def forward(self, prenet_out, attn_out, attn_rnn_hidden):
        
        # We concatenate the context vector and the attention RNN cell output 
        x = torch.cat([prenet_out, attn_out], dim=1)

        # 1-layer GRU (256 cells)
        return self.gru(x, attn_rnn_hidden)
    

class DecoderRNN(nn.Module):
    """
    Tacotron paper - 3.3 Decoder

    We concatenate the context vector and the attention RNN cell output to form
    the input to the decoder RNNs. We use a stack of GRUs with vertical residual
    connections for the decoder.

    input : attention out (context vector)      [B, 256]
    input : attention rnn hidden                [B, 256] 
    input : decoder rnn hiddens                 [2, B, 256]

    output: decoder out                         [B, 256]
    output: decoder rnn hiddens                 [2, B, 256]
    """

    def __init__(self):

        super(DecoderRNN, self).__init__()
        self.linear = nn.Linear(512, 256, bias=False)
        self.gru1 = nn.GRUCell(256, 256)
        self.gru2 = nn.GRUCell(256, 256)


    def forward(self, attn_out, attn_rnn_hidden, dec_rnn_hiddens):

        # match input dimension with linear -> [B, 256]
        x = torch.cat([attn_rnn_hidden, attn_out], dim=1)
        x = self.linear(x)

        # first layer residual GRU -> [B, 256]
        dec_rnn_hiddens[0] = self.gru1(x, dec_rnn_hiddens[0])
        x = x + dec_rnn_hiddens[0]

        # second layer residual GRU -> [B, 256]
        dec_rnn_hiddens[1] = self.gru2(x, dec_rnn_hiddens[1])
        x = x + dec_rnn_hiddens[1]

        # [B, 256], [2, B, 256]
        return x, dec_rnn_hiddens
    

class Attention(nn.Module):
    """
    Tacotron paper - 3.3 Decoder

    Content-based tanh attention decoder (Vinyals et al. (2015))
    Grammar as a foreign language

    input : attention rnn hidden    [B, 256]
    input : encoder output          [B, 256]
    input : mask                    [B, text_len, 256]

    output: attention out           [B, 256]
    output: alignment               [B, text_len]

    """

    def __init__(self):

        super(Attention, self).__init__()
        self.W1         = nn.Linear(256, 256)
        self.W2         = nn.Linear(256, 256)
        self.v          = nn.Linear(256, 1, bias=False)
        self.tanh       = nn.Tanh()
        self.softmax    = nn.Softmax(dim=1)


    def forward(self, d, h, mask):
        
        # Projection of attention rnn hidden -> [B, 256]
        d_proj = self.W1(d)

        # Projection of encoder out -> [B, text_len, 256]
        h_proj = self.W2(h)

        # Expand attention rnn hidden dimension -> [B, 1, 256]
        if d_proj.dim() == 2:
            d_proj = d_proj.unsqueeze(1)

        # Add projection results and apply tanh -> [B, text_len, 256]
        o = self.tanh(d_proj + h_proj)
        
        # Calculate attention score -> [B, text_len, 1]
        u = self.v(o)

        # Squeeze output -> [B, text_len]
        u = u.squeeze(2)

        # if using masked attention
        if mask is not None:
            mask = mask.view(d.size(0), -1) # [B, -1]
            u.data.masked_fill_(mask, -1e9) # u = [B, text_len]

        # Convert to probability -> [B, text_len]
        a = self.softmax(u)

        # Matrix multiplication with attention score -> [B, 1, 256]
        h_prime = torch.bmm(a.unsqueeze(1), h)

        # Squeeze output -> [B, 256]
        h_prime = h_prime.squeeze(1)
        
        return h_prime, a