import torch
import torchaudio
import torchaudio.functional as taf
import librosa
import numpy as np
from hyperparams import hp
from scipy import signal


class TacotronDataset(torch.utils.data.Dataset):
    """
    https://github.com/Kyubyong/tacotron/blob/master/utils.py#L21
    https://github.com/ttaoREtw/Tacotron-pytorch/blob/master/src/utils.py
    """

    def __init__(self):
        self.dataset = torchaudio.datasets.LJSPEECH('.', download=True)
        self.text_preprocess = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # get data
        y, sr, _, text = self.dataset[idx]

        # text preprocess
        text, _ = self.text_preprocess(text)

        # audio preprocess
        lin, mel = self.audio_preprocess(y, sr)

        # (text_len), (2048, len), (n_mel, mel_len)
        return text.squeeze(), lin, mel
    
    # Preemphasis
    def preemphasis(self, wav):
        return signal.lfilter([1, -hp.preemphasis], [1], wav)
    
    # Short Time Frourier Transform
    def _stft(self, x):
        return librosa.stft(x, n_fft=hp.n_fft, hop_length=hp.hop_length, 
                            win_length=hp.win_length)

    # Spectrogram
    def spectrogram(self, wav):
        D = self._stft(self.preemphasis(wav))
        S = self._amp_to_db(np.abs(D)) - hp.ref_level_db
        return self._normalize(S)
    
    # Melspectrogram
    def melspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) # - hp.ref_level_db
        return self._normalize(S)
    
    # amplitude to decibel
    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))
    
    # spectrogram to melspectrogram
    def _linear_to_mel(self, mag):
        mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
        return np.dot(mel_basis, mag)
    
    # normalization
    def _normalize(self, x):
        return np.clip((x - hp.min_level_db) / -hp.min_level_db, 0, 1)

    def audio_preprocess(self, y, sr):
        
        y = y.numpy().squeeze()

        # trimming
        y, _ = librosa.effects.trim(y)

        # spectrogram, melspectrogram
        spectrogram = self.spectrogram(y)
        melspectrogram = self.melspectrogram(y)
        
        return torch.FloatTensor(spectrogram), torch.FloatTensor(melspectrogram)


class TacotronCollate():

    def __init__(self):
        ...

    def __call__(self, batch):

        # get decreasing order by text length within batch
        text_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(text) for text, _, _ in batch]),
            dim=0, descending=True
        )

        # all zero padded tensor
        max_text_len = text_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_text_len)
        text_padded.zero_()

        # allocate text to zero padded tensor
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # get maximum length of sequence within batch
        num_lins = batch[0][1].size(0)
        num_mels = batch[0][2].size(0)
        max_seq_len = max([lin.size(1) for _, lin, _ in batch])
        max_seq_len = max_seq_len + (hp.reduction_factor - max_seq_len % hp.reduction_factor)


        # all zero padded tensor
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_seq_len)
        mel_padded.zero_()
        lin_padded = torch.FloatTensor(len(batch), num_lins, max_seq_len)
        lin_padded.zero_()
        seq_lengths = torch.LongTensor(len(batch))

        
        for i in range(len(ids_sorted_decreasing)):
            _, lin, mel = batch[ids_sorted_decreasing[i]]
            lin_padded[i, :, :lin.size(1)] = lin
            mel_padded[i, :, :mel.size(1)] = mel
            seq_lengths[i] = lin.size(1)


        return (
            text_padded,
            lin_padded.transpose(1, 2),
            mel_padded.transpose(1, 2),
            text_lengths,
            seq_lengths
        )
