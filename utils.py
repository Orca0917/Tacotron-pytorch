import torch
import numpy as np
import matplotlib.pyplot as plt


def show_melspectrogram(mel_pred, mel_targ, step=0):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    im1 = ax1.imshow(mel_pred, aspect="auto", origin="lower", interpolation="none")
    im2 = ax2.imshow(mel_targ, aspect="auto", origin="lower", interpolation="none")
    
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f"/opt/ml/fig/melspectrogram-{step}.png")
    plt.close()


def show_alignment(alignment, step=0):

    fig, ax = plt.subplots(1, 1, figsize=(7, 8))
    im1 = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im1, ax=ax)

    plt.tight_layout()
    plt.savefig(f"/opt/ml/fig/alignment-{step}.png")
    plt.close()


def mask_from_lengths(ref, lengths):
    
    B, text_len, _ = ref.size()

    mask = ref.new(B, text_len).fill_(1)
    for idx, l in enumerate(lengths):
        mask[idx][l:] = 0

    return mask.bool()


def _learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr