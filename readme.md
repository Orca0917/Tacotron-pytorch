# Tacotron (2017)

This is a repository for an unofficial implementation of the [Tacotron](https://ar5iv.labs.arxiv.org/html/1703.10135) speech synthesis model using PyTorch. Tacotron is a model that converts given text into a spectrogram and then uses the Griffin-Lim algorithm to synthesize speech. For more detailed information, please refer to [here](#reference).

**The implementation is still in progress.**

<br>

## Architecture

<div align="center">

![alt text](/img/model-architecture.png)

</div>

<br>

## How to run

All implementations were carried out on a container utilizing a PyTorch-based Docker image. There is no need to download a separate dataset, as the torchaudio package is used to automatically download the LJSpeech-1.1 dataset.

1. Prepare docker image / container

    ```text
    docker pull pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
    ```

2. Install requirements

    ```bash
    pip install -r requirements.txt
    ```

3. Start training

    ```bash
    python train.py
    ```

<br>

## Reference

[1] Wang, Yuxuan, et al. "Tacotron: Towards end-to-end speech synthesis." arXiv preprint arXiv:1703.10135 (2017).

[2] https://github.com/r9y9/tacotron_pytorch

[3] https://github.com/Kyubyong/tacotron

[4] Keith Ito and Linda Johnson, The LJ Speech Dataset, https://keithito.com/LJ-Speech-Dataset/, 2017