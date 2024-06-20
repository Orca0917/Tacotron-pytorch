class Hyperparams:
    # Audio preprocessing
    preemphasis = 0.97
    n_fft = 2048
    frame_length_ms = 50
    frame_shift_ms = 12.5
    sr = 22050
    ref_level_db = 20
    min_level_db = -100
    
    n_mels = 80
    win_length = int(round(frame_length_ms * sr / 1000))
    hop_length = int(round(frame_shift_ms * sr / 1000))
    n_phonemes = 60

    # model
    reduction_factor = 2
    ch_emb_dim = 256
    encoder_K = 16
    decoder_K = 8
    encoder_projection = [128, 128]
    decoder_projection = [256, 80]

    # training
    batch_size = 16 # all sequences are padded to max length
    max_decoding_timestep = 200
    learning_rate = 0.002
    epoch = 50

hp = Hyperparams()