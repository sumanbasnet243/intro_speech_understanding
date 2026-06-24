import numpy as np
import torch
import torch.nn as nn

def get_features(waveform, Fs):
    '''
    Extract spectrogram features and labels.
    '''

    # --------------------------
    # Pre-emphasis
    # --------------------------
    preemph = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # --------------------------
    # Spectrogram parameters
    # --------------------------
    frame_len = int(0.004 * Fs)   # 4 ms
    frame_step = int(0.002 * Fs)  # 2 ms

    nframes = 1 + (len(preemph) - frame_len) // frame_step

    frames = np.zeros((nframes, frame_len))

    for i in range(nframes):
        start = i * frame_step
        frames[i] = preemph[start:start + frame_len]

    # Hamming window
    frames *= np.hamming(frame_len)

    # FFT and keep low-frequency half
    spec = np.abs(np.fft.rfft(frames, axis=1))
    features = spec.astype(np.float32)

    # --------------------------
    # VAD labels
    # --------------------------
    vad_len = int(0.025 * Fs)     # 25 ms
    vad_step = int(0.010 * Fs)    # 10 ms

    vad_frames = []
    for start in range(0, len(waveform) - vad_len + 1, vad_step):
        frame = waveform[start:start + vad_len]
        vad_frames.append(np.sum(frame**2))

    vad_frames = np.array(vad_frames)

    threshold = 0.1 * np.max(vad_frames)

    labels = np.zeros(nframes, dtype=np.int64)

    current_label = 1

    for i, energy in enumerate(vad_frames):

        if energy > threshold:

            start_time = i * vad_step / Fs
            end_time = (i * vad_step + vad_len) / Fs

            feat_start = int(start_time / 0.002)
            feat_end = int(end_time / 0.002)

            feat_end = min(feat_end, nframes)

            labels[feat_start:feat_end] = current_label

            current_label += 1

    return features, labels


def train_neuralnet(features, labels, iterations):
    '''
    Train Sequential(LayerNorm, Linear)
    '''

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    nfeats = features.shape[1]
    nlabels = int(labels.max()) + 1

    model = nn.Sequential(
        nn.LayerNorm(nfeats),
        nn.Linear(nfeats, nlabels)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    lossvalues = np.zeros(iterations)

    for i in range(iterations):

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        lossvalues[i] = loss.item()

    return model, lossvalues


def test_neuralnet(model, features):
    '''
    Return softmax probabilities
    '''

    X = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)

    return probabilities.detach().numpy()