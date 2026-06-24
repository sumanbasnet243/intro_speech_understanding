import numpy as np

def VAD(waveform, Fs):
    '''
    Extract segments whose frame energy is greater than 10% of maximum.
    Frame length = 25 ms, frame step = 10 ms.
    '''

    frame_len = int(0.025 * Fs)
    frame_step = int(0.010 * Fs)

    energies = []
    frame_starts = []

    for start in range(0, len(waveform) - frame_len + 1, frame_step):
        frame = waveform[start:start + frame_len]
        energy = np.sum(frame.astype(float) ** 2)
        energies.append(energy)
        frame_starts.append(start)

    energies = np.array(energies)

    if len(energies) == 0:
        return []

    threshold = 0.1 * np.max(energies)

    speech_frames = energies > threshold

    segments = []
    in_segment = False

    for i, speech in enumerate(speech_frames):

        if speech and not in_segment:
            seg_start = frame_starts[i]
            in_segment = True

        elif not speech and in_segment:
            seg_end = frame_starts[i] + frame_len
            segments.append(waveform[seg_start:seg_end])
            in_segment = False

    if in_segment:
        segments.append(waveform[seg_start:])

    return segments


def segments_to_models(segments, Fs):
    '''
    Create average log-spectrum model for each segment.
    '''

    models = []

    frame_len = int(0.004 * Fs)      # 4 ms
    frame_step = int(0.002 * Fs)     # 2 ms

    preemph = 0.97

    for segment in segments:

        if len(segment) < frame_len:
            continue

        # Pre-emphasis
        emphasized = np.append(segment[0],
                               segment[1:] - preemph * segment[:-1])

        spectra = []

        for start in range(0,
                           len(emphasized) - frame_len + 1,
                           frame_step):

            frame = emphasized[start:start + frame_len]

            # Hamming window
            frame = frame * np.hamming(frame_len)

            spectrum = np.abs(np.fft.rfft(frame))

            # Log spectrum
            spectrum = np.log(spectrum + 1e-10)

            spectra.append(spectrum)

        if len(spectra) == 0:
            continue

        spectra = np.array(spectra)

        # Average spectrum over time
        model = np.mean(spectra, axis=0)

        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize speech using cosine similarity.
    '''

    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))

    test_outputs = []

    for k, test_model in enumerate(test_models):

        for y, model in enumerate(models):

            min_len = min(len(model), len(test_model))

            m1 = model[:min_len]
            m2 = test_model[:min_len]

            denom = (np.linalg.norm(m1) *
                     np.linalg.norm(m2))

            if denom == 0:
                sim = 0
            else:
                sim = np.dot(m1, m2) / denom

            sims[y, k] = sim

        best_idx = np.argmax(sims[:, k])
        test_outputs.append(labels[best_idx])

    return sims, test_outputs