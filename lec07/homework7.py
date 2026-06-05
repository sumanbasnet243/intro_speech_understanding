import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord.
    '''
    t = np.arange(int(0.5 * Fs)) / Fs

    # Root, major third, and major fifth
    f_root = f
    f_third = f * 2**(4/12)
    f_fifth = f * 2**(7/12)

    x = (np.cos(2*np.pi*f_root*t) +
         np.cos(2*np.pi*f_third*t) +
         np.cos(2*np.pi*f_fifth*t))

    return x


def dft_matrix(N):
    '''
    Create an NxN DFT transform matrix.
    '''
    n = np.arange(N)
    k = n.reshape((N, 1))

    W = np.exp(-2j * np.pi * k * n / N)

    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    N = len(x)

    # DFT
    W = dft_matrix(N)
    X = W @ x

    # Magnitude spectrum
    mag = np.abs(X)

    # Keep only non-negative frequencies
    mag = mag[:N//2]
    freqs = np.arange(N//2) * Fs / N

    # Indices of three largest peaks
    idx = np.argsort(mag)[-3:]

    # Corresponding frequencies
    loudest = np.sort(freqs[idx])

    f1, f2, f3 = loudest

    return f1, f2, f3