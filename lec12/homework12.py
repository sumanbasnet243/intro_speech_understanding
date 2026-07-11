import numpy as np

def voiced_excitation(duration, F0, Fs):
    '''
    Create voiced speech excitation.
    '''
    excitation = np.zeros(duration)

    # Pitch period in samples
    period = int(np.round(Fs / F0))

    # Place impulses
    excitation[::period] = -1

    return excitation


def resonator(x, F, BW, Fs):
    '''
    Generate the output of a resonator.
    '''
    N = len(x)
    y = np.zeros(N)

    # Resonator coefficients
    r = np.exp(-np.pi * BW / Fs)
    theta = 2 * np.pi * F / Fs

    a1 = 2 * r * np.cos(theta)
    a2 = -r ** 2
    b0 = 1 - r

    for n in range(N):
        y[n] = b0 * x[n]

        if n >= 1:
            y[n] += a1 * y[n - 1]

        if n >= 2:
            y[n] += a2 * y[n - 2]

    return y


def synthesize_vowel(duration, F0, F1, F2, F3, F4,
                     BW1, BW2, BW3, BW4, Fs):
    '''
    Synthesize a vowel.
    '''
    # Create excitation
    speech = voiced_excitation(duration, F0, Fs)

    # Cascade the four formant resonators
    speech = resonator(speech, F1, BW1, Fs)
    speech = resonator(speech, F2, BW2, Fs)
    speech = resonator(speech, F3, BW3, Fs)
    speech = resonator(speech, F4, BW4, Fs)

    return speech