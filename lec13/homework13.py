import numpy as np
import librosa
from scipy.signal import lfilter


def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.

    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute

    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frame
    excitation (nframes,frame_length) - linear prediction excitation frames
    '''

    nframes = 1 + (len(speech) - frame_length) // frame_skip

    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        # LPC coefficients
        a = librosa.lpc(frame, order=order)
        A[i] = a

        # Residual (excitation)
        excitation[i] = lfilter(a, [1.0], frame)

    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    '''

    nframes = A.shape[0]
    frame_length = len(e) // nframes

    synthesis = np.zeros(len(e))

    for i in range(nframes):
        start = i * frame_skip
        end = start + frame_length

        residual = e[i * frame_length:(i + 1) * frame_length]

        # LPC synthesis
        frame = lfilter([1.0], A[i], residual)

        synthesis[start:end] += frame

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Create pulse-train excitation with the same frame energy.
    '''

    nframes, frame_length = excitation.shape

    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):

        # RMS gain
        gain[i] = np.sqrt(np.mean(excitation[i] ** 2))

        # Pulse train
        frame = np.zeros(frame_skip)
        frame[::T0] = gain[i]

        e_robot[i * frame_skip:(i + 1) * frame_skip] = frame

    return gain, e_robot