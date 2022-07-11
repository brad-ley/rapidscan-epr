import numpy as np


def cycle(ref_signal, signal, start, stop):
    """cycle.

    :param ref_signal: complex reference signal
    :param signal: complex new signal to be phased
    """
    phase = np.angle(np.dot(np.conjugate(signal)[start:stop], ref_signal[start:stop]))
    # phase = 0
    return signal * np.exp(1j * phase)
