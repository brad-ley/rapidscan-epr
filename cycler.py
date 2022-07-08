import numpy as np


def cycle(ref_signal, signal):
    """cycle.

    :param ref_signal: complex reference signal
    :param signal: complex new signal to be phased
    """
    phase = np.angle(np.dot(np.conjugate(signal), ref_signal))
    # phase = 0
    return signal * np.exp(1j * phase)



if __name__ == "__main__":
    cycle()
