import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from readDataFile import read
from deconvolveRapidscan import deconvolve

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process(filename):
    coil = 0.1325/2
    amplitude = 150
    frequency = 70e3
    phase = 2.1
    B, M = deconvolve(filename, coil, amplitude, frequency, Bphase=-np.pi)
    M *= np.exp(1j * phase)
    d = pd.DataFrame({'B':B, 'M':M})
    d.to_csv(P(filename).parent.joinpath(P(filename).stem + '_slowscan.dat'))


if __name__ == "__main__":
    folder = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/8/31/sweeping mod field/filtered'
    files = [ii for ii in P(folder).iterdir() if ii.name.endswith('realFilterMagnitude.dat')]
    for i, f in enumerate(files):
        process(f)
