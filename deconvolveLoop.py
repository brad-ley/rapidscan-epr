import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from readDataFile import read
from deconvolveRapidscan import deconvolve
from statusBar import statusBar

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process(filename):
    coil = 0.62
    amplitude = 153
    frequency = 70e3
    phase = 2
    B, M = deconvolve(filename, coil, amplitude, frequency, Bphase=-np.pi)
    M *= np.exp(1j * phase)
    d = pd.DataFrame({'B':B, 'M':M})
    d.to_csv(P(filename).parent.joinpath(P(filename).stem + '_slowscan.dat'))


if __name__ == "__main__":
    folder = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/9/2/C RS 5 s off, 5 s on, off/filtered'
    if P(folder).is_file():
        folder = P(folder).parent
    files = [ii for ii in P(folder).iterdir() if ii.name.endswith('realFilterMagnitude.dat')]
    for i, f in enumerate(files):
        process(f)
        statusBar((i+1)/len(files)*100)
