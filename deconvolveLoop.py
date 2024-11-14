import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from deconvolveRapidscan import deconvolve
from statusBar import statusBar

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process(filename):
    coil = 0.6
    amplitude = 156
    frequency = 69e3
    Bphase = -1.4
    Mphase = 1.75
    B, M = deconvolve(filename, coil, amplitude, frequency, Bphase=Bphase)
    M *= np.exp(1j * Mphase)
    d = pd.DataFrame({"B": B, "M": M})
    d.to_csv(P(filename).parent.joinpath(P(filename).stem + "_slowscan.dat"))


if __name__ == "__main__":
    folder = "/Volumes/GoogleDrive/My Drive/Research/Data/2022/9/14"
    if P(folder).is_file():
        folder = P(folder).parent
    files = [
        ii
        for ii in P(folder).iterdir()
        if ii.name.endswith("realFilterMagnitude.dat")
    ]
    if files == []:
        files = [ii for ii in P(folder).iterdir() if ii.name.endswith("s.dat")]
    for i, f in enumerate(files):
        process(f)
        statusBar((i + 1) / len(files) * 100)
