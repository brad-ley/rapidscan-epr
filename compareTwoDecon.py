import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from readDataFile import read

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc
import pyarrow.feather as feather

if __name__ == "__main__":
    plt.style.use(["science"])
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rcParams = [
        ["font.family", "sans-serif"],
        ["font.size", 14],
        ["axes.linewidth", 1],
        ["lines.linewidth", 2],
        ["xtick.major.size", 5],
        ["xtick.major.width", 1],
        ["xtick.minor.size", 2],
        ["xtick.minor.width", 1],
        ["ytick.major.size", 5],
        ["ytick.major.width", 1],
        ["ytick.minor.size", 2],
        ["ytick.minor.width", 1],
    ]
    plt.rcParams.update(dict(rcParams))


def main(f1, f2):
    fig, a = plt.subplots()
    for f in f1, f2:
        f = P(f)
        dat = feather.read_feather(f)
        col = [ii for ii in dat.columns if "abs" in ii]
        dat["B"] += 9.5
        a.plot(
            dat["B"][abs(dat["B"]) < 15],
            dat[col][abs(dat["B"]) < 15] / 0.0025,
            label=f.parent.name.title(),
        )

    a.legend(handlelength=0.5)
    a.set_ylabel("Signal (arb. u)")
    a.set_xlabel("Field (G)")

    fig.savefig(f.parent.parent.joinpath("comparedRS.png"), dpi=1200)


if __name__ == "__main__":
    f1 = "/Users/Brad/Downloads/no mesh/100mA_23.2kHz_acq10s_25000avgs_filtered_batchDecon.feather"
    f2 = "/Users/Brad/Downloads/mesh/100mA_23.2kHz_acq10s_25000avgs_filtered_batchDecon.feather"
    main(f1, f2)
    plt.show()
