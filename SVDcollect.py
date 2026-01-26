import ast
import contextlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

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


EXP_MAX = 350


def main(base_folder: str) -> None:
    temps = []
    w1 = []
    w2 = []
    exp = []
    for folder in Path(base_folder).rglob("SVD"):
        if folder.joinpath("PC2_fits.txt").exists():
            d = pd.read_csv(folder.joinpath("PC2_fits.txt"))
            # print(folder.parent)
            with contextlib.suppress(ValueError):
                # if True:
                if (
                    float(d.loc[d["name"] == "w1"]["stderr"].iloc[0]) < 1
                    and float(d.loc[d["name"] == "w2"]["stderr"].iloc[0]) < 1
                    and abs(
                        [
                            float(ii)
                            for ii in d[d["name"].str.contains("exponential")]["value"]
                            .iloc[0]
                            .strip("[")
                            .strip("]")
                            .split()
                        ][-1]
                    )
                    < EXP_MAX
                ):
                    temps.append(float(folder.parent.name.split("K")[0]))
                    w1.append(float(d.loc[d["name"] == "w1"]["value"].iloc[0]))
                    w2.append(float(d.loc[d["name"] == "w2"]["value"].iloc[0]))
                    exp.append(
                        [
                            float(ii)
                            for ii in d[d["name"].str.contains("exponential")]["value"]
                            .iloc[0]
                            .strip("[")
                            .strip("]")
                            .split()
                        ][-1],
                    )

    f, a = plt.subplots(figsize=(7, 5))
    a.set_title(r"Linewidths of SVDs with $\tau<350\,$s")
    a.scatter(temps, w1, label=r"$\omega_1$")
    a.scatter(temps, w2, label=r"$\omega_2$")
    a.set_xlabel("Temp (K)")
    a.set_ylabel("Linewidth (G)")
    a.legend()
    # a.scatter(temps, exp)
    f.savefig("/Users/Brad/Desktop/svd_linewidths.png", dpi=600)


if __name__ == "__main__":
    main(
        base_folder="/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024"
    )
    plt.show()
