import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.feather as feather
import lmfit
import pybroom
import warnings

from matplotlib import rc
from pathlib import Path as P
from scipy.optimize import curve_fit
from tqdm import tqdm
from typing import List, Tuple

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


def lorentzian(
    x: np.ndarray, x0: float, a: float, b: float, c: float
) -> np.ndarray:
    """lorentzian.

    :param x: x-axis array
    :param x0: center of Lorentzian
    :param a: baseline offset
    :param b: amplitude
    :param c: linewidth
    """
    return a + b / np.pi * (c / 2) / ((x - x0) ** 2 + (c / 2) ** 2)


def fit_fun(params, data, B):
    """
    Fit handler to return the RMSE deviation from ideal double_lorentzian
    """
    return (data - double_lorentzian(params, B)) ** 2


def double_lorentzian(params, x):
    """
    Need to normalize both to make sure the total number of spins going from
    one to the other is conserved
    """
    parvals = params.valuesdict()
    x0 = parvals["x0"]
    a1 = parvals["a1"]
    a2 = parvals["a2"]
    b1 = parvals["b1"]
    b2 = parvals["b2"]
    w1 = parvals["w1"]
    w2 = parvals["w2"]
    return lorentzian(x, x0, a1, b1, w1) + lorentzian(x, x0, a2, b2, w2)


def exp(x, a, b, t):
    return a + b * np.exp(-(x - 0) / t)


class DataSet:
    """Class for processing rs TiGGER SVDs"""

    def __init__(self, filename: str, plotfield: float = 30):
        self.filename = filename

        self.mat = feather.read_feather(self.filename)

        self.cols = [ii for ii in self.mat.columns if "abs" in ii]
        B = self.mat["B"].to_numpy()
        first = self.mat[self.cols[0]].to_numpy()[np.abs(B) < plotfield]
        plotcenter = B[
            np.where(np.abs(B) < plotfield)[0][0] + np.argmax(first)
        ]
        plotlim = plotcenter + np.array([-plotfield, plotfield])
        lims = np.where(np.logical_and(B >= plotlim[0], B < plotlim[1]))
        self.low = lims[0][0]
        self.high = lims[0][-1]
        self.B = B[self.low : self.high] - plotcenter
        self.times = np.array(
            ast.literal_eval(
                P(self.filename).parent.joinpath("times.txt").read_text()
            )
        )

        self.dat = self.mat[self.cols].to_numpy()[self.low : self.high, :]

        if "_pre" in P(self.filename).stem:
            pre = "".join(
                [
                    ii
                    for ii in P(self.filename).stem.split("_")
                    if "pre" in ii and "s" in ii
                ]
            )
            self.pre = float("".join([ii for ii in list(pre) if ii.isdigit()]))
        if "_on" in P(self.filename).stem:
            on = "".join(
                [
                    ii
                    for ii in P(self.filename).stem.split("_")
                    if "on" in ii and "s" in ii
                ]
            )
            self.on = float("".join([ii for ii in list(on) if ii.isdigit()]))

    def center(self):
        self.dat = np.zeros(((self.high - self.low), len(self.cols)))

        for ind, col in enumerate(self.cols):
            # center = np.argmax(mat[col][l:h].to_numpy()) + l
            tdat = self.mat[col][self.low : self.high].to_numpy()
            # n = 2**3
            n = 2**7
            rolling = np.array(
                [
                    (
                        np.mean(tdat[ii - n : ii + n])
                        if (ii > n and len(tdat) - ii > n)
                        else 0
                    )
                    for ii, _ in enumerate(tdat)
                ]
            )
            center = np.argmax(rolling) + self.low
            coldat = self.mat[col].to_numpy()
            try:
                self.dat[:, ind] = coldat[
                    center - int((self.high - self.low) / 2) : center
                    + int((self.high - self.low) / 2)
                ]
            except ValueError:
                self.dat[:, ind] = coldat[
                    center - int((self.high - self.low) / 2) : center
                    + int((self.high - self.low) / 2)
                    + 1
                ]

        return self

    def wrap(self, wrap_time: float = 0):
        try:
            wrap_idx = np.where(self.times > wrap_time)[0][0]
            self.dat = np.roll(self.dat, self.dat.shape[1] - wrap_idx, axis=1)
        except IndexError:
            pass
        return self

    def mean_center(self):
        self.mean_centered = True
        self.mu = np.mean(self.dat, axis=1)
        self.dat -= self.mu[:, np.newaxis]  # type: ignore
        return self

    def svd(self, k=None):
        U, E, Vh = np.linalg.svd(self.dat)

        if not k:
            ratios = np.array(
                [
                    E[idx] / E[idx + 1] if idx + 1 < len(E) else 0
                    for idx, _ in enumerate(E)
                ]
            )
            k = np.where(ratios < 2)[0][0]  # find first time it goes below 2
        if k == 1:
            k = 2  # have at least 2 eigenvalues so the plotting indices don't error out
        self.k = k

        self.U = U[:, :k]
        self.E = E[:k]
        self.Vh = Vh[:k, :]
        self.subbasis_data = self.U @ np.diag(self.E) @ self.Vh
        return self

    def plotMatrix(self, subbasis=False):
        """plotMatrix.

        :param subbasis: if True, show imshow image of the data recreated with k principal components
        """
        self.imshow_f, self.a = plt.subplots(figsize=(8, 6))
        dat = self.dat

        if subbasis:
            self.subbasis = True
            dat = self.subbasis_data
            self.a.set_title(f"PCA ($k = {self.k}$) sub-basis recreated data")

        if hasattr(self, "mean_centered"):
            dat += self.mu[:, np.newaxis]  # type: ignore
        self.a.imshow(dat, aspect="auto")
        self.a.set_xlabel("Time (s)")
        self.a.set_ylabel("Intensity (arb. u)")

        return self

    def show(self):
        plt.show()
        return self

    def plotSVs(self):
        self.plotLSVs()
        self.plotRSVs()
        return self

    def plotLSVs(self):
        self.lsvf, self.lsva = plt.subplots(
            nrows=self.Vh.shape[0], figsize=(8, 6)
        )
        for idx, u in enumerate(self.U[0, :]):
            (line,) = self.lsva[idx].plot(
                self.B, self.U[:, idx], label=f"$C_{{{idx + 1}}}$", alpha=0.5
            )
            find = 2
            if hasattr(self, "mean_centered"):
                find = 1
            try:
                if idx + 1 == find - 1:
                    popt, pcov = curve_fit(lorentzian, self.B, self.U[:, idx])
                    self.lsva[idx].plot(
                        self.B,
                        lorentzian(self.B, *popt),
                        c=line.get_color(),
                        ls="--",
                        label=rf"$\omega={popt[-1]:.1f}\,$G",
                    )
                if idx + 1 == find:
                    params = lmfit.create_params(
                        x0=dict(
                            value=0.0,
                            vary=False,
                            min=-np.max(self.B),
                            max=np.max(self.B),
                        ),
                        a1=dict(
                            value=0,
                            vary=True,
                            min=-np.max(self.U[:, idx]),
                            max=np.max(self.U[:, idx]),
                        ),
                        a2=dict(
                            value=0,
                            vary=True,
                            min=-np.max(self.U[:, idx]),
                            max=np.max(self.U[:, idx]),
                        ),
                        b1=dict(
                            value=0.1,
                            vary=True,
                            min=0.1,
                            max=np.max(self.E),
                        ),
                        b2=dict(
                            value=-0.1,
                            vary=True,
                            min=-np.max(self.E),
                            max=-0.1,
                        ),
                        w1=dict(
                            value=10, vary=True, min=3, max=np.max(self.B)
                        ),
                        w2=dict(
                            value=30, vary=True, min=3, max=np.max(self.B)
                        ),
                    )

                    obj = lmfit.Minimizer(
                        fit_fun,
                        params,
                        fcn_args=(self.U[:, idx], self.B),
                    )
                    self.res = obj.minimize(method="basinhopping")
                    # popt, pcov = curve_fit(
                    #     double_lorentzian,
                    #     self.B,
                    #     self.U[:, idx],
                    #     p0=[
                    #         0,
                    #         np.max(self.U[:, idx]),
                    #         -np.max(self.U[:, idx]),
                    #         5,
                    #         40,
                    #     ],
                    #     # maxfev=10000,
                    # )
                    # self.lsva[idx].plot(
                    #     self.B,
                    #     double_lorentzian(self.B, *popt),
                    #     c=line.get_color(),
                    #     ls="--",
                    #     label=rf"$\omega_1={popt[-4]:.1f}\,$G, $\omega_2={popt[-1]:.1f}\,$G",
                    # )
                    parvals = self.res.params.valuesdict()  # type: ignore

                    x0 = parvals["x0"]
                    a1 = parvals["a1"]
                    a2 = parvals["a2"]
                    b1 = parvals["b1"]
                    b2 = parvals["b2"]
                    w1 = parvals["w1"]
                    w2 = parvals["w2"]

                    self.lsva[idx].plot(
                        self.B,
                        double_lorentzian(self.res.params, self.B),  # type: ignore
                        c=line.get_color(),
                        ls="--",
                        label=r"$F_{\omega_1} + F_{\omega_2}$",
                    )
                    self.lsva[idx].plot(
                        self.B,
                        lorentzian(self.B, x0, a1, b1, w1),
                        # c=line.get_color(),
                        ls="--",
                        label=rf"$\omega_1={w1:.1f}\,$G",
                    )
                    self.lsva[idx].plot(
                        self.B,
                        lorentzian(self.B, x0, a2, b2, w2),
                        # c=line.get_color(),
                        ls="--",
                        label=rf"$\omega_2={w2:.1f}\,$G",
                    )

            except RuntimeError:
                print(f"Could not fit w_{idx+1} component")
            self.lsva[idx].legend(
                loc="upper right",
                handlelength=0.75,
                labelspacing=0.25,
            )
        self.lsvf.supxlabel("Field (G)")
        self.lsvf.supylabel("Intensity")
        return self

    def plotRSVs(self):
        self.rsvf, self.rsva = plt.subplots(
            nrows=self.Vh.shape[0], figsize=(8, 6)
        )
        find = 2
        if hasattr(self, "mean_centered"):
            find = 1
        for idx, v in enumerate(self.Vh[:, 0]):
            (line,) = self.rsva[idx].plot(
                self.times,
                self.E[idx] * self.Vh[idx, :] + 0.0 * idx,
                label=f"$w_{{{idx + 1}}}(t)$",
                alpha=0.5,
            )
            try:
                popt, pcov = curve_fit(
                    exp,
                    self.times[self.times > self.pre + self.on]
                    - (self.pre + self.on),
                    self.E[idx]
                    * self.Vh[idx, :][self.times > self.pre + self.on],
                    p0=[
                        np.min(self.Vh[idx, :]),
                        np.max(self.Vh[idx, :]),
                        np.max(self.times) / 2,
                    ],
                )
                err = 2 * np.sqrt(np.diag(pcov))
                self.rsva[idx].plot(
                    self.times[self.times > self.pre + self.on],
                    exp(
                        self.times[self.times > self.pre + self.on]
                        - (self.pre + self.on),
                        *popt,
                    )
                    + 0.0 * idx,
                    label=rf"$\tau={popt[-1]:.1f}\pm{err[-1]:.1f}\,$s",
                    c=line.get_color(),
                    ls="--",
                )
                if idx + 1 == find:
                    self.popt = popt
                    self.err = err
            except RuntimeError:
                print(f"Could not fit w_{idx+1} component")
            try:
                self.rsva[idx].axvspan(
                    self.pre,
                    self.pre + self.on,
                    facecolor="#00A7CA",
                    alpha=0.25,
                    label="Laser on",
                )
            except ValueError:
                pass
            self.rsva[idx].legend()
            # self.rsva[idx].set_ylim(
            # 1.05 * np.array([np.min(self.Vh), np.max(self.Vh)])
            # )
        self.rsvf.supxlabel("Time (s)")
        self.rsvf.supylabel(r"Intensity (weighted by $\Sigma_i$)")
        return self

    def saveResults(self):
        path = P(self.filename).parent.joinpath("SVD/")
        if not path.exists():
            path.mkdir()
        if hasattr(self, "sim_as_data"):
            add = "sim_"
        else:
            add = ""
        if hasattr(self, "rsvf"):
            self.rsvf.savefig(path.joinpath(f"{add}SVDweights.png"), dpi=1200)
        if hasattr(self, "lsvf"):
            self.lsvf.savefig(path.joinpath(f"{add}SVDvectors.png"), dpi=1200)
        if hasattr(self, "imshow_f"):
            if hasattr(self, "subbasis"):
                add += "subbasis_"
            self.imshow_f.savefig(path.joinpath(f"{add}imshow.png"), dpi=1200)

        if hasattr(self, "res") or hasattr(self, "popt"):
            dt1 = pd.DataFrame(columns=["name", "value", "stderr"])  # type: ignore
            dt2 = pd.DataFrame()
            if hasattr(self, "popt"):
                dt1.at[0, "name"] = "exponential (b, A, tau)"
                dt1.at[0, "value"] = self.popt  # type: ignore
                dt1.at[0, "stderr"] = self.err  # type: ignore
            if hasattr(self, "res"):
                dt2 = pybroom.tidy(self.res)  # type: ignore
            dt = pd.concat([dt2, dt1])
            dt.to_csv(
                path.joinpath(f"{add}PC2_fits.txt"),
                index=False,
            )
        return self

    def simulate(
        self,
        linewidths: List[Tuple[float, float]] = [(9, -1.5)],
        ratios: List[float] = [1],
    ):
        """simulate.

        :param linewidths: [(start_1, delta_1), ..., (start_n, delta_n)] input tuple for each Lorenztian that you want to appear in the time-dependent lineshape simulation
        start is initial linewidth, delta is the change when activated
        :type linewidths: List[Tuple[float, float, float]]
        :param ratios: [ratio_1_to_1,... , ratio_n_to_1]
        :type ratios: List[float]
        """

        lw_matrix = np.zeros((len(linewidths), self.times.shape[0]))
        add = np.zeros(self.times.shape[0])
        for idx, (linewidth, delta) in enumerate(linewidths):
            add[self.times > self.pre + self.on] = exp(
                self.times[self.times > self.pre + self.on]
                - (self.pre + self.on),
                0,
                delta,
                150,
            )  # 150s decay from 8 G to 10 G
            lw_matrix[idx, :] += linewidth * np.ones(self.times.shape[0]) + add

        self.simulated = np.zeros(self.dat.shape)
        for idx, t in enumerate(self.times):
            for idx2, _ in enumerate(lw_matrix[:, idx]):
                self.simulated[:, idx] += ratios[idx2] * lorentzian(
                    self.B, 0, 0, 1, lw_matrix[idx2, idx]
                )

    def use_simulate_as_data(self, **kwargs):
        if not hasattr(self, "simulated"):
            self.simulate(**kwargs)
        self.sim_as_data = True
        self.dat = self.simulated


def main(filename):
    DS = DataSet(filename)
    DS.center()
    DS.use_simulate_as_data(linewidths=[(10, -5, 5), (10, 10, 5)])
    # DS.wrap(wrap_time=1064)
    DS.mean_center()
    DS.svd(k=2)
    DS.plotSVs()
    DS.plotMatrix(subbasis=True)
    DS.saveResults()
    DS.show()


if __name__ == "__main__":
    folder = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/1/31/406-537-414"
    # ================================ #
    testing = True  # change to True to only run on the file in the next line
    filename = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/1/31/406-537 WT/282.93 K 2/105.5mA_23.5kHz_pre30s_on5s_off235s_25000avgs_filtered_batchDecon.feather"
    warnings.filterwarnings("ignore", category=FutureWarning)
    fnames = []
    for fold in [fold for fold in P(folder).iterdir() if fold.is_dir()]:
        if any(
            [file for file in P(fold).iterdir() if file.suffix == ".feather"]
        ):
            fnames.append(
                *[
                    file
                    for file in P(fold).iterdir()
                    if file.suffix == ".feather"
                ]
            )
    if testing:
        fnames = [filename]
    for fname in tqdm(fnames):
        try:
            main(fname)
        # except ValueError:
        except RuntimeError:
            print(f"ERROR with file: {fname}")
