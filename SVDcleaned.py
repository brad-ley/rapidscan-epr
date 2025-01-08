import ast
import sys
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
    # rc("text.latex", preamble=r"\usepackage{cmbright}")
    rcParams = [
        ["font.family", "serif"],
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
    figsize = (6, 4)
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
    return (data - double_lorentzian(params, B)) ** 2 * lorentzian_sum(
        params, B
    ) ** 2


def lorentzian_sum(params, x):
    """
    Need to normalize both to make sure the total number of spins going from
    one to the other is conserved
    """
    return 1


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
    # b2 = b1
    w1 = parvals["w1"]
    w2 = parvals["w2"]
    return lorentzian(x, x0, a1, b1, w1) - lorentzian(x, x0, a2, b2, w2)


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

        nearcenter = (
            self.low + (self.high - self.low) // 2
        )  # should be near middle of field
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
            if (
                np.abs(nearcenter - center) / nearcenter < 0.2
            ):  # if there is a big variation don't bother
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

        ratios = np.array(
            [
                E[idx] / E[idx + 1] if idx + 1 < len(E) else 0
                for idx, _ in enumerate(E)
            ]
        )
        lim = 1.5
        if not k:
            k = np.where(ratios < lim)[0][
                0
            ]  # find first time it goes below 1.5

        # if k == 0:
        if k < 2:
            k = 2
        self.k = k

        self.fscree, self.ascree = plt.subplots()
        max = 10
        self.ascree.scatter(range(1, max + 1), ratios[:max], c="k")
        self.ascree.set_yscale("log")
        self.ascree.set_ylabel(r"$\sigma_{n}/\sigma_{n+1}$")
        self.ascree.set_xlabel("Component number")
        self.ascree.set_xticks(range(1, max, 5))
        self.ascree.axhline(lim, alpha=0.5, ls="--", c="gray")

        self.U = U[:, :k]
        self.E = E[:k]
        self.Vh = Vh[:k, :]
        self.subbasis_data = self.U @ np.diag(self.E) @ self.Vh
        return self

    def plotMatrix(self, subbasis=False):
        """plotMatrix.

        :param subbasis: if True, show imshow image of the data recreated with k principal components
        """
        if subbasis:
            self.subbasis = True
            dat = self.subbasis_data
            self.imshow_sim, self.a_sim = plt.subplots()
            a = self.a_sim
            self.a_sim.set_title(
                f"PCA ($k = {self.k}$) sub-basis recreated data"
            )
        else:
            dat = self.dat
            self.imshow_raw, self.a_raw = plt.subplots()
            a = self.a_raw

        if hasattr(self, "mean_centered"):
            # dat += self.mu[:, np.newaxis]  # type: ignore
            dat += 0
        a.imshow(dat, aspect="auto")
        a.set_xlabel("Time (s)")
        a.set_ylabel("Intensity (arb. u)")

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
            nrows=self.Vh.shape[0],
            figsize=figsize,
            squeeze=False,
            sharex=True,
            layout="constrained",
        )
        self.lsvf.text(0.01, 0.975, "a)", fontsize=20)
        for idx, u in enumerate(self.U[0, :]):
            (line,) = self.lsva[idx, 0].plot(
                self.B,
                self.U[:, idx],
                label=rf"$\mathbf{{u}}_{{{idx + 1}}}$",
                alpha=0.5,
                c="k",
            )
            find = 2
            if hasattr(self, "mean_centered"):
                find = 1
            try:
                if idx + 1 == find - 1:
                    popt, pcov = curve_fit(lorentzian, self.B, self.U[:, idx])
                    err = 2 * np.sqrt(np.diag(pcov))
                    self.lsva[idx, 0].plot(
                        self.B,
                        lorentzian(self.B, *popt),
                        c=line.get_color(),
                        ls="--",
                        label=rf"$\omega={popt[-1]:.2f}\,$G",
                    )
                    print("single", popt, pcov)
                    print("single", err)
                    print("single", err[-1])
                if idx + 1 == find:
                    # print(
                    #     "Data integrals: (raw, absolute value)",
                    #     np.trapz(self.U[:, idx]),
                    #     np.trapz(np.abs(self.U[:, idx])),
                    # )
                    params = lmfit.create_params(
                        x0=dict(
                            value=0.0,
                            vary=True,
                            min=-np.max(self.B) / 4,
                            max=np.max(self.B) / 4,
                        ),
                        a1=dict(
                            value=0,
                            vary=True,
                            min=-np.max(self.U[:, idx]),
                            max=np.max(self.U[:, idx]),
                        ),
                        a2=dict(
                            value=0,
                            vary=not hasattr(self, "mean_centered"),
                            min=-np.max(self.U[:, idx]),
                            max=np.max(self.U[:, idx]),
                        ),
                        b1=dict(
                            value=0.0,
                            vary=True,
                            # min=0.1,
                            # max=np.max(self.E),
                        ),
                        b2=dict(
                            # value=-0.1,
                            value=0.0,
                            vary=not hasattr(self, "mean_centered"),
                            # min=-np.max(self.E),
                            # max=-0.1,
                        ),
                        w1=dict(value=5, vary=True, min=3, max=np.max(self.B)),
                        w2=dict(
                            value=12,
                            vary=not hasattr(self, "mean_centered"),
                            min=3,  # max=np.max(self.B)
                        ),
                    )

                    obj = lmfit.Minimizer(
                        fit_fun,
                        params,
                        fcn_args=(self.U[:, idx], self.B),
                    )
                    self.res = obj.minimize(method="leastsquares")
                    parvals = self.res.params.valuesdict()  # type: ignore
                    print([(key, parvals[key]) for key in parvals])
                    print(self.res.covar)
                    print(2 * np.sqrt(np.diag(self.res.covar)))  # type: ignore

                    x0 = parvals["x0"]
                    a1 = parvals["a1"]
                    a2 = parvals["a2"]
                    b1 = parvals["b1"]
                    b2 = parvals["b2"]
                    # b2 = b1
                    w1 = parvals["w1"]
                    w2 = parvals["w2"]
                    # print(
                    #     "Fit vars",
                    #     a1,
                    #     b1,
                    #     w1,
                    #     "\n",
                    #     a2,
                    #     b2,
                    #     w2,
                    #     "\n",
                    #     "Fit integral",
                    #     np.trapz(double_lorentzian(self.res.params, self.B)),  # type: ignore
                    # )

                    self.lsva[idx, 0].plot(
                        self.B,
                        double_lorentzian(self.res.params, self.B),  # type: ignore
                        c=line.get_color(),
                        ls="--",
                        # label=rf"$\omega={popt[-1]:.2f}\,$G",
                        # label=r"$F_{\omega_1}+$"
                        # r"$F_{\omega_2}$"
                        # "\n"
                        # rf"(${w1:.2f}\,$G; "
                        # rf"${w2:.2f}\,$G)",
                    )
                    self.lsva[idx, 0].plot(
                        self.B,
                        lorentzian(self.B, x0, a1, b1, w1),
                        # c=line.get_color(),
                        ls="--",
                        label=rf"$\omega_1={w1:.1f}\,$G",
                    )
                    self.lsva[idx, 0].plot(
                        self.B,
                        -1 * lorentzian(self.B, x0, a2, b2, w2),
                        # c=line.get_color(),
                        ls="--",
                        label=rf"$\omega_2={w2:.1f}\,$G",
                    )
                    self.lsva[idx, 0].legend(loc="lower right")

            except RuntimeError:
                print(f"Could not fit w_{idx + 1} component")
            if idx == 2:
                self.lsva[idx, 0].legend(
                    loc="upper right",
                    handlelength=0.75,
                    labelspacing=0.25,
                )
            else:
                self.lsva[idx, 0].legend(
                    loc="right",
                    handlelength=0.75,
                    labelspacing=0.25,
                )
        self.lsvf.supxlabel("Field (G)")
        self.lsvf.supylabel("Intensity (arb. u)")
        # self.lsvf.tight_layout()
        return self

    def plotRSVs(self):
        self.rsvf, self.rsva = plt.subplots(
            nrows=self.Vh.shape[0],
            figsize=figsize,
            squeeze=False,
            sharex=True,
            layout="constrained",
        )
        self.rsvf.text(0.01, 0.975, "b)", fontsize=20)
        find = 2
        if hasattr(self, "mean_centered"):
            find = 1
        for idx, v in enumerate(self.Vh[:, 0]):
            (line,) = self.rsva[idx, 0].plot(
                self.times,
                # self.E[idx] * self.Vh[idx, :] + 0.0 * idx,
                np.sign(self.E[idx]) * self.Vh[idx, :] + 0.0 * idx,
                label=rf"$\mathbf{{v}}_{{{idx + 1}}}(t)$",
                alpha=0.5,
                c="k",
            )
            try:
                popt, pcov = curve_fit(
                    exp,
                    self.times[self.times > self.pre + self.on]
                    - (self.pre + self.on),
                    # self.E[idx]
                    np.sign(self.E[idx])
                    * self.Vh[idx, :][self.times > self.pre + self.on],
                    p0=[
                        np.min(self.Vh[idx, :]),
                        np.max(self.Vh[idx, :]),
                        np.max(self.times) / 2,
                    ],
                )
                err = 2 * np.sqrt(np.diag(pcov))
                if err[-1] == np.inf:
                    error_bar = "error"
                else:
                    error_bar = f"{err[-1]:.1f}"

                if err[-1] < 100:
                    self.rsva[idx, 0].plot(
                        self.times[self.times > self.pre + self.on],
                        exp(
                            self.times[self.times > self.pre + self.on]
                            - (self.pre + self.on),
                            *popt,
                        )
                        + 0.0 * idx,
                        # label=rf"$\tau={popt[-1]:.1f}\pm${error_bar}$\,$s",
                        label=rf"$\tau={popt[-1]:.1f}\,$s",
                        c=line.get_color(),
                        ls="--",
                    )
                if idx + 1 == find:
                    self.popt = popt
                    self.err = err

                print("tau", error_bar)
            except RuntimeError:
                print(f"Could not fit w_{idx + 1} component")
            try:
                self.rsva[idx, 0].axvspan(
                    self.pre,
                    self.pre + self.on,
                    facecolor="#00A7CA",
                    alpha=0.25,
                    label="Laser on",
                )
            except ValueError:
                pass
            self.rsva[idx, 0].legend(
                loc="right",
                handlelength=0.75,
                labelspacing=0.25,
            )
        self.rsvf.supxlabel("Time (s)")
        # self.rsvf.supylabel(r"Intensity (weighted by $\Sigma_i$)")
        self.rsvf.supylabel(r"Intensity (arb. u)")
        # self.rsvf.tight_layout()
        return self

    def saveResults(self):
        path = P(self.filename).parent.joinpath("SVD/")
        if not path.exists():
            path.mkdir()

        add = ""
        if hasattr(self, "sim_as_data"):
            add += f"sim-{self.sim_components}components_"
        if hasattr(self, "mean_centered"):
            add += "mean-centered_"
        if hasattr(self, "rsvf"):
            self.rsvf.savefig(path.joinpath(f"{add}SVDweights.png"), dpi=1200)
        if hasattr(self, "lsvf"):
            self.lsvf.savefig(path.joinpath(f"{add}SVDvectors.png"), dpi=1200)
        if hasattr(self, "imshow_sim"):
            if hasattr(self, "subbasis"):
                add += "subbasis_"
            self.imshow_sim.savefig(
                path.joinpath(f"{add}imshow_sim.png"), dpi=1200
            )
        if hasattr(self, "imshow_raw"):
            if hasattr(self, "subbasis"):
                add += "subbasis_"
            self.imshow_raw.savefig(
                path.joinpath(f"{add}imshow_raw.png"), dpi=1200
            )
        if hasattr(self, "fscree"):
            self.fscree.savefig(path.joinpath("scree.png"), dpi=1200)

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
        linewidths: List[float] = [9],
        ratio_to_first: List[Tuple[float, float]] = [(1, -1)],
        naive: bool = False,
        tau: List[float] | float = 150,
    ):
        """simulate.

        :param linewidths: [linewidth_1, linewidth_n] input width for each Lorenztian that you want to appear in the time-dependent lineshape simulation
        :type linewidths: List[Tuple[float, float]]
        :param ratio_to_first: [(ratio11_start, ratio11_delta),... , (ration1_start, ration1_delta)]
        if naive is True, ratio_to_first[0][1] is the amount the linewidth will change in linewidth units and ratio_to_first[0][0] is unbound
        :type ratio_to_first: List[Tuple[float, float]]
        :param naive: runs the simulation with only linewidth that is changing in time, not multiple populations
        :type naive: bool
        :param tau: relaxation decay constants for each component, input can be list with a tau for each component or single float used for all of them
        :type naive: List[float]|float
        """
        if ratio_to_first[0][0] != 1:
            raise Exception("ratio_to_first of first must be 1")
        if len(ratio_to_first) != len(linewidths):
            raise Exception(
                "Linewidth and ratio_to_first list lengths do not match"
            )
        if type(tau) is list:
            taus = np.copy(tau)
            if len(tau) != len(linewidths):
                raise Exception(
                    "Tau and ratio_to_first list lengths do not match"
                )
        else:
            taus = [tau] * len(linewidths)
        if naive and len(linewidths) > 1:
            raise Exception(
                "Naive takes only one linewidth and delta as input"
            )

        if naive:
            linewidth = linewidths + np.concatenate(
                (
                    np.zeros(np.where(self.times > (self.pre))[0][0]),
                    exp(
                        self.times[self.times > (self.pre)] - (self.pre),
                        0,
                        ratio_to_first[0][1],
                        taus[0],
                    ),
                )
            )
            self.simulated = np.zeros((self.B.shape[0], self.times.shape[0]))
            for idx, lw in enumerate(linewidth):
                self.simulated[:, idx] = lorentzian(self.B, 0, 0, 1, lw)

        else:
            lw_matrix = np.zeros((self.B.shape[0], len(linewidths)))

            for idx, linewidth in enumerate(linewidths):
                lw_matrix[:, idx] = lorentzian(self.B, 0, 0, 1, linewidth)

            t_matrix = np.zeros((len(ratio_to_first), self.times.shape[0]))
            for idx, (ratio, ratio_delta) in enumerate(ratio_to_first):
                t_matrix[idx, :] = ratio + np.concatenate(
                    (
                        np.zeros(np.where(self.times > (self.pre))[0][0]),
                        exp(
                            self.times[self.times > (self.pre)] - (self.pre),
                            0,
                            ratio_delta,
                            taus[idx],
                        ),
                    )
                )

                if idx:  # make sure that it isn't the first component, then subtract the weight of the 2-nth component
                    t_matrix[0, :] -= t_matrix[idx, :]

            self.simulated = lw_matrix @ t_matrix
        self.sim_components = len(linewidths)
        return self

    def use_simulate_as_data(self, **kwargs):
        if not hasattr(self, "simulated"):
            self.simulate(**kwargs)
        self.sim_as_data = True
        self.dat = self.simulated
        return self

    def normalize(self):
        def fwhm(x, y):
            low = np.where(y > 0.5 * np.max(y))[0][0]
            high = np.where(y > 0.5 * np.max(y))[0][-1]
            return x[high] - x[low]

        for col in range(self.dat.shape[1]):
            popt, pcov = curve_fit(lorentzian, self.B, self.dat[:, col])
            self.dat[:, col] -= popt[1]
            self.dat[:, col] *= (
                2 / (np.pi * popt[-1]) / np.trapz(lorentzian(self.B, *popt))
            )  # normalize to get peak to 2 / pi * FWHM
            # self.dat[:, col] /= np.max(
            #     lorentzian(
            #         self.B, popt[0], 0, *popt[2:]
            #     )  # force the offset to zero
            # )

        return self

    def zero_pad(self, n=1):
        self.dat = np.pad(
            self.dat,
            pad_width=((self.dat.shape[0] * n, self.dat.shape[0] * n), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        self.B = np.linspace(
            -(n + 1) * np.min(self.B),
            (n + 1) * np.max(self.B),
            self.dat.shape[0],
        )
        return self


def clean_input(folder):
    print(folder)
    if P(folder).is_dir():
        fnames = []
        for fold in [fold for fold in P(folder).iterdir() if fold.is_dir()]:
            if any(
                [
                    file
                    for file in P(fold).iterdir()
                    if file.suffix == ".feather"
                ]
            ):
                fnames.append(
                    *[
                        file
                        for file in P(fold).iterdir()
                        if file.suffix == ".feather"
                    ]
                )
    else:
        if not P(folder).stem.endswith("Decon"):
            folder = P(folder).parent.joinpath(
                P(folder).stem + "_batchDecon.feather"
            )
        fnames = [folder]
    return fnames


def main(filename):
    DS = DataSet(filename).center()
    DS.normalize()
    DS.simulate(
        linewidths=[5, 16],
        ratio_to_first=[(1, 0), (0.95, -0.40)],
        # linewidths=[10],
        # ratio_to_first=[(1, -2)],
        # naive=True,
    )
    # DS.use_simulate_as_data()
    # DS.wrap(wrap_time=1064)
    # DS.mean_center()
    DS.svd()
    DS.plotSVs()
    DS.plotMatrix()
    DS.saveResults()
    # DS.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        folder = sys.argv[1]
    except IndexError:
        folder = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/2/26/T406C"
    fnames = clean_input(folder)
    for fname in tqdm(fnames):
        try:
            main(fname)
        # except ValueError:
        except IndexError:
            print(f"ERROR with file: {fname}")
        plt.close()
