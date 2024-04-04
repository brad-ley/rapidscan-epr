import matplotlib.pyplot as plt
import numpy as np

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


def main():
    # these are all CGS units
    GAMMA = -1.76e7
    g = 2
    bohr = 9.27e-21
    h = 6.626e-27

    def relaxation_broadening(t_corr, R, f):
        """relaxation_broadening. Returns broadening in Gauss

        :param t_corr: correlation time (ns)
        :param R: distance (nm)
        :param f: frequency of EPR (GHz)
        """
        t_corr *= 1e-9
        R = np.copy(R) * 1e-7
        f *= 1e9

        return (
            3
            / 10
            * GAMMA**4
            / (g * bohr)
            * (h / (2 * np.pi)) ** 3
            * t_corr
            / R**6
        )

    r = np.linspace(0.1, 10, 10000)

    def static_dipolar(r):
        """static_dipolar. Returns dipolar coupling strength in units of field (G).

        :param r: nm
        """
        mu0 = 1.257e-6  # N/A^2
        muB = 9.27e-24  # J/T
        h = 6.63e-34  # Js
        r = np.copy(r) * 1e-9
        return (
            mu0
            / (4 * np.pi)
            * g**2
            * muB**2
            / h
            * (1 / r) ** 3
            * 1e-6
        )

    def w(r, n):
        return 4 * np.pi * r**2 * n * np.exp(-4 * np.pi * r**3 * n / 3)

    def conc(n):
        # converts molarity into molecules/nm
        return n * 0.6023

    f, a = plt.subplots(figsize=(8, 6))
    c = 1.6e-3

    broad1 = relaxation_broadening(10, r, 240) * 2.8  # convert to MHz
    broad1 *= (w(r, c) > 0.05)
    a.plot(r, broad1, label='Relaxation')
    broad2 = static_dipolar(r)
    broad2 = broad2 ** 2 * 1 / (7)  # BPP 1948, Anderson 1954 eqn (1) for estimated values of w_p and w_e
    broad2 *= (w(r, c) > 0.05)
    print(max(broad1), max(broad2))
    a.plot(r, broad2, label='Static')
    # a.plot(r, broad2 / broad1 / 100, label=r'Static/Relax $0.01\times$')
    a.plot(r, broad2 / broad1, label=r'Static/Relax')
    a2 = a.twinx()
    a2.plot(r, w(r, c), label='$P(r)$', ls='--', c='k')
    a2.set_ylabel('$P(r)$')
    a2.legend()
    a.set_ylabel('Interaction (MHz)')
    a.set_xlabel('Distance (nm)')
    a.legend(loc='upper left')


if __name__ == "__main__":
    main()
    plt.savefig('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Group meetings/2024-03-15/compare_sim.png', dpi=1200)
    plt.show()
