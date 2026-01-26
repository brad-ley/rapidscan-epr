from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def main():
    pake_data = np.loadtxt(pake_patterns, delimiter=",")
    field_data = pake_data[0, :].copy()
    field_data = 1e3 * (field_data - np.mean(field_data))
    pake_data = pake_data[1:, :]

    integrals = np.trapz(pake_data, axis=1)
    plt.plot(integrals)

    # for ind, row in enumerate(pake_data[:: pake_data.shape[0] // 8, :]):
    #     plt.plot(field_data, row / np.max(row) - ind)
    #     # plt.plot(row / np.max(row) - ind)


if __name__ == "__main__":
    basepath = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/"
    pake_patterns = Path(basepath).joinpath(
        "Code/dipolar averaging/tumbling_pake_1-2_7-2_unlike_morebaseline_13.8ns_tcorr.txt",
    )
    main()
    plt.show()
