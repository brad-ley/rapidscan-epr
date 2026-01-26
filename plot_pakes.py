import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def main(fnames):
    fix, ax = plt.subplots()
    for fname in fnames:
        dat = np.loadtxt(fname, delimiter=',')
        x = dat[0, :]
        for ind in range(1, 8):
            y = dat[dat.shape[0] // 8 * ind, :]/np.max(dat[dat.shape[0] // 8 * ind, :]) - ind
            ax.plot(x, y, label=f"{fname.stem}_ind{ind}")
    ax.legend()

if __name__ == "__main__":
	basepath = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/"
	fnames = [Path(basepath).joinpath(
		"Code/dipolar averaging/tumbling_7-2_7-2_10mT_unlike-g_12.4ns_tcorr.txt"),
		Path(basepath).joinpath("Code/dipolar averaging/tumbling_pake_1-2_7-2_unlike_morebaseline_13.8ns_tcorr.txt"),]
	main(fnames)
	plt.show()