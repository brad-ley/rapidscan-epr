from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["science"])
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.linewidth": 1,
    "lines.linewidth": 2,
    "xtick.major.size": 5,
    "xtick.major.width": 1,
    "xtick.minor.size": 2,
    "xtick.minor.width": 1,
    "ytick.major.size": 5,
    "ytick.major.width": 1,
    "ytick.minor.size": 2,
    "ytick.minor.width": 1,
})

t_on  = 30.0
t_off = 40.0
t_max = 400.0
w_end = 10 / 100

t = np.linspace(0, t_max, 400)
w = (w_end) ** (np.abs(t - t_off) / (t_max - t_off))

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(t, w, "b-")

for x, label in [(0, "0"), (t_on, "$t_1$"), (t_off, "$t_2$"), (t_max, "$t_\\mathrm{max}$")]:
    ax.axvline(x, color="gray", linestyle="--", linewidth=0.8)
    ax.text(x, -0.08, label, ha="center", va="top", transform=ax.get_xaxis_transform())

ax.set_xlabel("Time (s)")
ax.set_ylabel("Weight")
ax.set_xlim(0, t_max * 1.05)
ax.set_ylim(0, 1.05)
ax.set_xticks([])

fig.tight_layout()

output_dir = Path(__file__).parent.parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "time_window.png", dpi=150)
plt.show()
