import numpy as np
import matplotlib.pyplot as plt
from tibs.models import Repressilator
from tibs.analysis import classify_timeseries

alphas = np.linspace(1, 30, 25)
ns = np.linspace(1, 4, 20)

label_to_int = {"steady": 0, "oscillatory": 1, "other": 2, "blowup": 3}
grid = np.zeros((len(ns), len(alphas)), dtype=int)

m = Repressilator()
for i, n in enumerate(ns):
    for j, a in enumerate(alphas):
        res = m.simulate(t_span=(0, 80), dt=0.02, params={"alpha": float(a), "n": float(n)})
        lab = classify_timeseries(res.x).label
        grid[i, j] = label_to_int[lab]

plt.figure()
plt.imshow(
    grid,
    aspect="auto",
    origin="lower",
    extent=[alphas.min(), alphas.max(), ns.min(), ns.max()],
)
plt.xlabel("alpha")
plt.ylabel("n")
plt.title("Repressilator regime map (0=steady,1=osc,2=other,3=blowup)")
plt.tight_layout()
plt.show()
