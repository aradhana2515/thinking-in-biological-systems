import matplotlib.pyplot as plt

from tibs.models import Repressilator

m = Repressilator()
res = m.simulate(t_span=(0, 80), dt=0.02)
m.plot(res)

plt.show()
