from tibs.models import SIR

m = SIR()
res = m.simulate(t_span=(0, 160), dt=0.1)
m.plot(res)
