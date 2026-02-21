from tibs.models import ToggleSwitch

m = ToggleSwitch()
res = m.simulate(t_span=(0, 50), dt=0.02)
m.plot(res)
