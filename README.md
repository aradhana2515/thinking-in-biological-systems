[![CI](https://github.com/aradhana2515/thinking-in-biological-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/aradhana2515/thinking-in-biological-systems/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
# Thinking in Biological Systems

A lightweight Python toolkit for exploring qualitative dynamical regimes in biological models.

Small parameter changes can induce large qualitative shifts: steady states, oscillations, or runaway dynamics.
This repository provides minimal, transparent tools to simulate and classify those transitions.

## Install

```
git clone https://github.com/aradhana2515/thinking-in-biological-systems.git
cd thinking-in-biological-systems

python -m venv .venv
source .venv/bin/activate

python -m pip install -e .
python -m pip install matplotlib
```

## Qick Example

```
from tibs.models import Repressilator

m = Repressilator()
res = m.simulate(t_span=(0, 80), dt=0.02)
m.plot(res)
```

## Model Zoo

- ToggleSwitch — bistability
- Repressilator — sustained oscillations
- SIR — epidemic threshold dynamics

All models subclass `ODEModel` and expose:

- `simulate()`
- `sweep()`
- `plot()`

## Regime Classification

Automatically label time-series behavior:
```
from tibs.analysis import classify_timeseries
from tibs.models import Repressilator

m = Repressilator()
res = m.simulate(t_span=(0, 80), dt=0.02)

print(classify_timeseries(res.x))
```

Returns one of:

- `steady`
- `oscillatory`
- `blowup`
- `other`

## Regime Maps

Explore qualitative transitions across parameter space:

```
python examples/repressilator_regime_map.py
```

This generates a heatmap of dynamical regimes as parameters vary.

## Design Principles

- Minimal dependencies
- Explicit numerical methods (RK4 integrator)
- Small, readable model definitions
- Emphasis on qualitative behavior

## Extending

Add a new model by subclassing 'ODEModel' and implementing:

```
initial_state()
parameters()
rhs(t, x, params)
```

## License

MIT
