# §63 delay-hypothesis validation

Independent cross-check (via [Brian2](https://brian2.readthedocs.io/), an
established, widely-used spiking-network simulator — not another hand-rolled
script) of the §62/§63 finding in `RESEARCH_NOTES.md`: that the residual gap
between the local `snnTorch` encoder and its `hxtorch` (BrainScaleS-2 mock)
reconstruction is explained by a **1-`dt` synaptic delay** that is part of
hxtorch's documented `cuba_aelif_integration` formula (membrane update uses
the synaptic current from *before* this step's input is added) but absent
from `snnTorch`'s `Synaptic` neuron (which updates current first, then uses
the fresh value the same step).

## Why Brian2 and not just another manual script

`tests/manual_reimplementation.py` already reproduced this numerically
(error 0.0021 against the real hxtorch-measured rates). That's one
hand-written simulator agreeing with another hand-written simulator sharing
the same author — useful but not independent. Brian2 is a separately
engineered, peer-reviewed-in-practice simulation engine with its own
scheduling, numerics and spike-detection code. If it reproduces the same
result using the *same* formula, that's much stronger evidence the formula
(not a shared bug) is what's driving the match.

## Files

- `brian2_crossvalidation.py` — 3-layer feedforward CUBA-LIF network built
  from `NeuronGroup` + `TimedArray` (external per-timestep current, since
  this architecture is rate/graded-current-coded, not sparse event-driven —
  there is no native `Synapses` object between layers; each layer's spike
  output is turned into the next layer's per-timestep injected current
  exactly as `fc(x)` would in the original PyTorch model). The 1-`dt` delay
  is toggled via Brian2's own `run_regularly(..., order=...)` scheduling,
  not by changing the underlying simulation loop — this is the knob being
  tested.
- `tests/manual_reimplementation.py` — the pure-PyTorch reimplementation
  from §61–63 (moved here from scratch, cleaned up), used as the first
  comparison point before trusting the Brian2 run.

## Running

```
cd validation/delay_hypothesis
../../.venv/bin/python3 brian2_crossvalidation.py
```
