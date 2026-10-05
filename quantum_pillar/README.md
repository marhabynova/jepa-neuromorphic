# Quantum pillar — orientation stage

Third of the 3-pillar hybrid (von Neumann / neuromorphic / quantum). Started
2026-10-06, deliberately light-weight: tooling check + one orientation
circuit only. Deeper literature review and design are a separate, later
session (this pillar deserves the same grounding pass the neuromorphic
pillar got in `RESEARCH_NOTES.md` §65-65c, not a rushed add-on).

## Tooling choice: PennyLane, not Qiskit (for now)

[PennyLane](https://pennylane.ai/) was picked over Qiskit because it's
built for *differentiable* hybrid quantum-classical pipelines with a native
PyTorch interface — the rest of this project (`SpikingJEPA_V2`, the hxtorch
reconstruction) is already PyTorch-based, so a future hybrid
quantum+spiking pipeline would plug in more naturally. Qiskit remains an
option later specifically for real IBM Quantum hardware access (PennyLane
also has a Qiskit plugin for that, so this doesn't foreclose it).

## Honest scope limits (read before assuming anything about "quantum
advantage")

- Everything here runs as a **classical simulation** of a quantum circuit
  (PennyLane's `default.qubit` device), not on real quantum hardware.
  Classical simulation of an n-qubit circuit needs 2^n complex amplitudes,
  so this is only practical for small n (tens of qubits at most on a
  laptop) — nowhere near a scale where any claim of quantum speedup would
  be meaningful.
- The point of this stage is building working intuition and a tested
  toolchain, the same way `fpga_von_neumann/` started as RTL simulation
  before any real FPGA board, and the hxtorch work started in `mock=True`
  before any real BrainScaleS-2 hardware access. No capability or
  performance claims are made at this stage.

## Files

- `bell_state.py` — minimal orientation circuit: prepares a 2-qubit Bell
  state (maximal entanglement) and measures it, as a basic sanity check
  that the PennyLane install works and reproduces the textbook-expected
  50/50 `|00>`/`|11>` outcome distribution.
