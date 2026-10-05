"""
Orientation circuit for the quantum pillar (see README.md in this folder
for scope/honesty notes — this is a classical simulation, not real quantum
hardware, and makes no performance claims).

Prepares a 2-qubit Bell state |Phi+> = (|00> + |11>) / sqrt(2) via
Hadamard + CNOT, the standard first circuit anyone learning quantum
computing runs, specifically because its expected output is well-known
(exactly 50/50 between |00> and |11>, never |01> or |10>) and so makes a
good sanity check that the toolchain is set up correctly.
"""
import pennylane as qml
from pennylane import numpy as np

dev = qml.device('default.qubit', wires=2)


@qml.set_shots(1000)
@qml.qnode(dev)
def bell_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.counts()


if __name__ == '__main__':
    counts = bell_circuit()
    print("Measurement counts over 1000 shots:", dict(counts))

    total = sum(counts.values())
    p00 = counts.get('00', 0) / total
    p11 = counts.get('11', 0) / total
    p01 = counts.get('01', 0) / total
    p10 = counts.get('10', 0) / total
    print(f"P(00)={p00:.3f}  P(11)={p11:.3f}  P(01)={p01:.3f}  P(10)={p10:.3f}")
    print("Expected: P(00) ~= P(11) ~= 0.5, P(01) = P(10) = 0.0 "
          "(entanglement forbids the mixed outcomes entirely)")

    assert p01 == 0.0 and p10 == 0.0, "unexpected non-entangled outcome observed"
    print("\nSanity check passed: toolchain confirmed working.")
