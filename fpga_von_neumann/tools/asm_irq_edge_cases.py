#!/usr/bin/env python3
"""§99: swept-irq-timing program for the Opus audit's A1/A2 regression
(RESEARCH_NOTES.md §99, recommendation C1). Doesn't test timing by varying
delays in software - the testbench (tb_irq_edge_cases.v) injects the two
interrupt pulses at EXACT cycles found by watching cpu.v's own internal
control signals (irq_dis/exec_now for the DI race, irq_en/exec_now for the
handler's own EI right before its IRET), so the race windows are hit
deterministically rather than by sweeping delay values and hoping to land
on the right cycle.

Handler is deliberately just [EI, IRET] - two instructions back to back, no
counter instruction in between - so that a pulse injected exactly on the
handler's EI cycle becomes pending+enabled exactly one cycle later, which
is the handler's own IRET cycle (the A2 race). Handler-invocation count is
tracked by the testbench watching pc==4 at a FETCH boundary, not by a
register the handler writes - keeps the handler body length fixed
regardless of which race it's used for.
"""
import sys
sys.path.insert(0, ".")
from asm import Program, OP_ITYPE, OP_BRANCH, OP_JAL, i_type, b_type, j_type

OP_CUSTOM0 = 0b0001011


def custom0(funct3):
    return i_type(0, 0, funct3, 0, OP_CUSTOM0)


IRET = custom0(0b000)
EI = custom0(0b001)
DI = custom0(0b010)

p = Program()

# addr 0: skip the IRQ_VECTOR slot.
p.emit(lambda a, L: j_type(L['main'] - a, 0, OP_JAL))

# addr 4 (IRQ_VECTOR): handler - EXACTLY EI then IRET, nothing else.
p.emit(lambda a, L: EI)       # addr 4
p.emit(lambda a, L: IRET)     # addr 8

p.label('main')
p.emit(lambda a, L: EI)                                    # addr 12 - initial enable

# --- Phase 1 (A1): delay, then DI. Testbench injects irq EXACTLY on DI's
# own exec_now cycle - correct behavior defers it (handler must NOT run
# until the re-EI below), buggy behavior dispatches it immediately.
p.emit(lambda a, L: i_type(20, 0, 0b000, 7, OP_ITYPE))      # addi x7,x0,20
p.label('loop1')
p.emit(lambda a, L: i_type(-1, 7, 0b000, 7, OP_ITYPE))      # addi x7,x7,-1
p.emit(lambda a, L: b_type(L['loop1'] - a, 0, 7, 0b001, OP_BRANCH))  # bne x7,x0,loop1
p.emit(lambda a, L: DI)                                      # DI - A1 race point

p.emit(lambda a, L: i_type(10, 0, 0b000, 7, OP_ITYPE))      # post-DI observation window
p.label('loop2')
p.emit(lambda a, L: i_type(-1, 7, 0b000, 7, OP_ITYPE))
p.emit(lambda a, L: b_type(L['loop2'] - a, 0, 7, 0b001, OP_BRANCH))
p.emit(lambda a, L: i_type(111, 0, 0b000, 21, OP_ITYPE))    # marker A: DI window closed

p.emit(lambda a, L: EI)                                      # re-enable - deferred irq should fire now
p.emit(lambda a, L: i_type(10, 0, 0b000, 7, OP_ITYPE))
p.label('loop3')
p.emit(lambda a, L: i_type(-1, 7, 0b000, 7, OP_ITYPE))
p.emit(lambda a, L: b_type(L['loop3'] - a, 0, 7, 0b001, OP_BRANCH))
p.emit(lambda a, L: i_type(222, 0, 0b000, 22, OP_ITYPE))    # marker B: phase 1 resolved

# --- Phase 2 (A2): testbench injects a fresh irq during loop4 (ordinary,
# non-racy - just to get the handler running), watches for the handler's
# OWN EI (unambiguous from here on - main has no more EI instructions) and
# injects a SECOND irq exactly then - lands on the handler's IRET cycle.
p.emit(lambda a, L: i_type(20, 0, 0b000, 7, OP_ITYPE))
p.label('loop4')
p.emit(lambda a, L: i_type(-1, 7, 0b000, 7, OP_ITYPE))
p.emit(lambda a, L: b_type(L['loop4'] - a, 0, 7, 0b001, OP_BRANCH))
p.emit(lambda a, L: i_type(333, 0, 0b000, 23, OP_ITYPE))    # marker C: A2 race resolved

p.emit(lambda a, L: i_type(10, 0, 0b000, 7, OP_ITYPE))      # window for the preserved irq to land
p.label('loop5')
p.emit(lambda a, L: i_type(-1, 7, 0b000, 7, OP_ITYPE))
p.emit(lambda a, L: b_type(L['loop5'] - a, 0, 7, 0b001, OP_BRANCH))
p.emit(lambda a, L: i_type(444, 0, 0b000, 24, OP_ITYPE))    # marker D: final

p.label('halt')
p.emit(lambda a, L: j_type(L['halt'] - a, 0, OP_JAL))

if __name__ == "__main__":
    words = p.assemble()
    with open("program_irq_edge_cases.hex", "w") as f:
        for w in words:
            f.write(f"{w:08x}\n")
    print(f"wrote {len(words)} instructions to program_irq_edge_cases.hex")
