#!/usr/bin/env python3
"""Tests the two new instructions (§95): JALR (real call/return convention)
and LUI (build a 32-bit constant without the old addi+sll shift trick).

Program: a genuine CALL/RETURN round trip - JAL saves the return address in
x1 (the standard RISC-V "ra" register convention), the callee does real
work (x11 = x10+100), then JALR returns to whatever JAL actually computed
(not a hardcoded address) - the whole point of JALR over JAL alone. If this
works, x12 gets set to 999 AFTER the call returns, proving execution
resumed at the correct point rather than falling through/crashing.
Also tests LUI standalone: x20 should become 0x12345000.
"""
import sys
sys.path.insert(0, ".")
from asm import Program, OP_ITYPE, OP_JAL, i_type, j_type

OP_JALR = 0b1100111
OP_LUI = 0b0110111


def u_type(imm20, rd, opcode):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


p = Program()

# LUI test: x20 = 0x12345000
p.emit(lambda a, L: u_type(0x12345, 20, OP_LUI))

# CALL/RETURN test
p.emit(lambda a, L: i_type(5, 0, 0b000, 10, OP_ITYPE))   # addi x10, x0, 5
p.emit(lambda a, L: j_type(L['func'] - a, 1, OP_JAL))     # jal  x1, func   (x1 = return addr = here+4)
p.emit(lambda a, L: i_type(999, 0, 0b000, 12, OP_ITYPE))  # addi x12, x0, 999  (must run AFTER return)
p.emit(lambda a, L: j_type(L['halt'] - a, 0, OP_JAL))     # jal  x0, halt

p.label('func')
p.emit(lambda a, L: i_type(100, 10, 0b000, 11, OP_ITYPE))  # addi x11, x10, 100  -> x11=105
p.emit(lambda a, L: i_type(0, 1, 0b000, 0, OP_JALR))       # jalr x0, 0(x1)      -> return

p.label('halt')
p.emit(lambda a, L: j_type(L['halt'] - a, 0, OP_JAL))

if __name__ == "__main__":
    words = p.assemble()
    with open("program_jalr_lui.hex", "w") as f:
        for w in words:
            f.write(f"{w:08x}\n")
    print(f"wrote {len(words)} instructions to program_jalr_lui.hex")
