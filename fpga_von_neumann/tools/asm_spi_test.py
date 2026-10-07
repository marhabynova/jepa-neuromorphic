#!/usr/bin/env python3
"""Small end-to-end program: the CPU itself (not the testbench driving the
bus directly) configures spi_master, starts an 8-bit transfer, polls STATUS
until done, then writes the received byte out through the MMIO debug port
so a testbench can check it without peeking at internal signals - the
closest thing to "did a real program running on this CPU actually talk to
the SPI peripheral" that's checkable without real hardware.

Reuses asm.py's instruction encoders (not a separate assembler).
"""
import sys
sys.path.insert(0, ".")
from asm import Program, OP_ITYPE, OP_RTYPE, OP_LOAD, OP_STORE, OP_BRANCH, OP_JAL, i_type, r_type, s_type, b_type, j_type

p = Program()

# x16 = 0xFF000000 (MMIO base), same "shift a byte into place" trick asm.py uses.
p.emit(lambda a, L: i_type(255, 0, 0b000, 14, OP_ITYPE))   # addi x14, x0, 255
p.emit(lambda a, L: i_type(24, 0, 0b000, 15, OP_ITYPE))    # addi x15, x0, 24
p.emit(lambda a, L: r_type(0, 15, 14, 0b001, 16, OP_RTYPE))  # sll x16, x14, x15

p.emit(lambda a, L: i_type(2, 0, 0b000, 1, OP_ITYPE))      # addi x1, x0, 2      (CLKDIV value)
p.emit(lambda a, L: s_type(0x14, 1, 16, 0b010, OP_STORE))  # sw  x1, 0x14(x16)   CLKDIV

p.emit(lambda a, L: i_type(0, 0, 0b000, 2, OP_ITYPE))      # addi x2, x0, 0      (CTRL: mode0, START=0)
p.emit(lambda a, L: s_type(0x10, 2, 16, 0b010, OP_STORE))  # sw  x2, 0x10(x16)   CTRL

p.emit(lambda a, L: i_type(165, 0, 0b000, 3, OP_ITYPE))    # addi x3, x0, 165=0xA5
p.emit(lambda a, L: s_type(0x18, 3, 16, 0b010, OP_STORE))  # sw  x3, 0x18(x16)   TXDATA

p.emit(lambda a, L: i_type(1, 0, 0b000, 4, OP_ITYPE))      # addi x4, x0, 1      (START=1)
p.emit(lambda a, L: s_type(0x10, 4, 16, 0b010, OP_STORE))  # sw  x4, 0x10(x16)   CTRL (go)

p.label('poll_loop')
p.emit(lambda a, L: i_type(0x1C, 16, 0b010, 5, OP_LOAD))   # lw  x5, 0x1C(x16)   STATUS
p.emit(lambda a, L: i_type(1, 5, 0b111, 5, OP_ITYPE))      # andi x5, x5, 1      (mask BUSY)
p.emit(lambda a, L: b_type(L['poll_loop'] - a, 0, 5, 0b001, OP_BRANCH))  # bne x5, x0, poll_loop

p.emit(lambda a, L: i_type(0x20, 16, 0b010, 6, OP_LOAD))   # lw  x6, 0x20(x16)   RXDATA
p.emit(lambda a, L: s_type(0, 6, 16, 0b010, OP_STORE))     # sw  x6, 0(x16)      debug port <= RXDATA

p.label('halt')
p.emit(lambda a, L: j_type(L['halt'] - a, 0, OP_JAL))

if __name__ == "__main__":
    words = p.assemble()
    with open("program_spi.hex", "w") as f:
        for w in words:
            f.write(f"{w:08x}\n")
    print(f"wrote {len(words)} instructions to program_spi.hex")
