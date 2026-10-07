#!/usr/bin/env python3
"""§98: exercises SPI and GPIO in the SAME program, interleaved, to prove
the generic peripheral bus genuinely supports more than one peripheral at
once (disjoint address decode: SPI 0x10-0x20, GPIO 0x30/0x34) rather than
just working because only one peripheral was ever wired up before.

Sequence: write GPIO_OUT, read GPIO_IN (testbench drives a known pattern),
THEN configure+run a full SPI transfer (polling, same as asm_spi_test.py),
THEN write to GPIO_OUT again with a different value and read GPIO_IN
again - checking that GPIO's own state wasn't corrupted by the SPI
transaction in between, and that SPI's result is still correct despite
GPIO traffic before and after it.
"""
import sys
sys.path.insert(0, ".")
from asm import Program, OP_ITYPE, OP_RTYPE, OP_LOAD, OP_STORE, OP_BRANCH, OP_JAL, i_type, r_type, s_type, b_type, j_type

p = Program()

p.emit(lambda a, L: i_type(255, 0, 0b000, 14, OP_ITYPE))   # addi x14, x0, 255
p.emit(lambda a, L: i_type(24, 0, 0b000, 15, OP_ITYPE))    # addi x15, x0, 24
p.emit(lambda a, L: r_type(0, 15, 14, 0b001, 16, OP_RTYPE))  # sll x16, x14, x15 (MMIO base)

# GPIO round 1: write 0xAB, read back whatever the testbench drives on gpio_in
p.emit(lambda a, L: i_type(0xAB, 0, 0b000, 5, OP_ITYPE))   # addi x5, x0, 0xAB
p.emit(lambda a, L: s_type(0x30, 5, 16, 0b010, OP_STORE))  # sw x5, 0x30(x16)  GPIO_OUT
p.emit(lambda a, L: i_type(0x34, 16, 0b010, 6, OP_LOAD))   # lw x6, 0x34(x16)  GPIO_IN -> x6

# SPI: CLKDIV=2, mode0, TXDATA=0xA5, START=1, poll STATUS, read RXDATA
p.emit(lambda a, L: i_type(2, 0, 0b000, 1, OP_ITYPE))
p.emit(lambda a, L: s_type(0x14, 1, 16, 0b010, OP_STORE))
p.emit(lambda a, L: i_type(0, 0, 0b000, 2, OP_ITYPE))
p.emit(lambda a, L: s_type(0x10, 2, 16, 0b010, OP_STORE))
p.emit(lambda a, L: i_type(165, 0, 0b000, 3, OP_ITYPE))
p.emit(lambda a, L: s_type(0x18, 3, 16, 0b010, OP_STORE))
p.emit(lambda a, L: i_type(1, 0, 0b000, 4, OP_ITYPE))
p.emit(lambda a, L: s_type(0x10, 4, 16, 0b010, OP_STORE))
p.label('poll_loop')
p.emit(lambda a, L: i_type(0x1C, 16, 0b010, 8, OP_LOAD))   # lw x8, 0x1C(x16) STATUS
p.emit(lambda a, L: i_type(1, 8, 0b111, 8, OP_ITYPE))      # andi x8, x8, 1
p.emit(lambda a, L: b_type(L['poll_loop'] - a, 0, 8, 0b001, OP_BRANCH))
p.emit(lambda a, L: i_type(0x20, 16, 0b010, 9, OP_LOAD))   # lw x9, 0x20(x16) RXDATA -> x9

# GPIO round 2: write a DIFFERENT value, read again - proves GPIO_OUT's
# own state is independent of whatever SPI was doing in between.
p.emit(lambda a, L: i_type(0x55, 0, 0b000, 10, OP_ITYPE))  # addi x10, x0, 0x55
p.emit(lambda a, L: s_type(0x30, 10, 16, 0b010, OP_STORE))  # sw x10, 0x30(x16) GPIO_OUT
p.emit(lambda a, L: i_type(0x34, 16, 0b010, 11, OP_LOAD))  # lw x11, 0x34(x16) GPIO_IN -> x11

p.label('halt')
p.emit(lambda a, L: j_type(L['halt'] - a, 0, OP_JAL))

if __name__ == "__main__":
    words = p.assemble()
    with open("program_multi_periph.hex", "w") as f:
        for w in words:
            f.write(f"{w:08x}\n")
    print(f"wrote {len(words)} instructions to program_multi_periph.hex")
