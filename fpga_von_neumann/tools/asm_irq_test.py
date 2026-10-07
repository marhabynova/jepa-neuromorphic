#!/usr/bin/env python3
"""§96: real interrupt-driven demo, not a bus-level unit test. main()
configures spi_master, enables interrupts, then runs a busy-wait loop that
NEVER touches SPI's STATUS register - no polling at all. When the
transfer finishes, spi_master pulses irq, the CPU traps to IRQ_VECTOR
(address 4), the handler reads RXDATA and writes it to the debug port,
sets a flag (x30), then IRET returns to exactly where the main loop left
off. The loop then finishes normally and sets its own marker (x31) -
proving both that the interrupt fired AND that execution correctly
resumed in the interrupted code afterward, not just jumped there once.
"""
import sys
sys.path.insert(0, ".")
from asm import Program, OP_ITYPE, OP_RTYPE, OP_STORE, OP_LOAD, OP_BRANCH, OP_JAL, i_type, r_type, s_type, b_type, j_type

OP_CUSTOM0 = 0b0001011


def custom0(funct3):
    return i_type(0, 0, funct3, 0, OP_CUSTOM0)


IRET = custom0(0b000)
EI = custom0(0b001)
DI = custom0(0b010)

p = Program()

# addr 0: unconditional jump over the reserved IRQ_VECTOR slot at addr 4.
p.emit(lambda a, L: j_type(L['main'] - a, 0, OP_JAL))

# addr 4 (IRQ_VECTOR): handler. Assumes x16 already holds the MMIO base
# (set once in main before interrupts are ever enabled, never touched
# again by main after that) - a real multi-handler system would need a
# proper register-save convention; this toy demo doesn't need one since
# main and the handler use entirely disjoint registers.
p.emit(lambda a, L: i_type(0x20, 16, 0b010, 6, OP_LOAD))   # lw  x6, 0x20(x16)  RXDATA
p.emit(lambda a, L: s_type(0, 6, 16, 0b010, OP_STORE))     # sw  x6, 0(x16)     debug port <= RXDATA
p.emit(lambda a, L: i_type(1, 0, 0b000, 30, OP_ITYPE))     # addi x30, x0, 1    handler-ran flag
p.emit(lambda a, L: IRET)                                   # iret

p.label('main')
p.emit(lambda a, L: i_type(255, 0, 0b000, 14, OP_ITYPE))   # addi x14, x0, 255
p.emit(lambda a, L: i_type(24, 0, 0b000, 15, OP_ITYPE))    # addi x15, x0, 24
p.emit(lambda a, L: r_type(0, 15, 14, 0b001, 16, OP_RTYPE))  # sll x16, x14, x15  (MMIO base)

p.emit(lambda a, L: i_type(2, 0, 0b000, 1, OP_ITYPE))      # addi x1, x0, 2     CLKDIV
p.emit(lambda a, L: s_type(0x14, 1, 16, 0b010, OP_STORE))
p.emit(lambda a, L: i_type(0, 0, 0b000, 2, OP_ITYPE))      # addi x2, x0, 0     CTRL mode0 start0
p.emit(lambda a, L: s_type(0x10, 2, 16, 0b010, OP_STORE))
p.emit(lambda a, L: i_type(165, 0, 0b000, 3, OP_ITYPE))    # addi x3, x0, 165=0xA5
p.emit(lambda a, L: s_type(0x18, 3, 16, 0b010, OP_STORE))  # TXDATA
p.emit(lambda a, L: i_type(1, 0, 0b000, 4, OP_ITYPE))      # addi x4, x0, 1     START=1
p.emit(lambda a, L: s_type(0x10, 4, 16, 0b010, OP_STORE))

p.emit(lambda a, L: EI)                                      # enable interrupts - NO polling from here on

p.emit(lambda a, L: i_type(50, 0, 0b000, 7, OP_ITYPE))     # addi x7, x0, 50    busy-loop counter
p.label('busy_loop')
p.emit(lambda a, L: i_type(-1, 7, 0b000, 7, OP_ITYPE))     # addi x7, x7, -1
p.emit(lambda a, L: b_type(L['busy_loop'] - a, 0, 7, 0b001, OP_BRANCH))  # bne x7, x0, busy_loop

p.emit(lambda a, L: i_type(777, 0, 0b000, 31, OP_ITYPE))   # addi x31, x0, 777  loop finished normally

p.label('halt')
p.emit(lambda a, L: j_type(L['halt'] - a, 0, OP_JAL))

if __name__ == "__main__":
    words = p.assemble()
    with open("program_irq.hex", "w") as f:
        for w in words:
            f.write(f"{w:08x}\n")
    print(f"wrote {len(words)} instructions to program_irq.hex")
