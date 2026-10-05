#!/usr/bin/env python3
"""Minimal hand-rolled RV32I encoder for the cpu.v smoke-test program.

Not a general assembler - just enough instruction-format logic (R/I/S/B/J)
to hand-encode a fixed test program, written from the ISA bit layouts
directly rather than calling an existing RISC-V toolchain. Matches the
"built from scratch" scope of this sub-project (see RESEARCH_NOTES.md §42).

v2 adds a tiny two-pass label resolver so branch/jump offsets don't have
to be hand-computed (that was getting error-prone as the test program grew
past the original 9 instructions - see §42d note about BNE/BLT/BGE/SLL/SRL).

Outputs program.hex: one 8-hex-digit word per line, for Verilog $readmemh.
"""

OP_RTYPE  = 0b0110011
OP_ITYPE  = 0b0010011
OP_LOAD   = 0b0000011
OP_STORE  = 0b0100011
OP_BRANCH = 0b1100011
OP_JAL    = 0b1101111


def r_type(funct7, rs2, rs1, funct3, rd, opcode):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def i_type(imm, rs1, funct3, rd, opcode):
    imm &= 0xFFF
    return (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def s_type(imm, rs2, rs1, funct3, opcode):
    imm &= 0xFFF
    imm_hi = (imm >> 5) & 0x7F
    imm_lo = imm & 0x1F
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode


def b_type(imm, rs2, rs1, funct3, opcode):
    imm &= 0x1FFF  # 13-bit signed byte offset, bit0 always 0
    bit12 = (imm >> 12) & 1
    bit11 = (imm >> 11) & 1
    bits10_5 = (imm >> 5) & 0x3F
    bits4_1 = (imm >> 1) & 0xF
    return (bit12 << 31) | (bits10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | \
           (bits4_1 << 8) | (bit11 << 7) | opcode


def j_type(imm, rd, opcode):
    imm &= 0x1FFFFF  # 21-bit signed byte offset, bit0 always 0
    bit20 = (imm >> 20) & 1
    bits10_1 = (imm >> 1) & 0x3FF
    bit11 = (imm >> 11) & 1
    bits19_12 = (imm >> 12) & 0xFF
    return (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) | (bits19_12 << 12) | (rd << 7) | opcode


# --- Two-pass assembler with labels -----------------------------------
# Each program line is either a label (string) or a callable that takes
# (addr, labels_dict) and returns the encoded 32-bit word. Branch/jump
# instructions are lambdas so their offset is computed AFTER all labels
# are known (pass 2), instead of by hand.

class Program:
    def __init__(self):
        self.items = []  # list of (label_name_or_None, fn_or_None)

    def label(self, name):
        self.items.append((name, None))

    def emit(self, fn):
        self.items.append((None, fn))

    def assemble(self):
        labels = {}
        addr = 0
        for name, fn in self.items:
            if fn is None:
                labels[name] = addr
            else:
                addr += 4
        words = []
        addr = 0
        for name, fn in self.items:
            if fn is None:
                continue
            words.append(fn(addr, labels) & 0xFFFFFFFF)
            addr += 4
        return words


p = Program()

# addi x1, x0, 5
p.emit(lambda a, L: i_type(5, 0, 0b000, 1, OP_ITYPE))
# addi x2, x0, 10
p.emit(lambda a, L: i_type(10, 0, 0b000, 2, OP_ITYPE))
# add  x3, x1, x2   -> x3 = 15
p.emit(lambda a, L: r_type(0, 2, 1, 0b000, 3, OP_RTYPE))
# sw   x3, 0(x0)    -> mem[0] = 15
p.emit(lambda a, L: s_type(0, 3, 0, 0b010, OP_STORE))
# lw   x4, 0(x0)    -> x4 = 15
p.emit(lambda a, L: i_type(0, 0, 0b010, 4, OP_LOAD))
# beq  x3, x4, skipA  -> taken (15==15), skips the addi x5,999 below
p.emit(lambda a, L: b_type(L['skipA'] - a, 4, 3, 0b000, OP_BRANCH))
# addi x5, x0, 999  -> must be SKIPPED
p.emit(lambda a, L: i_type(999, 0, 0b000, 5, OP_ITYPE))
p.label('skipA')
# addi x5, x0, 42   -> branch target, x5 = 42
p.emit(lambda a, L: i_type(42, 0, 0b000, 5, OP_ITYPE))

# --- v2 additions: BNE / BLT / BGE / SLL / SRL ---
# addi x6, x0, 3
p.emit(lambda a, L: i_type(3, 0, 0b000, 6, OP_ITYPE))
# addi x7, x0, 2    -> shift amount
p.emit(lambda a, L: i_type(2, 0, 0b000, 7, OP_ITYPE))
# sll  x9, x6, x7   -> x9 = 3 << 2 = 12
p.emit(lambda a, L: r_type(0, 7, 6, 0b001, 9, OP_RTYPE))
# srl  x10, x9, x7  -> x10 = 12 >> 2 = 3
p.emit(lambda a, L: r_type(0, 7, 9, 0b101, 10, OP_RTYPE))

# bne x3, x4, failBNE  -> x3==x4 (both 15), NOT taken, falls through
p.emit(lambda a, L: b_type(L['failBNE'] - a, 4, 3, 0b001, OP_BRANCH))
# addi x11, x0, 111  -> executes because BNE correctly not-taken
p.emit(lambda a, L: i_type(111, 0, 0b000, 11, OP_ITYPE))
# jal x0, afterBNE -> skip the fail-marker line
p.emit(lambda a, L: j_type(L['afterBNE'] - a, 0, OP_JAL))
p.label('failBNE')
p.emit(lambda a, L: i_type(666, 0, 0b000, 11, OP_ITYPE))  # should NEVER execute
p.label('afterBNE')

# blt x1, x2, takeB  -> 5 < 10, taken
p.emit(lambda a, L: b_type(L['takeB'] - a, 2, 1, 0b100, OP_BRANCH))
# addi x12, x0, 222  -> must be SKIPPED (blt taken)
p.emit(lambda a, L: i_type(222, 0, 0b000, 12, OP_ITYPE))
p.label('takeB')
# addi x12, x0, 333  -> branch target
p.emit(lambda a, L: i_type(333, 0, 0b000, 12, OP_ITYPE))

# bge x2, x1, takeC  -> 10 >= 5, taken
p.emit(lambda a, L: b_type(L['takeC'] - a, 1, 2, 0b101, OP_BRANCH))
# addi x13, x0, 444  -> must be SKIPPED (bge taken)
p.emit(lambda a, L: i_type(444, 0, 0b000, 13, OP_ITYPE))
p.label('takeC')
# addi x13, x0, 555  -> branch target
p.emit(lambda a, L: i_type(555, 0, 0b000, 13, OP_ITYPE))

p.label('halt')
# jal x0, halt  -> infinite self-loop
p.emit(lambda a, L: j_type(L['halt'] - a, 0, OP_JAL))


if __name__ == "__main__":
    words = p.assemble()
    with open("program.hex", "w") as f:
        for w in words:
            f.write(f"{w:08x}\n")
    print(f"wrote {len(words)} instructions to program.hex")
