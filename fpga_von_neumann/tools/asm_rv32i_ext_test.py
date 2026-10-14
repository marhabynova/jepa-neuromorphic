#!/usr/bin/env python3
"""Directed test program for the instructions added in §100:
SLLI/SRLI/SRAI, SLTI/SLTIU, XORI, R-type SRA and SLTU, BLTU/BGEU, AUIPC,
plus the "unimplemented encodings must NOT silently execute as something
else" behaviour (LB/SB).

Written from the RISC-V unprivileged ISA spec's definitions of each
instruction, NOT from reading control.v/alu.v:
  - SLLI/SRLI/SRAI take shamt = imm[4:0]; SRLI zero-fills, SRAI
    sign-fills (imm[10] is the only bit that distinguishes them).
  - SLTI compares as signed; SLTIU compares as unsigned AFTER
    sign-extending the 12-bit immediate (so `sltiu rd, x0, -1` is
    0 < 0xFFFFFFFF = 1, not 0 < -1 = 0).
  - SRA is sign-replicating; SRL is zero-filling. They differ on every
    negative operand, which is why decoding SRA as SRL is silently wrong.
  - BLTU/BGEU compare as unsigned: 1 < 0xFFFFFFFF unsigned, but
    1 > 0xFFFFFFFF signed - the two give OPPOSITE answers here.
  - AUIPC: rd = pc + (imm20 << 12), using the address of the AUIPC
    instruction itself.

Every expected value is chosen so the OLD behaviour produces a DIFFERENT
number, not an accidentally-equal one:
  old control.v decoded all the new I-type ops as ADDI, R-type SRA as
  SRL, R-type SLTU as ADD, BLTU/BGEU as never-taken, AUIPC as a NOP, and
  LB/SB as LW/SW.

Edge values covered: shift by 0 and by 31, negative immediates,
signed-vs-unsigned comparison at INT_MIN / -1 / 0 / 1, AUIPC at a
non-zero pc with a zero, positive and negative (bit-19 set) immediate.
"""
import sys
sys.path.insert(0, ".")
from asm import (Program, OP_ITYPE, OP_RTYPE, OP_LOAD, OP_STORE, OP_BRANCH,
                 OP_JAL, i_type, r_type, s_type, b_type, j_type)

OP_LUI = 0b0110111
OP_AUIPC = 0b0010111

F7_ZERO = 0b0000000
F7_ALT = 0b0100000   # the "alternate" funct7: SUB, SRA, SRAI


def u_type(imm20, rd, opcode):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


def shift_imm(shamt, alt, rs1, funct3, rd):
    """SLLI/SRLI/SRAI: imm[11:5] = funct7, imm[4:0] = shamt."""
    return i_type(((F7_ALT if alt else F7_ZERO) << 5) | (shamt & 0x1F),
                  rs1, funct3, rd, OP_ITYPE)


p = Program()

# ---- constants -------------------------------------------------------
p.emit(lambda a, L: i_type(1, 0, 0b000, 1, OP_ITYPE))        # x1  = 1
p.emit(lambda a, L: i_type(-1, 0, 0b000, 2, OP_ITYPE))       # x2  = 0xFFFFFFFF
p.emit(lambda a, L: u_type(0x80000, 3, OP_LUI))              # x3  = 0x80000000 (INT_MIN)
p.emit(lambda a, L: i_type(0, 0, 0b000, 4, OP_ITYPE))        # x4  = 0
p.emit(lambda a, L: i_type(31, 0, 0b000, 23, OP_ITYPE))      # x23 = 31 (shift amount)

# ---- shift immediates (shamt 0 and 31, sign-fill vs zero-fill) -------
p.emit(lambda a, L: shift_imm(31, 0, 1, 0b001, 5))           # x5  = slli x1,31  = 0x80000000
p.emit(lambda a, L: shift_imm(0, 0, 1, 0b001, 6))            # x6  = slli x1,0   = 1
p.emit(lambda a, L: shift_imm(31, 0, 3, 0b101, 7))           # x7  = srli x3,31  = 1
p.emit(lambda a, L: shift_imm(31, 1, 3, 0b101, 8))           # x8  = srai x3,31  = 0xFFFFFFFF
p.emit(lambda a, L: shift_imm(0, 1, 3, 0b101, 9))            # x9  = srai x3,0   = 0x80000000
p.emit(lambda a, L: shift_imm(4, 0, 2, 0b101, 10))           # x10 = srli x2,4   = 0x0FFFFFFF
p.emit(lambda a, L: shift_imm(4, 1, 2, 0b101, 11))           # x11 = srai x2,4   = 0xFFFFFFFF
p.emit(lambda a, L: shift_imm(31, 0, 2, 0b001, 12))          # x12 = slli x2,31  = 0x80000000

# ---- SLTI / SLTIU at INT_MIN / -1 / 0 / 1 ----------------------------
p.emit(lambda a, L: i_type(1, 3, 0b010, 13, OP_ITYPE))       # x13 = slti  x3,1  = 1 (INT_MIN < 1 signed)
p.emit(lambda a, L: i_type(1, 3, 0b011, 14, OP_ITYPE))       # x14 = sltiu x3,1  = 0 (0x80000000 > 1 unsigned)
p.emit(lambda a, L: i_type(0, 2, 0b010, 15, OP_ITYPE))       # x15 = slti  x2,0  = 1 (-1 < 0)
p.emit(lambda a, L: i_type(0, 2, 0b011, 16, OP_ITYPE))       # x16 = sltiu x2,0  = 0 (0xFFFFFFFF > 0)
p.emit(lambda a, L: i_type(-1, 4, 0b011, 17, OP_ITYPE))      # x17 = sltiu x4,-1 = 1 (0 < 0xFFFFFFFF)
p.emit(lambda a, L: i_type(-1, 4, 0b010, 18, OP_ITYPE))      # x18 = slti  x4,-1 = 0 (0 > -1)
p.emit(lambda a, L: i_type(1, 1, 0b010, 19, OP_ITYPE))       # x19 = slti  x1,1  = 0 (1 == 1, not <)

# ---- XORI (including the all-ones immediate = bitwise NOT) -----------
p.emit(lambda a, L: i_type(-1, 2, 0b100, 20, OP_ITYPE))      # x20 = xori x2,-1  = 0
p.emit(lambda a, L: i_type(3, 1, 0b100, 21, OP_ITYPE))       # x21 = xori x1,3   = 2

# ---- R-type SRA vs SRL (funct7[5] is the ONLY difference) + SLTU -----
p.emit(lambda a, L: r_type(F7_ALT, 23, 3, 0b101, 22, OP_RTYPE))   # x22 = sra x3,x23 = 0xFFFFFFFF
p.emit(lambda a, L: r_type(F7_ZERO, 23, 3, 0b101, 25, OP_RTYPE))  # x25 = srl x3,x23 = 1
p.emit(lambda a, L: r_type(F7_ZERO, 2, 1, 0b011, 24, OP_RTYPE))   # x24 = sltu x1,x2 = 1 (1 < 0xFFFFFFFF)

# ---- BLTU / BGEU: 6 sub-tests accumulated into a bitmask in x26 ------
# Expected total = 1+2+4+8+16+32 = 63. Old code (branch never taken for
# funct3 110/111) scores 2+8 = 10.
p.emit(lambda a, L: i_type(0, 0, 0b000, 26, OP_ITYPE))       # x26 = 0

# (1) bltu x1,x2 MUST be taken (1 < 0xFFFFFFFF unsigned)
p.emit(lambda a, L: b_type(L['bt1'] - a, 2, 1, 0b110, OP_BRANCH))
p.emit(lambda a, L: j_type(L['ba1'] - a, 0, OP_JAL))
p.label('bt1')
p.emit(lambda a, L: i_type(1, 26, 0b000, 26, OP_ITYPE))
p.label('ba1')

# (2) bltu x2,x1 must NOT be taken (0xFFFFFFFF > 1 unsigned)
p.emit(lambda a, L: b_type(L['bt2'] - a, 1, 2, 0b110, OP_BRANCH))
p.emit(lambda a, L: i_type(2, 26, 0b000, 26, OP_ITYPE))
p.label('bt2')

# (3) bgeu x2,x1 MUST be taken
p.emit(lambda a, L: b_type(L['bt3'] - a, 1, 2, 0b111, OP_BRANCH))
p.emit(lambda a, L: j_type(L['ba3'] - a, 0, OP_JAL))
p.label('bt3')
p.emit(lambda a, L: i_type(4, 26, 0b000, 26, OP_ITYPE))
p.label('ba3')

# (4) bgeu x1,x2 must NOT be taken
p.emit(lambda a, L: b_type(L['bt4'] - a, 2, 1, 0b111, OP_BRANCH))
p.emit(lambda a, L: i_type(8, 26, 0b000, 26, OP_ITYPE))
p.label('bt4')

# (5) bgeu x1,x1 MUST be taken (equal operands - the ">=" half of BGEU)
p.emit(lambda a, L: b_type(L['bt5'] - a, 1, 1, 0b111, OP_BRANCH))
p.emit(lambda a, L: j_type(L['ba5'] - a, 0, OP_JAL))
p.label('bt5')
p.emit(lambda a, L: i_type(16, 26, 0b000, 26, OP_ITYPE))
p.label('ba5')

# (6) bltu x4,x1 MUST be taken (0 < 1, the trivial direction)
p.emit(lambda a, L: b_type(L['bt6'] - a, 1, 4, 0b110, OP_BRANCH))
p.emit(lambda a, L: j_type(L['ba6'] - a, 0, OP_JAL))
p.label('bt6')
p.emit(lambda a, L: i_type(32, 26, 0b000, 26, OP_ITYPE))
p.label('ba6')

# ---- AUIPC ------------------------------------------------------------
# Self-checking against JAL's link address, so the test never has to know
# an absolute program address. Pre-seeded with a wrong value so that a
# NOP-ing AUIPC (the old behaviour) cannot accidentally leave a 0 behind.
p.emit(lambda a, L: i_type(123, 0, 0b000, 30, OP_ITYPE))     # x30 = 123 (poison)
p.emit(lambda a, L: u_type(0, 30, OP_AUIPC))                 # A:   x30 = A
p.emit(lambda a, L: j_type(L['al'] - a, 31, OP_JAL))         # A+4: x31 = A+8
p.label('al')
p.emit(lambda a, L: i_type(-8, 31, 0b000, 31, OP_ITYPE))     # x31 = A
p.emit(lambda a, L: r_type(F7_ALT, 31, 30, 0b000, 30, OP_RTYPE))  # x30 = A - A = 0

# Non-zero immediate: two AUIPCs 4 bytes apart, difference = 0x10000 - 4.
p.emit(lambda a, L: i_type(123, 0, 0b000, 29, OP_ITYPE))     # x29 = 123 (poison)
p.emit(lambda a, L: u_type(0x10, 29, OP_AUIPC))              # C:   x29 = C + 0x10000
p.emit(lambda a, L: u_type(0, 28, OP_AUIPC))                 # C+4: x28 = C + 4
p.emit(lambda a, L: r_type(F7_ALT, 28, 29, 0b000, 29, OP_RTYPE))  # x29 = 0x10000 - 4 = 0xFFFC

# Negative immediate (imm20 bit 19 set -> imm_u = 0xFFFFF000 = -4096).
p.emit(lambda a, L: i_type(123, 0, 0b000, 27, OP_ITYPE))     # x27 = 123 (poison)
p.emit(lambda a, L: u_type(0xFFFFF, 27, OP_AUIPC))           # E:   x27 = E - 4096
p.emit(lambda a, L: u_type(0, 28, OP_AUIPC))                 # E+4: x28 = E + 4
p.emit(lambda a, L: r_type(F7_ALT, 27, 28, 0b000, 27, OP_RTYPE))  # x27 = (E+4)-(E-4096) = 4100

# ---- unimplemented encodings must not execute as something else ------
# mem[0] = 0 and mem[16] = 0xFFFFFFFF are set up first so the OLD
# behaviour produces a specific, deterministic WRONG value rather than X.
p.emit(lambda a, L: s_type(0, 4, 0, 0b010, OP_STORE))        # sw x4, 0(x0)    -> mem[0]  = 0
p.emit(lambda a, L: s_type(64, 2, 0, 0b010, OP_STORE))       # sw x2, 64(x0)   -> mem[16] = 0xFFFFFFFF

# LB (funct3=000) - old decode ignored funct3 and ran it as LW, which
# would overwrite x5 (0x80000000) with mem[0] (0). It must be a no-op.
p.emit(lambda a, L: i_type(0, 0, 0b000, 5, OP_LOAD))         # lb x5, 0(x0)
# SB (funct3=000) - old decode ran it as SW, clobbering all 4 bytes of
# mem[16] with x1 (=1). It must be a no-op.
p.emit(lambda a, L: s_type(64, 1, 0, 0b000, OP_STORE))       # sb x1, 64(x0)
# Read mem[16] back into x28 - must still be 0xFFFFFFFF.
p.emit(lambda a, L: i_type(64, 0, 0b010, 28, OP_LOAD))       # lw x28, 64(x0)

p.label('halt')
p.emit(lambda a, L: j_type(L['halt'] - a, 0, OP_JAL))

if __name__ == "__main__":
    words = p.assemble()
    with open("program_rv32i_ext.hex", "w") as f:
        for w in words:
            f.write(f"{w:08x}\n")
    print(f"wrote {len(words)} instructions to program_rv32i_ext.hex")
