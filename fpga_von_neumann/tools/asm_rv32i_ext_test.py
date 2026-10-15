#!/usr/bin/env python3
"""Directed test program for the instructions added in §100:
SLLI/SRLI/SRAI, SLTI/SLTIU, XORI, R-type SRA and SLTU, BLTU/BGEU, AUIPC,
plus the §102 byte/halfword load-store set (LB/LH/LBU/LHU/SB/SH) and a
check that whatever's STILL unimplemented (MUL, now that LB is real)
doesn't silently execute as something else.

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
  SRL, R-type SLTU as ADD, BLTU/BGEU as never-taken, AUIPC as a NOP.
  LB/SB used to be the "stays illegal" anchor instruction (§100); §102
  makes them real, so MUL (RV32M, still out of scope) takes over that
  role here.

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
# §102: x5 is reused as a general scratch register for the byte/half
# load-store section near the end of this program (every architectural
# register is otherwise already spoken for by a later check) - park its
# SLLI result in memory NOW, at word index 32 (byte 128), before it gets
# overwritten. tb_rv32i_ext.v checks dmem there instead of regs[5].
p.emit(lambda a, L: s_type(128, 5, 0, 0b010, OP_STORE))      # sw x5,128(x0)
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

# ---- §102: byte/half load-store, now genuinely implemented -----------
# x5 is pure scratch from here on (its only earlier role, the SLLI
# check, was already relocated to memory above). Every value needed
# (byte/half constants, the preloaded test word) is built fresh into x5
# immediately before use, so this whole section needs no register
# beyond the one every other section has already finished with.
#
# WORD = 0x9A0281E1: byte0=0xE1, byte1=0x81, byte2=0x02, byte3=0x9A
# (bytes 0,1,3 all have bit7 set, so sign- vs zero-extend genuinely
# differ for them; byte2 doesn't, which checks plain alignment instead).
# half0=0x81E1 (bit15 set), half1=0x9A02 (bit15 set) - same reasoning
# one level up.

# -- SB byte-lane test: baseline 0x11223344 at byte 64, then overwrite
# each byte individually. If any SB wrote outside its own lane, the
# reconstructed word below would not equal WORD exactly.
p.emit(lambda a, L: u_type(0x11223, 5, OP_LUI))
p.emit(lambda a, L: i_type(836, 5, 0b000, 5, OP_ITYPE))      # x5 = 0x11223344
p.emit(lambda a, L: s_type(64, 5, 0, 0b010, OP_STORE))       # sw x5,64(x0)  baseline
p.emit(lambda a, L: i_type(-31, 0, 0b000, 5, OP_ITYPE))      # x5 = 0xFFFFFFE1 (0xE1)
p.emit(lambda a, L: s_type(64, 5, 0, 0b000, OP_STORE))       # sb x5,64(x0)  byte0
p.emit(lambda a, L: i_type(-127, 0, 0b000, 5, OP_ITYPE))     # x5 = 0xFFFFFF81 (0x81)
p.emit(lambda a, L: s_type(65, 5, 0, 0b000, OP_STORE))       # sb x5,65(x0)  byte1
p.emit(lambda a, L: i_type(2, 0, 0b000, 5, OP_ITYPE))        # x5 = 0x00000002
p.emit(lambda a, L: s_type(66, 5, 0, 0b000, OP_STORE))       # sb x5,66(x0)  byte2
p.emit(lambda a, L: i_type(-102, 0, 0b000, 5, OP_ITYPE))     # x5 = 0xFFFFFF9A (0x9A)
p.emit(lambda a, L: s_type(67, 5, 0, 0b000, OP_STORE))       # sb x5,67(x0)  byte3
# expect dmem word index 16 (byte 64) == WORD = 0x9A0281E1

# -- SH byte-lane test: same idea, baseline 0x11223344 at byte 68, then
# overwrite the lower half then the upper half.
p.emit(lambda a, L: u_type(0x11223, 5, OP_LUI))
p.emit(lambda a, L: i_type(836, 5, 0b000, 5, OP_ITYPE))      # x5 = 0x11223344
p.emit(lambda a, L: s_type(68, 5, 0, 0b010, OP_STORE))       # sw x5,68(x0)  baseline
p.emit(lambda a, L: u_type(0x8, 5, OP_LUI))
p.emit(lambda a, L: i_type(481, 5, 0b000, 5, OP_ITYPE))      # x5 = 0x000081E1 (half0)
p.emit(lambda a, L: s_type(68, 5, 0, 0b001, OP_STORE))       # sh x5,68(x0)  lower half
p.emit(lambda a, L: u_type(0xA, 5, OP_LUI))
p.emit(lambda a, L: i_type(-1534, 5, 0b000, 5, OP_ITYPE))    # x5 = 0x00009A02 (half1)
p.emit(lambda a, L: s_type(70, 5, 0, 0b001, OP_STORE))       # sh x5,70(x0)  upper half
# expect dmem word index 17 (byte 68) == WORD = 0x9A0281E1

# -- preload WORD at byte 80 (word index 20), independent of the SB/SH
# writes above, for the LB/LH/LBU/LHU correctness checks below.
p.emit(lambda a, L: u_type(0x9A028, 5, OP_LUI))
p.emit(lambda a, L: i_type(481, 5, 0b000, 5, OP_ITYPE))      # x5 = 0x9A0281E1
p.emit(lambda a, L: s_type(80, 5, 0, 0b010, OP_STORE))       # sw x5,80(x0)

# -- LB/LBU/LH/LHU: load, then park the result in its own memory word
# (word indices 40-50, bytes 160-200) so the testbench can check it
# directly, same style as the SB/SH checks above - every architectural
# register is spoken for, this needs none.
p.emit(lambda a, L: i_type(80, 0, 0b000, 5, OP_LOAD))
p.emit(lambda a, L: s_type(160, 5, 0, 0b010, OP_STORE))      # LB  @+0 -> 0xFFFFFFE1
p.emit(lambda a, L: i_type(80, 0, 0b100, 5, OP_LOAD))
p.emit(lambda a, L: s_type(164, 5, 0, 0b010, OP_STORE))      # LBU @+0 -> 0x000000E1
p.emit(lambda a, L: i_type(81, 0, 0b000, 5, OP_LOAD))
p.emit(lambda a, L: s_type(168, 5, 0, 0b010, OP_STORE))      # LB  @+1 -> 0xFFFFFF81
p.emit(lambda a, L: i_type(81, 0, 0b100, 5, OP_LOAD))
p.emit(lambda a, L: s_type(172, 5, 0, 0b010, OP_STORE))      # LBU @+1 -> 0x00000081
p.emit(lambda a, L: i_type(82, 0, 0b000, 5, OP_LOAD))
p.emit(lambda a, L: s_type(176, 5, 0, 0b010, OP_STORE))      # LB  @+2 -> 0x00000002
p.emit(lambda a, L: i_type(83, 0, 0b000, 5, OP_LOAD))
p.emit(lambda a, L: s_type(180, 5, 0, 0b010, OP_STORE))      # LB  @+3 -> 0xFFFFFF9A
p.emit(lambda a, L: i_type(83, 0, 0b100, 5, OP_LOAD))
p.emit(lambda a, L: s_type(184, 5, 0, 0b010, OP_STORE))      # LBU @+3 -> 0x0000009A
p.emit(lambda a, L: i_type(80, 0, 0b001, 5, OP_LOAD))
p.emit(lambda a, L: s_type(188, 5, 0, 0b010, OP_STORE))      # LH  @+0 -> 0xFFFF81E1
p.emit(lambda a, L: i_type(80, 0, 0b101, 5, OP_LOAD))
p.emit(lambda a, L: s_type(192, 5, 0, 0b010, OP_STORE))      # LHU @+0 -> 0x000081E1
p.emit(lambda a, L: i_type(82, 0, 0b001, 5, OP_LOAD))
p.emit(lambda a, L: s_type(196, 5, 0, 0b010, OP_STORE))      # LH  @+2 -> 0xFFFF9A02
p.emit(lambda a, L: i_type(82, 0, 0b101, 5, OP_LOAD))
p.emit(lambda a, L: s_type(200, 5, 0, 0b010, OP_STORE))      # LHU @+2 -> 0x00009A02

# -- MUL (RV32M, funct7=0000001) is the new anchor for "an unimplemented
# encoding must not execute as something else" - LB held that role
# before §102 made it a real instruction. rd=x5 is irrelevant: control.v
# forces reg_write=0 for anything illegal, so this is guaranteed to
# never actually write x5, whatever x5 holds at this point.
p.emit(lambda a, L: r_type(0b0000001, 2, 1, 0b000, 5, OP_RTYPE))  # mul x5,x1,x2 (illegal)

p.label('halt')
p.emit(lambda a, L: j_type(L['halt'] - a, 0, OP_JAL))

if __name__ == "__main__":
    words = p.assemble()
    with open("program_rv32i_ext.hex", "w") as f:
        for w in words:
            f.write(f"{w:08x}\n")
    print(f"wrote {len(words)} instructions to program_rv32i_ext.hex")
