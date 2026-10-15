module control (
    input  [6:0] opcode,
    input  [2:0] funct3,
    input  [6:0] funct7,
    output       reg_write,
    output       mem_write,
    output       mem_to_reg,
    output       alu_src,
    output       branch,
    output       jump,
    output       jalr,
    output       lui,
    // §100: AUIPC was simply absent - its opcode fell through to the
    // `default` arm, so it decoded as "no control signal set at all",
    // i.e. a silent NOP. Any pc-relative address computation (the whole
    // point of AUIPC) produced nothing and the program carried on with a
    // stale/garbage register. cpu.v uses this to feed `pc` into the ALU's
    // A input instead of rs1, so AUIPC costs one 32-bit mux and reuses
    // the adder that's already there.
    output       auipc,
    output       iret,
    output       irq_en,
    output       irq_dis,
    output [3:0] alu_op,
    // §102: sub-word memory access width for LOAD/STORE - 2'b00=byte,
    // 2'b01=half, 2'b10=word. Only meaningful when mem_to_reg or
    // mem_write is actually set; cpu.v ignores it otherwise.
    output [1:0] mem_size,
    // §102: LOAD only - zero-extend (LBU/LHU) instead of sign-extend
    // (LB/LH). Meaningless for STORE/word loads.
    output       load_unsigned,
    // §100: asserted for any encoding this core does NOT implement.
    // Previously such encodings fell into a `default:` arm that left
    // alu_op at ADD with reg_write still set by the enclosing opcode case
    // - so SLLI/SRLI/SRAI/SLTI/SLTIU/XORI all executed as ADDI, and
    // LB/LH/LBU/LHU/SB/SH as LW/SW. That is the worst failure mode
    // available: not a crash, not a NOP, but a DIFFERENT instruction
    // quietly producing a plausible-looking wrong number. Everything
    // flagged here now has reg_write=0 and mem_write=0, i.e. it retires
    // as an architectural NOP, and cpu.v latches a sticky illegal_seen
    // flag so a testbench can prove the instruction was rejected rather
    // than mis-executed.
    output       illegal
);
    localparam OP_RTYPE  = 7'b0110011;
    localparam OP_ITYPE  = 7'b0010011;
    localparam OP_LOAD   = 7'b0000011;
    localparam OP_STORE  = 7'b0100011;
    localparam OP_BRANCH = 7'b1100011;
    localparam OP_JAL    = 7'b1101111;
    localparam OP_JALR   = 7'b1100111;
    localparam OP_LUI    = 7'b0110111;
    localparam OP_AUIPC  = 7'b0010111;
    // §96: interrupt control instructions, under opcode 0001011 - this is
    // "custom-0" in the RISC-V spec, explicitly reserved for exactly this
    // kind of non-standard extension, not a real/colliding RV32I opcode.
    // No CSRs exist in this core (mstatus/mepc/mtvec would be the standard
    // RISC-V way), so this is a deliberately minimal custom mechanism, not
    // spec-compliant M-mode trap handling.
    localparam OP_CUSTOM0 = 7'b0001011;

    reg rw, mw, m2r, asrc, br, jp, jr, lu, au, ir, ien, idis, ill;
    reg [3:0] aop;
    reg [1:0] msize;
    reg       lu_unsigned;

    assign mem_size     = msize;
    assign load_unsigned = lu_unsigned;
    assign auipc      = au;
    assign illegal    = ill;
    assign reg_write  = rw;
    assign mem_write  = mw;
    assign mem_to_reg = m2r;
    assign alu_src    = asrc;
    assign branch     = br;
    assign jump       = jp;
    assign jalr       = jr;
    assign lui        = lu;
    assign iret       = ir;
    assign irq_en     = ien;
    assign irq_dis    = idis;
    assign alu_op     = aop;

    // Scope after §102: the full RV32I integer-register set except
    // FENCE, ECALL/EBREAK and the CSR ops - i.e. ADD/SUB/SLL/SLT/SLTU/
    // XOR/SRL/SRA/OR/AND (R-type), ADDI/SLLI/SLTI/SLTIU/XORI/SRLI/SRAI/
    // ORI/ANDI (I-type), LB/LH/LW/LBU/LHU, SB/SH/SW (§102 - cpu.v now
    // has the byte-lane extract/merge stage dmem.v needed), BEQ/BNE/
    // BLT/BGE/BLTU/BGEU (cpu.v has a dedicated comparator - see there),
    // LUI, AUIPC, JAL, JALR. §96's IRET/EI/DI (custom-0) stay.
    //
    // DELIBERATELY STILL OUT OF SCOPE, reported as `illegal` instead of
    // silently doing something else:
    //   - MUL/DIV (M extension), FENCE, ECALL/EBREAK, all CSR ops.
    always @(*) begin
        rw = 0; mw = 0; m2r = 0; asrc = 0; br = 0; jp = 0; jr = 0; lu = 0;
        au = 0; ir = 0; ien = 0; idis = 0; ill = 0; aop = 4'b0000;
        msize = 2'b10; lu_unsigned = 0;

        case (opcode)
            OP_RTYPE: begin
                rw = 1; asrc = 0;
                // funct7 must be 0000000, except ADD/SUB and SRL/SRA
                // which also accept 0100000 as the "alternate" variant.
                // Rejecting everything else is what keeps MUL/DIV
                // (funct7=0000001) from decoding as ADD/SLL/etc.
                case (funct3)
                    3'b000:  begin // ADD / SUB
                        aop = funct7[5] ? 4'b0001 : 4'b0000;
                        if (funct7 != 7'b0000000 && funct7 != 7'b0100000) begin rw = 0; ill = 1; end
                    end
                    3'b001:  begin aop = 4'b0110; if (funct7 != 7'b0000000) begin rw = 0; ill = 1; end end // SLL
                    3'b010:  begin aop = 4'b0101; if (funct7 != 7'b0000000) begin rw = 0; ill = 1; end end // SLT
                    3'b011:  begin aop = 4'b1001; if (funct7 != 7'b0000000) begin rw = 0; ill = 1; end end // SLTU (new)
                    3'b100:  begin aop = 4'b0100; if (funct7 != 7'b0000000) begin rw = 0; ill = 1; end end // XOR
                    3'b101:  begin // SRL / SRA - funct7[5] IS the distinction
                        aop = funct7[5] ? 4'b1000 : 4'b0111;
                        if (funct7 != 7'b0000000 && funct7 != 7'b0100000) begin rw = 0; ill = 1; end
                    end
                    3'b110:  begin aop = 4'b0011; if (funct7 != 7'b0000000) begin rw = 0; ill = 1; end end // OR
                    default: begin aop = 4'b0010; if (funct7 != 7'b0000000) begin rw = 0; ill = 1; end end // AND (3'b111)
                endcase
            end
            OP_ITYPE: begin
                rw = 1; asrc = 1;
                // The shift-immediates take their shift amount from
                // imm[4:0] = instr[24:20], which is exactly what the ALU
                // already indexes out of its B input (b[4:0]) - so no
                // separate shamt path is needed, the ordinary
                // sign-extended I-immediate works unchanged, including
                // for SRAI where imm[10] (funct7[5]) is set.
                case (funct3)
                    3'b000:  aop = 4'b0000; // ADDI
                    3'b010:  aop = 4'b0101; // SLTI  (signed)
                    3'b011:  aop = 4'b1001; // SLTIU (unsigned compare of the SIGN-EXTENDED imm, per spec)
                    3'b100:  aop = 4'b0100; // XORI
                    3'b110:  aop = 4'b0011; // ORI
                    3'b111:  aop = 4'b0010; // ANDI
                    3'b001:  begin // SLLI - imm[11:5] must be 0000000
                        aop = 4'b0110;
                        if (funct7 != 7'b0000000) begin rw = 0; ill = 1; end
                    end
                    default: begin // 3'b101: SRLI / SRAI, imm[10] picks which
                        aop = funct7[5] ? 4'b1000 : 4'b0111;
                        if (funct7 != 7'b0000000 && funct7 != 7'b0100000) begin rw = 0; ill = 1; end
                    end
                endcase
            end
            OP_LOAD: begin
                // §102: full width/sign-extension set. funct3 000/001/010
                // = LB/LH/LW (sign-extend), 100/101 = LBU/LHU (zero-extend).
                // 011/110/111 are unassigned in RV32I.
                rw = 1; asrc = 1; m2r = 1; aop = 4'b0000; // address = rs1 + imm
                case (funct3)
                    3'b000: begin msize = 2'b00; lu_unsigned = 0; end // LB
                    3'b001: begin msize = 2'b01; lu_unsigned = 0; end // LH
                    3'b010: begin msize = 2'b10; lu_unsigned = 0; end // LW
                    3'b100: begin msize = 2'b00; lu_unsigned = 1; end // LBU
                    3'b101: begin msize = 2'b01; lu_unsigned = 1; end // LHU
                    default: begin rw = 0; ill = 1; end
                endcase
            end
            OP_STORE: begin
                // §102: funct3 000/001/010 = SB/SH/SW. 011/100-111
                // unassigned in RV32I.
                mw = 1; asrc = 1; aop = 4'b0000;
                case (funct3)
                    3'b000: msize = 2'b00; // SB
                    3'b001: msize = 2'b01; // SH
                    3'b010: msize = 2'b10; // SW
                    default: begin mw = 0; ill = 1; end
                endcase
            end
            OP_BRANCH: begin
                // br=1 is all this stage needs; cpu.v has its own
                // eq/lt/ltu comparators and picks the condition from
                // funct3 directly, independent of the ALU/aop path.
                // funct3 = 010/011 are unassigned in RV32I.
                if (funct3 == 3'b010 || funct3 == 3'b011) begin
                    ill = 1;
                end else begin
                    br = 1; asrc = 0;
                end
            end
            OP_JAL: begin
                rw = 1; jp = 1;
            end
            OP_AUIPC: begin
                // rd <= pc + imm_u. asrc=1 routes imm (cpu.v's imm mux
                // selects imm_u for this opcode) into ALU B, au=1 routes
                // pc into ALU A, aop=ADD - so the result arrives on the
                // ordinary alu_result writeback path with no new adder.
                rw = 1; asrc = 1; au = 1; aop = 4'b0000;
            end
            OP_JALR: begin
                // rd <= pc+4 (same as JAL, cpu.v's wb_data mux already
                // handles that via `jump`); target computed via the ALU
                // path (rs1 + I-type imm) instead of pc+imm, so asrc=1
                // routes alu_b=imm and aop=ADD gives alu_result=rs1+imm -
                // cpu.v's pc_next mux picks this up via the new `jalr` signal.
                if (funct3 == 3'b000) begin
                    rw = 1; asrc = 1; jp = 1; jr = 1; aop = 4'b0000;
                end else begin
                    ill = 1; // only funct3=000 is JALR; the rest is unassigned
                end
            end
            OP_LUI: begin
                // rd <= imm_u (top 20 bits, bottom 12 zero) - no ALU
                // involvement needed, cpu.v routes this straight through.
                rw = 1; lu = 1;
            end
            OP_CUSTOM0: begin
                // funct3 selects which of the 3 interrupt-control ops this
                // is; none of them write a general-purpose register
                // (rw stays 0 - rd is don't-care/ignored in the encoding).
                case (funct3)
                    3'b000: ir   = 1; // IRET: pc<=epc, irq_enable<=1 (cpu.v)
                    3'b001: ien  = 1; // EI:   irq_enable<=1
                    3'b010: idis = 1; // DI:   irq_enable<=0
                    default: ill = 1;
                endcase
            end
            default: begin
                // Unimplemented opcode: all control signals stay at their
                // safe defaults (no register write, no memory write), and
                // it is now REPORTED rather than just quietly ignored.
                ill = 1;
            end
        endcase
    end
endmodule
