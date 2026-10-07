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
    output       iret,
    output       irq_en,
    output       irq_dis,
    output [3:0] alu_op
);
    localparam OP_RTYPE  = 7'b0110011;
    localparam OP_ITYPE  = 7'b0010011;
    localparam OP_LOAD   = 7'b0000011;
    localparam OP_STORE  = 7'b0100011;
    localparam OP_BRANCH = 7'b1100011;
    localparam OP_JAL    = 7'b1101111;
    localparam OP_JALR   = 7'b1100111;
    localparam OP_LUI    = 7'b0110111;
    // §96: interrupt control instructions, under opcode 0001011 - this is
    // "custom-0" in the RISC-V spec, explicitly reserved for exactly this
    // kind of non-standard extension, not a real/colliding RV32I opcode.
    // No CSRs exist in this core (mstatus/mepc/mtvec would be the standard
    // RISC-V way), so this is a deliberately minimal custom mechanism, not
    // spec-compliant M-mode trap handling.
    localparam OP_CUSTOM0 = 7'b0001011;

    reg rw, mw, m2r, asrc, br, jp, jr, lu, ir, ien, idis;
    reg [3:0] aop;

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

    // v2 scope: ADD/SUB/AND/OR/XOR/SLT/SLL/SRL (R-type), ADDI/ANDI/ORI (I-type),
    // LW, SW, BEQ/BNE/BLT/BGE (cpu.v has a dedicated comparator - see there),
    // JAL. v3 adds JALR (real function-call/return convention - without it,
    // no subroutine could return to a dynamic caller, only to a fixed
    // offset) and LUI (build a 32-bit constant in 2 instructions instead of
    // the addi+sll shift trick asm.py used everywhere before this). §96
    // adds IRET/EI/DI for a minimal interrupt mechanism (see cpu.v).
    // SRA, BLTU/BGEU, MUL/DIV, AUIPC, and the rest of RV32I are still TODO.
    always @(*) begin
        rw = 0; mw = 0; m2r = 0; asrc = 0; br = 0; jp = 0; jr = 0; lu = 0;
        ir = 0; ien = 0; idis = 0; aop = 4'b0000;

        case (opcode)
            OP_RTYPE: begin
                rw = 1; asrc = 0;
                case (funct3)
                    3'b000:  aop = funct7[5] ? 4'b0001 : 4'b0000; // SUB : ADD
                    3'b111:  aop = 4'b0010; // AND
                    3'b110:  aop = 4'b0011; // OR
                    3'b100:  aop = 4'b0100; // XOR
                    3'b010:  aop = 4'b0101; // SLT
                    3'b001:  aop = 4'b0110; // SLL
                    3'b101:  aop = 4'b0111; // SRL (SRA not distinguished yet)
                    default: aop = 4'b0000;
                endcase
            end
            OP_ITYPE: begin
                rw = 1; asrc = 1;
                case (funct3)
                    3'b000:  aop = 4'b0000; // ADDI
                    3'b111:  aop = 4'b0010; // ANDI
                    3'b110:  aop = 4'b0011; // ORI
                    default: aop = 4'b0000;
                endcase
            end
            OP_LOAD: begin
                rw = 1; asrc = 1; m2r = 1; aop = 4'b0000; // address = rs1 + imm
            end
            OP_STORE: begin
                mw = 1; asrc = 1; aop = 4'b0000;
            end
            OP_BRANCH: begin
                // br=1 is all this stage needs; cpu.v has its own eq/lt
                // comparator and picks the condition from funct3 directly
                // (BEQ/BNE/BLT/BGE), independent of the ALU/aop path.
                br = 1; asrc = 0;
            end
            OP_JAL: begin
                rw = 1; jp = 1;
            end
            OP_JALR: begin
                // rd <= pc+4 (same as JAL, cpu.v's wb_data mux already
                // handles that via `jump`); target computed via the ALU
                // path (rs1 + I-type imm) instead of pc+imm, so asrc=1
                // routes alu_b=imm and aop=ADD gives alu_result=rs1+imm -
                // cpu.v's pc_next mux picks this up via the new `jalr` signal.
                rw = 1; asrc = 1; jp = 1; jr = 1; aop = 4'b0000;
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
                    default: ;
                endcase
            end
            default: begin
                // unimplemented opcode: all control signals stay at safe defaults (no writes)
            end
        endcase
    end
endmodule
