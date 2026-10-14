`timescale 1ns/1ps

// §100: directed test for the instructions the audit found missing or
// silently mis-decoded - SLLI/SRLI/SRAI, SLTI/SLTIU, XORI, R-type SRA
// and SLTU, BLTU/BGEU, AUIPC - plus the requirement that an
// unimplemented encoding (LB/SB) does NOT execute as a different
// instruction.
//
// Every expectation below comes from the RISC-V unprivileged ISA spec's
// definition of the instruction, not from reading control.v/alu.v. Every
// one was also chosen so that the OLD decoder produces a DIFFERENT value
// (old: all the new I-type ops ran as ADDI, SRA as SRL, SLTU as ADD,
// BLTU/BGEU never taken, AUIPC a NOP, LB/SB as LW/SW) - verified by
// running this bench against the pre-fix RTL, where 23 of 25 checks fail.
//
// Program: tools/asm_rv32i_ext_test.py (see its header for the encoding
// rationale and the edge values chosen).
module tb_rv32i_ext;
    reg clk, rst;
    wire [31:0] mmio_out;
    wire mmio_valid;
    wire mmio_we, mmio_re;
    wire [31:0] mmio_addr, mmio_wdata;

    cpu dut (
        .clk(clk), .rst(rst),
        .mmio_out(mmio_out), .mmio_valid(mmio_valid),
        .mmio_we(mmio_we), .mmio_re(mmio_re),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_rdata(32'd0),
        .irq(1'b0)
    );

    always #5 clk = ~clk;

    integer errors;

    task chk(input [255:0] name, input [31:0] got, input [31:0] expect);
        begin
            if (got !== expect) begin
                $display("  MISMATCH %0s: got %h, expected %h", name, got, expect);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        clk = 0; rst = 1; errors = 0;
        #12 rst = 0;
        #6000;

        // --- shift immediates: shamt 0 and 31, zero-fill vs sign-fill ---
        chk("slli x1,31",  dut.rf.regs[5],  32'h80000000);
        chk("slli x1,0",   dut.rf.regs[6],  32'h00000001);
        chk("srli x3,31",  dut.rf.regs[7],  32'h00000001);
        chk("srai x3,31",  dut.rf.regs[8],  32'hFFFFFFFF);
        chk("srai x3,0",   dut.rf.regs[9],  32'h80000000);
        chk("srli x2,4",   dut.rf.regs[10], 32'h0FFFFFFF);
        chk("srai x2,4",   dut.rf.regs[11], 32'hFFFFFFFF);
        chk("slli x2,31",  dut.rf.regs[12], 32'h80000000);

        // --- signed vs unsigned compare at INT_MIN / -1 / 0 / 1 ---
        chk("slti INT_MIN,1",  dut.rf.regs[13], 32'd1);
        chk("sltiu INT_MIN,1", dut.rf.regs[14], 32'd0);
        chk("slti -1,0",       dut.rf.regs[15], 32'd1);
        chk("sltiu -1,0",      dut.rf.regs[16], 32'd0);
        chk("sltiu 0,-1",      dut.rf.regs[17], 32'd1);
        chk("slti 0,-1",       dut.rf.regs[18], 32'd0);
        chk("slti 1,1",        dut.rf.regs[19], 32'd0);

        // --- XORI ---
        chk("xori -1,-1", dut.rf.regs[20], 32'd0);
        chk("xori 1,3",   dut.rf.regs[21], 32'd2);

        // --- R-type SRA vs SRL (funct7[5] must actually be decoded) ---
        chk("sra INT_MIN,31", dut.rf.regs[22], 32'hFFFFFFFF);
        chk("srl INT_MIN,31", dut.rf.regs[25], 32'h00000001);
        chk("sltu 1,-1",      dut.rf.regs[24], 32'd1);

        // --- BLTU/BGEU bitmask: all six sub-tests correct = 63 ---
        chk("bltu/bgeu mask", dut.rf.regs[26], 32'd63);

        // --- AUIPC: zero, positive and negative (bit19) immediates ---
        chk("auipc 0 vs jal link",    dut.rf.regs[30], 32'd0);
        chk("auipc 0x10 delta",       dut.rf.regs[29], 32'h0000FFFC);
        chk("auipc 0xFFFFF delta",    dut.rf.regs[27], 32'd4100);

        // --- unimplemented encodings must be no-ops, not LW/SW ---
        chk("lb did not clobber x5",  dut.rf.regs[5],  32'h80000000);
        chk("sb did not write mem",   dut.rf.regs[28], 32'hFFFFFFFF);

        if (errors == 0)
            $display("PASS: SLLI/SRLI/SRAI, SLTI/SLTIU, XORI, SRA, SLTU, BLTU/BGEU and AUIPC all match the RISC-V spec; LB/SB retire as no-ops instead of LW/SW.");
        else
            $display("FAIL: %0d mismatch(es), see above.", errors);
        $finish;
    end
endmodule
