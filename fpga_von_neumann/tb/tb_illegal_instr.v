`timescale 1ns/1ps

// §100: the reporting half of the "no silent mis-execution" fix.
// tb_rv32i_ext.v already proves the ARCHITECTURAL effect (LB/SB change
// nothing instead of acting as LW/SW); this bench proves the core also
// KNOWS it rejected something, via control.v's `illegal` and cpu.v's
// sticky illegal_seen/illegal_pc.
//
// The program (tools/asm_rv32i_ext_test.py) is entirely legal RV32I up
// to a deliberate `lb x5, 0(x0)` near the end, so:
//   - illegal_seen must be 1 (something was rejected), and
//   - illegal_pc must point at an instruction word that really IS that
//     LB encoding (0x00000283) - which simultaneously proves no EARLIER,
//     legal instruction was wrongly flagged, since illegal_pc latches
//     the FIRST offender only.
// Checking the opcode found at illegal_pc instead of a hard-coded
// address keeps this test independent of the program's layout.
//
// This bench deliberately references internal state: illegal_seen has no
// output port (a real board has no pin to spare - see §99c's pin-budget
// finding) and yosys sweeps it away during synthesis, so it exists for
// exactly this kind of verification.
module tb_illegal_instr;
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

    // lb x5, 0(x0) = imm 0 | rs1 x0 | funct3 000 | rd x5 | opcode 0000011
    localparam [31:0] LB_X5_0_X0 = 32'h00000283;

    integer errors;
    reg [31:0] flagged_instr;

    initial begin
        clk = 0; rst = 1; errors = 0;
        #12 rst = 0;
        #6000;

        flagged_instr = dut.imem_u.mem[dut.illegal_pc[11:2]];
        $display("illegal_seen = %b (expect 1)", dut.illegal_seen);
        $display("illegal_pc   = %0d, instruction there = %h (expect %h = lb x5,0(x0))",
                 dut.illegal_pc, flagged_instr, LB_X5_0_X0);

        if (dut.illegal_seen !== 1'b1) begin
            $display("  MISMATCH: no illegal instruction was reported at all");
            errors = errors + 1;
        end
        if (flagged_instr !== LB_X5_0_X0) begin
            $display("  MISMATCH: illegal_pc points at %h, not the deliberate LB", flagged_instr);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS: the first unimplemented encoding is reported, and no legal instruction before it was flagged.");
        else
            $display("FAIL: %0d problem(s), see above.", errors);
        $finish;
    end
endmodule
