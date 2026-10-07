`timescale 1ns/1ps

// §95: verifies JALR (real call/return, not just JAL's fixed-offset jump)
// and LUI (32-bit constant build) against program_jalr_lui.hex.
module tb_jalr_lui;
    reg clk, rst;
    wire [31:0] mmio_out;
    wire mmio_valid;
    wire mmio_we, mmio_re;
    wire [31:0] mmio_addr, mmio_wdata;

    cpu dut (
        .clk(clk), .rst(rst),
        .mmio_out(mmio_out), .mmio_valid(mmio_valid),
        .mmio_we(mmio_we), .mmio_re(mmio_re),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_rdata(32'd0)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1;
        #12 rst = 0;
        #500;
        $display("x10 = %0d (expect 5)", dut.rf.regs[10]);
        $display("x11 = %0d (expect 105, set INSIDE func, proves func body ran)", dut.rf.regs[11]);
        $display("x12 = %0d (expect 999, set AFTER jalr returns - proves real call/return)", dut.rf.regs[12]);
        $display("x20 = %0h (expect 12345000, LUI result)", dut.rf.regs[20]);
        if (dut.rf.regs[10] == 5 && dut.rf.regs[11] == 105 && dut.rf.regs[12] == 999 &&
            dut.rf.regs[20] == 32'h12345000)
            $display("PASS: JALR call/return and LUI both work correctly.");
        else
            $display("FAIL: see register dump above.");
        $finish;
    end
endmodule
