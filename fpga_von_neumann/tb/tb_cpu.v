`timescale 1ns/1ps

module tb_cpu;
    reg clk;
    reg rst;
    wire [31:0] mmio_out;
    wire        mmio_valid;

    cpu dut (.clk(clk), .rst(rst), .mmio_out(mmio_out), .mmio_valid(mmio_valid));

    always #5 clk = ~clk;

    // NOTE: mmio_out is a register that HOLDS its value (doesn't pulse) -
    // only mmio_valid pulses for one cycle. Reading mmio_out directly at the
    // end of the run (after the one SW-to-MMIO instruction has long since
    // executed) is enough; no need to latch on mmio_valid's pulse. (First
    // draft tried latching on mmio_valid and had an off-by-one-cycle race -
    // mmio_valid and cpu.v's own `mmio_out <= rd_data2` both fire on the
    // same posedge, so sampling mmio_out at that edge reads its OLD value,
    // not the newly-written one. Confirmed via a throwaway debug testbench
    // with per-cycle $display before landing on this simpler fix.)

    initial begin
        clk = 0;
        rst = 1;
        #12 rst = 0;

        #400; // ~22 instructions need ~22 cycles; this gives plenty of margin

        $display("x1  = %0d (expect 5)",              dut.rf.regs[1]);
        $display("x2  = %0d (expect 10)",              dut.rf.regs[2]);
        $display("x3  = %0d (expect 15)",               dut.rf.regs[3]);
        $display("x4  = %0d (expect 15, via lw)",        dut.rf.regs[4]);
        $display("x5  = %0d (expect 42, NOT 999 -- BEQ)",      dut.rf.regs[5]);
        $display("x6  = %0d (expect 3)",                 dut.rf.regs[6]);
        $display("x7  = %0d (expect 2)",                 dut.rf.regs[7]);
        $display("x9  = %0d (expect 12, SLL 3<<2)",        dut.rf.regs[9]);
        $display("x10 = %0d (expect 3, SRL 12>>2)",        dut.rf.regs[10]);
        $display("x11 = %0d (expect 111, NOT 666 -- BNE)", dut.rf.regs[11]);
        $display("x12 = %0d (expect 333, NOT 222 -- BLT)", dut.rf.regs[12]);
        $display("x13 = %0d (expect 555, NOT 444 -- BGE)", dut.rf.regs[13]);
        $display("mmio_out = %0d (expect 777, SW to addr 0xFF000000)", mmio_out);

        if (dut.rf.regs[1] == 5 && dut.rf.regs[2] == 10 && dut.rf.regs[3] == 15 &&
            dut.rf.regs[4] == 15 && dut.rf.regs[5] == 42 && dut.rf.regs[6] == 3 &&
            dut.rf.regs[7] == 2 && dut.rf.regs[9] == 12 && dut.rf.regs[10] == 3 &&
            dut.rf.regs[11] == 111 && dut.rf.regs[12] == 333 && dut.rf.regs[13] == 555 &&
            mmio_out == 777) begin
            $display("PASS: single-cycle RV32I core (ADD/ADDI/LW/SW/SLL/SRL/BEQ/BNE/BLT/BGE/JAL + MMIO debug port) executed correctly.");
        end else begin
            $display("FAIL: see register dump above.");
        end

        $finish;
    end
endmodule
