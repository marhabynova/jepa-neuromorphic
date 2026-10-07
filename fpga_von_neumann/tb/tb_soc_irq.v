`timescale 1ns/1ps

// §96: real interrupt-driven end-to-end check (program_irq.hex). main()
// enables interrupts then runs a busy-wait loop with ZERO SPI polling;
// success requires BOTH x30 (handler ran) and x31 (main loop resumed and
// finished normally afterward) to be set, and debug_out to hold the
// correct loopback byte - proof the interrupt actually fired during the
// loop (not before EI, not stuck forever) and execution resumed correctly
// rather than the handler just looping forever or corrupting state.
module tb_soc_irq;
    reg clk, rst;
    wire spi_sck, spi_mosi, spi_cs_n;
    wire [31:0] debug_out;
    wire debug_valid;

    soc_top dut (
        .clk(clk), .rst(rst),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(spi_mosi), .spi_cs_n(spi_cs_n),
        .debug_out(debug_out), .debug_valid(debug_valid)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1;
        #12 rst = 0;
        #4000;
        $display("x30 (handler ran)        = %0d (expect 1)", dut.cpu_u.rf.regs[30]);
        $display("x31 (main resumed+done)  = %0d (expect 777)", dut.cpu_u.rf.regs[31]);
        $display("debug_out (RXDATA via irq handler) = %h (expect a5)", debug_out);
        if (dut.cpu_u.rf.regs[30] == 1 && dut.cpu_u.rf.regs[31] == 777 && debug_out == 32'h000000a5)
            $display("PASS: interrupt fired during the busy-loop (no polling), handler ran, execution resumed correctly.");
        else
            $display("FAIL: see register dump above.");
        $finish;
    end
endmodule
