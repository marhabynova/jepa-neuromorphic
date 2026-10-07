`timescale 1ns/1ps

// §98: proves the generic MMIO bus genuinely supports TWO peripherals at
// disjoint addresses (spi_master 0x10-0x20, gpio 0x30/0x34) without
// conflict - not just "it works because only one was ever wired up."
// program_multi_periph.hex interleaves GPIO writes/reads around a full
// SPI transfer; success requires ALL of: both GPIO reads reflecting the
// driven input, the correct SPI loopback byte, AND gpio_out holding the
// LATEST value written (0x55) at the end - proving neither peripheral's
// state leaked into or got corrupted by the other's bus traffic.
module tb_soc_multi_periph;
    reg clk, rst;
    wire spi_sck, spi_mosi, spi_cs_n;
    wire [31:0] debug_out;
    wire debug_valid;
    wire [7:0] gpio_out_w;
    reg  [7:0] gpio_in_r;

    soc_top dut (
        .clk(clk), .rst(rst),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(spi_mosi), .spi_cs_n(spi_cs_n),
        .debug_out(debug_out), .debug_valid(debug_valid),
        .gpio_out(gpio_out_w), .gpio_in(gpio_in_r)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; gpio_in_r = 8'hC3;
        #12 rst = 0;
        #3000;
        $display("x6  (1st GPIO_IN read)      = %h (expect c3)", dut.cpu_u.rf.regs[6]);
        $display("x9  (SPI RXDATA)            = %h (expect a5)", dut.cpu_u.rf.regs[9]);
        $display("x11 (2nd GPIO_IN read)      = %h (expect c3)", dut.cpu_u.rf.regs[11]);
        $display("gpio_out (final, after 2 writes 0xAB then 0x55) = %h (expect 55)", gpio_out_w);
        if (dut.cpu_u.rf.regs[6] == 8'hC3 && dut.cpu_u.rf.regs[9] == 8'hA5 &&
            dut.cpu_u.rf.regs[11] == 8'hC3 && gpio_out_w == 8'h55)
            $display("PASS: SPI and GPIO coexist correctly on the same generic bus - no address-decode conflict.");
        else
            $display("FAIL: see register/signal dump above.");
        $finish;
    end
endmodule
