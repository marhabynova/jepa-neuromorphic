`timescale 1ns/1ps

// End-to-end check: a real program running on the CPU (program_spi.hex,
// see tools/asm_spi_test.py) configures spi_master, starts an 8-bit
// transfer, polls STATUS itself (no testbench bus driving), and writes the
// received byte out the debug port. MISO looped to MOSI, so success means
// debug_out ends up 0xA5 - proof the CPU and the new peripheral bus
// actually talk to each other through real RV32I instructions, not just
// that spi_master works when driven directly (already shown by tb_spi.v).
module tb_soc_spi;
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
        #3000;
        $display("debug_out = %h (expect a5, SPI RXDATA read back by the CPU program itself)", debug_out);
        if (debug_out == 32'h000000a5)
            $display("PASS: CPU program configured+drove spi_master end-to-end via the MMIO bus.");
        else
            $display("FAIL.");
        $finish;
    end
endmodule
