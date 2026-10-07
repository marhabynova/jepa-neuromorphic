// §99: synthesis-only top module. nextpnr-ice40's first real place-and-
// route run against soc_top directly failed with a genuine pin-budget
// problem: 59 I/O pins requested, but the UP5K-SG48 package (the one
// matching the recommended UPduino v3.1 board) only has 39 user I/O -
// debug_out[31:0] alone is 32 of those 59 pins. debug_out/debug_valid
// are cpu.v's simulation-observability debug port (RESEARCH_NOTES.md
// §46 - added so yosys's synth pass wouldn't sweep the whole design away
// as unobserved dead logic back when there was nothing else to observe);
// a real board has no reason to wire 32 raw pins to that register, and
// every testbench that cares about it (tb_cpu.v, tb_soc_spi.v,
// tb_soc_irq.v) instantiates soc_top directly, not this module - so
// dropping it here has zero effect on any existing test.
//
// This wraps soc_top unchanged (not a duplicate/parallel copy of its
// wiring) and simply doesn't route debug_out/debug_valid to top-level
// pins - yosys optimizes the now-unused connections away during
// synthesis, so this costs nothing in LUTs either.
module synth_top (
    input  clk,
    input  rst,
    output        spi_sck,
    output        spi_mosi,
    input         spi_miso,
    output        spi_cs_n,
    output [7:0]  gpio_out,
    input  [7:0]  gpio_in,
    output        i2c_scl,
    output        i2c_sda_o,
    output        i2c_sda_oe,
    input         i2c_sda_i
);
    wire [31:0] unused_debug_out;
    wire        unused_debug_valid;

    soc_top soc_u (
        .clk(clk), .rst(rst),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(spi_miso), .spi_cs_n(spi_cs_n),
        .debug_out(unused_debug_out), .debug_valid(unused_debug_valid),
        .gpio_out(gpio_out), .gpio_in(gpio_in),
        .i2c_scl(i2c_scl), .i2c_sda_o(i2c_sda_o), .i2c_sda_oe(i2c_sda_oe), .i2c_sda_i(i2c_sda_i)
    );
endmodule
