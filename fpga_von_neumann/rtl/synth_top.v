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
    // §100: ONE physical pin each, bidirectional, as I2C requires. The
    // old port list exported i2c_sda_o / i2c_sda_oe / i2c_sda_i as three
    // SEPARATE top-level ports and i2c_scl as a plain output - so the
    // design asked the board for four pins to implement two bus lines,
    // and nothing anywhere actually tristated anything. An I2C line
    // physically cannot be built that way: SDA must be driven low,
    // released, and read, all on the same wire.
    inout         i2c_scl,
    inout         i2c_sda
);
    wire [31:0] unused_debug_out;
    wire        unused_debug_valid;

    wire scl_o, scl_oe, scl_i;
    wire sda_o, sda_oe, sda_i;

    // §100: the reset PIN is asynchronous to clk. Driving every
    // flip-flop's async reset from it directly (what this used to do)
    // means the design LEAVES reset on different clock edges in
    // different corners of the die. See reset_sync.v.
    wire rst_sync;
    reset_sync rst_sync_u (.clk(clk), .rst_in(rst), .rst_out(rst_sync));

    soc_top soc_u (
        .clk(clk), .rst(rst_sync),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(spi_miso), .spi_cs_n(spi_cs_n),
        .debug_out(unused_debug_out), .debug_valid(unused_debug_valid),
        .gpio_out(gpio_out), .gpio_in(gpio_in),
        .i2c_scl_o(scl_o), .i2c_scl_oe(scl_oe), .i2c_scl_i(scl_i),
        .i2c_sda_o(sda_o), .i2c_sda_oe(sda_oe), .i2c_sda_i(sda_i)
    );

    // Open-drain bidirectional pads, built from the iCE40 primitive
    // rather than from a Verilog `assign pin = oe ? 0 : 1'bz` (which
    // yosys would have to infer a tristate buffer from, and which says
    // nothing about the input path). PIN_TYPE = 6'b1010_01 is
    // PIN_OUTPUT_TRISTATE (bits 5:2) + simple unregistered input
    // (bits 1:0). D_OUT_0 is tied to 0 and only OUTPUT_ENABLE moves, so
    // the pad can pull the line low or let go - it can never source a
    // high, exactly like a real open-drain driver. PULLUP is left off:
    // the iCE40's internal weak pull-up is far too weak for an I2C bus,
    // which needs real external resistors (typically 2.2k-4.7k).
    SB_IO #(
        .PIN_TYPE(6'b1010_01),
        .PULLUP(1'b0)
    ) scl_pad (
        .PACKAGE_PIN(i2c_scl),
        .OUTPUT_ENABLE(scl_oe),
        .D_OUT_0(scl_o),
        .D_IN_0(scl_i)
    );

    SB_IO #(
        .PIN_TYPE(6'b1010_01),
        .PULLUP(1'b0)
    ) sda_pad (
        .PACKAGE_PIN(i2c_sda),
        .OUTPUT_ENABLE(sda_oe),
        .D_OUT_0(sda_o),
        .D_IN_0(sda_i)
    );
endmodule
