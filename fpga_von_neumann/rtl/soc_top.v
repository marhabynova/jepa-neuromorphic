module soc_top (
    input  clk,
    input  rst,
    output        spi_sck,
    output        spi_mosi,
    input         spi_miso,
    output        spi_cs_n,
    output [31:0] debug_out,
    output        debug_valid,
    // §98: second peripheral on the same generic bus as spi_master, at a
    // disjoint address range (0x30/0x34 vs SPI's 0x10-0x20) - proves the
    // bus genuinely supports more than one peripheral, not just one that
    // happened to work.
    output [7:0]  gpio_out,
    input  [7:0]  gpio_in
);
    wire mmio_we, mmio_re;
    wire [31:0] mmio_addr, mmio_wdata;
    wire [31:0] spi_rdata, gpio_rdata;
    wire spi_irq;

    // Each peripheral drives 0 on its own bus_rdata when its own address
    // isn't selected (see spi_master.v/gpio.v), so ORing them together is
    // a safe, correct combine as long as address ranges never overlap -
    // true here by construction (0x10-0x20 vs 0x30-0x34).
    wire [31:0] mmio_rdata = spi_rdata | gpio_rdata;

    cpu cpu_u (
        .clk(clk), .rst(rst),
        .mmio_out(debug_out), .mmio_valid(debug_valid),
        .mmio_we(mmio_we), .mmio_re(mmio_re),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_rdata(mmio_rdata),
        .irq(spi_irq)
    );

    spi_master spi_u (
        .clk(clk), .rst(rst),
        .bus_we(mmio_we), .bus_re(mmio_re),
        .bus_addr(mmio_addr), .bus_wdata(mmio_wdata), .bus_rdata(spi_rdata),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(spi_miso), .spi_cs_n(spi_cs_n),
        .irq_out(spi_irq)
    );

    gpio gpio_u (
        .clk(clk), .rst(rst),
        .bus_we(mmio_we), .bus_re(mmio_re),
        .bus_addr(mmio_addr), .bus_wdata(mmio_wdata), .bus_rdata(gpio_rdata),
        .gpio_out(gpio_out), .gpio_in(gpio_in)
    );
endmodule
