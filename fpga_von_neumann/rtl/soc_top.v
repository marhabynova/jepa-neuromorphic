module soc_top (
    input  clk,
    input  rst,
    output        spi_sck,
    output        spi_mosi,
    input         spi_miso,
    output        spi_cs_n,
    output [31:0] debug_out,
    output        debug_valid
);
    wire mmio_we, mmio_re;
    wire [31:0] mmio_addr, mmio_wdata, mmio_rdata;

    cpu cpu_u (
        .clk(clk), .rst(rst),
        .mmio_out(debug_out), .mmio_valid(debug_valid),
        .mmio_we(mmio_we), .mmio_re(mmio_re),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_rdata(mmio_rdata)
    );

    spi_master spi_u (
        .clk(clk), .rst(rst),
        .bus_we(mmio_we), .bus_re(mmio_re),
        .bus_addr(mmio_addr), .bus_wdata(mmio_wdata), .bus_rdata(mmio_rdata),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(spi_miso), .spi_cs_n(spi_cs_n)
    );
endmodule
