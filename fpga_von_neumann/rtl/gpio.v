module gpio (
    input         clk,
    input         rst,

    // Generic peripheral bus from cpu.v, same pattern as spi_master.v -
    // decodes its own address range (0x30/0x34), coexists with spi_master
    // (0x10-0x20) and cpu.v's own debug port (0x00) on the same bus.
    input         bus_we,
    input         bus_re,
    input  [31:0] bus_addr,
    input  [31:0] bus_wdata,
    output [31:0] bus_rdata,

    output [7:0]  gpio_out,
    input  [7:0]  gpio_in
);
    // Register map:
    //   0xFF000030  GPIO_OUT (W) [7:0] - drives gpio_out
    //   0xFF000034  GPIO_IN  (R) [7:0] - reads gpio_in
    localparam ADDR_GPIO_OUT = 8'h30;
    localparam ADDR_GPIO_IN  = 8'h34;

    wire sel_out = bus_addr[7:0] == ADDR_GPIO_OUT;
    wire sel_in  = bus_addr[7:0] == ADDR_GPIO_IN;

    reg [7:0] out_reg;
    always @(posedge clk or posedge rst) begin
        if (rst) out_reg <= 8'd0;
        else if (bus_we && sel_out) out_reg <= bus_wdata[7:0];
    end
    assign gpio_out = out_reg;

    reg [31:0] rd_data;
    always @(posedge clk) begin
        if (sel_in) rd_data <= {24'd0, gpio_in};
        else        rd_data <= 32'd0;
    end
    assign bus_rdata = rd_data;
endmodule
