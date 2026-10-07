`timescale 1ns/1ps
module tb_spi_debug;
    reg clk, rst;
    reg bus_we;
    reg [31:0] bus_addr, bus_wdata;
    wire [31:0] bus_rdata;
    wire spi_sck, spi_mosi, spi_cs_n;

    spi_master dut (
        .clk(clk), .rst(rst),
        .bus_we(bus_we), .bus_re(1'b0), .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_rdata(bus_rdata),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(spi_mosi), .spi_cs_n(spi_cs_n)
    );
    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; bus_we = 0; bus_addr = 0; bus_wdata = 0;
        #12 rst = 0;
        @(posedge clk);
        // CLKDIV=2
        bus_addr = 32'hFF000014; bus_wdata = 32'd2; bus_we = 1; @(posedge clk); #1; bus_we = 0;
        $display("after CLKDIV write: clkdiv=%0d", dut.clkdiv);
        // CTRL mode0, START=0
        bus_addr = 32'hFF000010; bus_wdata = 32'b0; bus_we = 1; @(posedge clk); #1; bus_we = 0;
        $display("after CTRL(mode0) write: cpol=%b cpha=%b cs_force=%b", dut.cpol, dut.cpha, dut.cs_force);
        // TXDATA = A5
        bus_addr = 32'hFF000018; bus_wdata = 8'hA5; bus_we = 1; @(posedge clk); #1; bus_we = 0;
        $display("after TXDATA write: txdata_reg=%h sel_txdata=%b", dut.txdata_reg, dut.sel_txdata);
        // CTRL START=1
        bus_addr = 32'hFF000010; bus_wdata = 32'b1; bus_we = 1; @(posedge clk); #1; bus_we = 0;
        $display("after START write: busy=%b shift_reg=%b txdata_reg=%h sel_ctrl=%b start_pulse_was=%b",
                  dut.busy, dut.shift_reg, dut.txdata_reg, dut.sel_ctrl, (dut.busy));

        repeat (40) begin
            @(posedge clk);
            $display("t=%0t busy=%b sck=%b mosi=%b cs_n=%b bits_left=%0d shift_reg=%b rx=%b div_cnt=%0d",
                      $time, dut.busy, spi_sck, spi_mosi, spi_cs_n, dut.bits_left, dut.shift_reg, dut.rxdata_reg, dut.div_cnt);
        end
        $finish;
    end
endmodule
