`timescale 1ns/1ps

// §99: spi_master.v verified against an INDEPENDENT slave model (not
// loopback - same methodology as tb_i2c.v/i2c_slave_model.v, and the
// Opus audit's own explicit recommendation after finding A4/A5 - bugs a
// self-consistent loopback test structurally cannot catch, since both
// sides of a loopback are the SAME wrong implementation agreeing with
// itself). Covers both directions (master->slave and slave->master) in
// all 4 SPI modes.
//
// Four separate slave instances (one per fixed CPOL/CPHA combination,
// since Verilog parameters are compile-time) all watch the same sck/
// mosi/cs_n; the testbench selects which one's miso output actually
// feeds back to the master for whichever mode is under test right now -
// the other three still process every transfer (harmlessly decoding
// garbage for modes that aren't theirs), but their outputs are unused.
module tb_spi_slave_model;
    reg clk, rst;
    reg bus_we, bus_re;
    reg [31:0] bus_addr, bus_wdata;
    wire [31:0] bus_rdata;
    wire spi_sck, spi_mosi, spi_cs_n;
    reg spi_miso;

    reg [7:0] slave_tx_byte;
    reg [1:0] cur_mode;

    wire [7:0] last_rx0, last_rx1, last_rx2, last_rx3;
    wire rx_valid0, rx_valid1, rx_valid2, rx_valid3;
    wire miso0, miso1, miso2, miso3;

    spi_master dut (
        .clk(clk), .rst(rst),
        .bus_we(bus_we), .bus_re(bus_re), .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_rdata(bus_rdata),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(spi_miso), .spi_cs_n(spi_cs_n)
    );

    spi_slave_model #(.CPOL(0), .CPHA(0)) slave0 (.sck(spi_sck), .mosi(spi_mosi), .cs_n(spi_cs_n), .miso(miso0), .tx_byte(slave_tx_byte), .last_rx(last_rx0), .rx_valid(rx_valid0));
    spi_slave_model #(.CPOL(0), .CPHA(1)) slave1 (.sck(spi_sck), .mosi(spi_mosi), .cs_n(spi_cs_n), .miso(miso1), .tx_byte(slave_tx_byte), .last_rx(last_rx1), .rx_valid(rx_valid1));
    spi_slave_model #(.CPOL(1), .CPHA(0)) slave2 (.sck(spi_sck), .mosi(spi_mosi), .cs_n(spi_cs_n), .miso(miso2), .tx_byte(slave_tx_byte), .last_rx(last_rx2), .rx_valid(rx_valid2));
    spi_slave_model #(.CPOL(1), .CPHA(1)) slave3 (.sck(spi_sck), .mosi(spi_mosi), .cs_n(spi_cs_n), .miso(miso3), .tx_byte(slave_tx_byte), .last_rx(last_rx3), .rx_valid(rx_valid3));

    always @(*) begin
        case (cur_mode)
            2'd0: spi_miso = miso0;
            2'd1: spi_miso = miso1;
            2'd2: spi_miso = miso2;
            default: spi_miso = miso3;
        endcase
    end

    always #5 clk = ~clk;

    task write_reg(input [7:0] off, input [31:0] data);
        begin
            bus_addr = {24'hFF0000, off};
            bus_wdata = data;
            bus_we = 1;
            @(posedge clk);
            #1;
            bus_we = 0;
        end
    endtask

    task do_transfer(input [7:0] tx_byte, input cpol_bit, input cpha_bit, input [15:0] div, output [7:0] rx_byte);
        begin
            write_reg(8'h14, div);
            write_reg(8'h10, {28'b0, cpha_bit, cpol_bit, 2'b00});
            write_reg(8'h18, tx_byte);
            write_reg(8'h10, {28'b0, cpha_bit, cpol_bit, 2'b01});
            wait (dut.busy == 1'b1);
            wait (dut.busy == 1'b0);
            @(posedge clk); #1;
            rx_byte = dut.rxdata_reg;
        end
    endtask

    reg [7:0] rx;
    integer errors;

    initial begin
        #10000 begin
            $display("WATCHDOG TIMEOUT");
            $finish;
        end
    end

    initial begin
        clk = 0; rst = 1; bus_we = 0; bus_re = 0; bus_addr = 0; bus_wdata = 0;
        slave_tx_byte = 8'h00; cur_mode = 2'd0; errors = 0;
        #12 rst = 0;
        @(posedge clk);

        // --- Mode 0 (CPOL=0,CPHA=0): master->slave, then slave->master ---
        cur_mode = 2'd0;
        do_transfer(8'hA5, 1'b0, 1'b0, 16'd2, rx);
        $display("mode0 master->slave: slave0 last_rx=%h rx_valid=%b (expect a5,1)", last_rx0, rx_valid0);
        if (last_rx0 !== 8'hA5) errors = errors + 1;  // rx_valid checked too late (cleared by cs_n deassert) - last_rx persisting is the real proof

        slave_tx_byte = 8'h3C;
        do_transfer(8'h00, 1'b0, 1'b0, 16'd2, rx);
        $display("mode0 slave->master: master RXDATA=%h (expect 3c)", rx);
        if (rx !== 8'h3C) errors = errors + 1;

        // --- Mode 1 (CPOL=0,CPHA=1) ---
        cur_mode = 2'd1;
        do_transfer(8'h7E, 1'b0, 1'b1, 16'd2, rx);
        $display("mode1 master->slave: slave1 last_rx=%h rx_valid=%b (expect 7e,1)", last_rx1, rx_valid1);
        if (last_rx1 !== 8'h7E) errors = errors + 1;

        slave_tx_byte = 8'hC3;
        do_transfer(8'h00, 1'b0, 1'b1, 16'd2, rx);
        $display("mode1 slave->master: master RXDATA=%h (expect c3)", rx);
        if (rx !== 8'hC3) errors = errors + 1;

        // --- Mode 2 (CPOL=1,CPHA=0) ---
        cur_mode = 2'd2;
        do_transfer(8'h91, 1'b1, 1'b0, 16'd2, rx);
        $display("mode2 master->slave: slave2 last_rx=%h rx_valid=%b (expect 91,1)", last_rx2, rx_valid2);
        if (last_rx2 !== 8'h91) errors = errors + 1;

        slave_tx_byte = 8'h6E;
        do_transfer(8'h00, 1'b1, 1'b0, 16'd2, rx);
        $display("mode2 slave->master: master RXDATA=%h (expect 6e)", rx);
        if (rx !== 8'h6E) errors = errors + 1;

        // --- Mode 3 (CPOL=1,CPHA=1) ---
        cur_mode = 2'd3;
        do_transfer(8'h5A, 1'b1, 1'b1, 16'd2, rx);
        $display("mode3 master->slave: slave3 last_rx=%h rx_valid=%b (expect 5a,1)", last_rx3, rx_valid3);
        if (last_rx3 !== 8'h5A) errors = errors + 1;

        slave_tx_byte = 8'hA1;
        do_transfer(8'h00, 1'b1, 1'b1, 16'd2, rx);
        $display("mode3 slave->master: master RXDATA=%h (expect a1)", rx);
        if (rx !== 8'hA1) errors = errors + 1;

        if (errors == 0)
            $display("PASS: spi_master verified against an independent slave model, both directions, all 4 modes.");
        else
            $display("FAIL: %0d error(s), see above.", errors);

        $finish;
    end
endmodule
