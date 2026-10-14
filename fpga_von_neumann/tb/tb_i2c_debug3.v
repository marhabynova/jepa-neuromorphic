`timescale 1ns/1ps
module tb_i2c_debug3;
    reg clk, rst;
    reg bus_we;
    reg [31:0] bus_addr, bus_wdata;
    wire [31:0] bus_rdata;
    // §100: SCL is open-drain now too, so the bench must model the bus
    // the way a real board does - a pull-up that anything may pull low -
    // instead of taking a push-pull output straight from the master.
    wire m_scl_o, m_scl_oe;
    wire scl = !(m_scl_oe && !m_scl_o);
    wire m_sda_o, m_sda_oe;
    wire s_sda_o, s_sda_oe;
    reg [7:0] slave_tx_byte;
    wire [7:0] slave_last_rx;
    wire slave_rx_valid;

    wire sda = !((m_sda_oe && !m_sda_o) || (s_sda_oe && !s_sda_o));

    i2c_master dut (
        .clk(clk), .rst(rst),
        .bus_we(bus_we), .bus_re(1'b0), .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_rdata(bus_rdata),
        .scl_o(m_scl_o), .scl_oe(m_scl_oe), .scl_i(scl), .sda_o(m_sda_o), .sda_oe(m_sda_oe), .sda_i(sda)
    );
    i2c_slave_model #(.MY_ADDR(7'h43)) slave (
        .rst(rst),
        .scl(scl), .sda_i(sda), .sda_o(s_sda_o), .sda_oe(s_sda_oe),
        .last_rx(slave_last_rx), .rx_valid(slave_rx_valid), .tx_byte(slave_tx_byte)
    );

    always #5 clk = ~clk;
    always @(scl) $display("t=%0t scl=%b sda=%b m_oe=%b m_o=%b s_oe=%b s_o=%b mstate=%0d mbit=%0d rxd=%h sstate=%0d sbit=%0d",
                            $time, scl, sda, m_sda_oe, m_sda_o, s_sda_oe, s_sda_o, dut.state, dut.bit_idx, dut.rxdata_reg, slave.st, slave.bit_cnt);

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

    initial begin
        clk = 0; rst = 1; bus_we = 0; bus_addr = 0; bus_wdata = 0;
        slave_tx_byte = 8'h5A;
        #12 rst = 0;
        @(posedge clk);
        write_reg(8'h44, 16'd3);
        write_reg(8'h48, (7'h43 << 1) | 1'b1); // address byte, RD=1
        write_reg(8'h40, 32'b0001);             // START=1, STOP_AFTER=0
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        $display("--- address phase done (addressed for READ) ---");
        write_reg(8'h40, 32'b1111); // START=1 STOP_AFTER=1 IS_READ=1 NACK_AFTER=1
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        $display("--- read done, rxdata_reg=%h ---", dut.rxdata_reg);
        bus_addr = 32'hFF000050; @(posedge clk); #1;
        $display("--- bus-level readback: bus_rdata=%h bus_addr=%h sel check ---", bus_rdata, bus_addr);
        $finish;
    end
endmodule
