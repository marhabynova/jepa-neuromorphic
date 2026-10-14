`timescale 1ns/1ps

// §99: i2c_master.v verified against an INDEPENDENT slave model (not
// loopback - the Opus audit's own methodology, since spi_master.v's real
// bugs (CPHA off-by-one, SCK park level) were specifically the kind of
// thing a self-consistent loopback test structurally cannot catch).
module tb_i2c;
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

    // Open-drain combine: the bus reads low if EITHER device is actively
    // pulling it low (oe=1 AND o=0); otherwise it's released (pulled high).
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

    integer errors;
    reg [7:0] status;

    initial begin
        clk = 0; rst = 1; bus_we = 0; bus_addr = 0; bus_wdata = 0;
        slave_tx_byte = 8'h00; errors = 0;
        #12 rst = 0;
        @(posedge clk);

        write_reg(8'h44, 16'd3); // CLKDIV - fast for simulation

        // --- TEST 1: write to the slave's real address (0x43), WR bit=0,
        // should ACK (ack_error must stay clear), and the slave should
        // receive the data byte correctly. One transaction, 2 bytes
        // (address, then data), STOP after the second.
        write_reg(8'h48, (7'h43 << 1) | 1'b0); // TXDATA = address<<1 | WR(0)
        write_reg(8'h40, 32'b0001);             // CTRL: START=1, STOP_AFTER=0, WR
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        // BUG FIX (caught by cross-checking against dut.ack_error directly
        // in a debug testbench, not by inspection): bus_rdata is a
        // SYNCHRONOUS register (updates on posedge clk), so reading it
        // right after setting bus_addr with only a #1 delay (no clock
        // edge) returns whatever was last clocked in for a DIFFERENT
        // address - stale, not the STATUS value just requested. Both
        // TEST1 and TEST2 were reading this way; TEST1 happened to pass
        // anyway because the stale value coincidentally matched its
        // expected 0.
        bus_addr = 32'hFF00004C; @(posedge clk); #1;
        status = bus_rdata[1:0];
        $display("TEST1 (address byte, matching addr): ack_error=%b (expect 0)", status[1]);
        if (status[1] !== 1'b0) errors = errors + 1;

        write_reg(8'h48, 8'hA5);     // TXDATA = data byte 0xA5
        write_reg(8'h40, 32'b0011);  // CTRL: START=1, STOP_AFTER=1, WR
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        #1;
        $display("TEST1 slave received: %h (expect a5), rx_valid pulsed=%b", slave_last_rx, slave_rx_valid);
        if (slave_last_rx !== 8'hA5) errors = errors + 1;

        // --- TEST 2: write to a NON-matching address - slave must NOT
        // ack, ack_error must be set.
        write_reg(8'h48, (7'h10 << 1) | 1'b0); // some other address
        write_reg(8'h40, 32'b0011);             // START=1, STOP_AFTER=1, WR
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        bus_addr = 32'hFF00004C; @(posedge clk); #1;
        status = bus_rdata[1:0];
        $display("TEST2 (non-matching addr): ack_error=%b (expect 1)", status[1]);
        if (status[1] !== 1'b1) errors = errors + 1;

        // --- TEST 3: read a byte from the slave (address 0x43, RD=1).
        slave_tx_byte = 8'h5A;
        write_reg(8'h48, (7'h43 << 1) | 1'b1); // address byte, RD=1
        write_reg(8'h40, 32'b0001);             // START=1, STOP_AFTER=0, WR(address phase is always "written")
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        write_reg(8'h40, 32'b1111);  // CTRL: START=1 STOP_AFTER=1 IS_READ=1 NACK_AFTER=1 (last/only byte)
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        bus_addr = 32'hFF000050; @(posedge clk); #1;
        $display("TEST3 (read): RXDATA=%h (expect 5a)", bus_rdata[7:0]);
        if (bus_rdata[7:0] !== 8'h5A) errors = errors + 1;

        if (errors == 0)
            $display("PASS: i2c_master verified against an independent slave model (address ACK/NACK, write, read).");
        else
            $display("FAIL: %0d error(s), see above.", errors);
        $finish;
    end
endmodule
