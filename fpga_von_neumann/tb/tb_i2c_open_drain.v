`timescale 1ns/1ps

// §100: electrical-level conformance test for i2c_master.v, written from
// the I2C bus specification rather than from the implementation.
//
// The spec facts this bench encodes:
//   1. I2C is a WIRED-AND bus. No device ever drives a line high; a
//      device either pulls a line low or releases it, and an external
//      pull-up supplies every high level. So at EVERY instant, "output
//      enable asserted" must imply "the driven value is 0".
//   2. START = SDA falls while SCL is high. STOP = SDA rises while SCL
//      is high. Both must appear as real, separate bus events.
//   3. A slave may hold SDA low (ACK, or a data byte of all zeros) and
//      the master must read the line, not its own intent.
//   4. A slave may hold SCL low to stall the master (clock stretching);
//      the master must wait for the line to actually go high rather
//      than free-running on its internal divider.
//   5. Standard-mode I2C runs at 100 kHz MAXIMUM. The reset-default
//      CLKDIV must therefore put SCL at or below 100 kHz on this SoC's
//      12 MHz clock, without software having to write CLKDIV first.
//
// Nothing here looks at i2c_master's internal state machine; everything
// is observed on the modelled bus wires, so the bench stays valid if the
// FSM is reorganised.
module tb_i2c_open_drain;
    // The SoC's real clock rate. The bench's own simulation clock is
    // arbitrary (10 ns here), so SCL rate is measured in CLOCK CYCLES
    // and converted with this constant - that keeps the 100 kHz check
    // meaningful regardless of the testbench timescale.
    localparam real SOC_CLK_HZ = 12_000_000.0;

    reg clk, rst;
    reg bus_we;
    reg [31:0] bus_addr, bus_wdata;
    wire [31:0] bus_rdata;

    wire m_scl_o, m_scl_oe;
    wire m_sda_o, m_sda_oe;
    wire s_sda_o, s_sda_oe;
    reg  force_sda_low;   // a third party (or a stuck slave) pulling SDA
    reg  force_scl_low;   // a slave stretching the clock

    reg [7:0] slave_tx_byte;
    wire [7:0] slave_last_rx;
    wire slave_rx_valid;

    // The bus itself: high unless SOMEBODY pulls it low. This is the
    // whole point - a pull-up plus open-drain drivers, wired-AND.
    wire sda = !((m_sda_oe && !m_sda_o) || (s_sda_oe && !s_sda_o) || force_sda_low);
    wire scl = !((m_scl_oe && !m_scl_o) || force_scl_low);

    i2c_master dut (
        .clk(clk), .rst(rst),
        .bus_we(bus_we), .bus_re(1'b0), .bus_addr(bus_addr),
        .bus_wdata(bus_wdata), .bus_rdata(bus_rdata),
        .scl_o(m_scl_o), .scl_oe(m_scl_oe), .scl_i(scl),
        .sda_o(m_sda_o), .sda_oe(m_sda_oe), .sda_i(sda)
    );

    i2c_slave_model #(.MY_ADDR(7'h43)) slave (
        .rst(rst),
        .scl(scl), .sda_i(sda), .sda_o(s_sda_o), .sda_oe(s_sda_oe),
        .last_rx(slave_last_rx), .rx_valid(slave_rx_valid), .tx_byte(slave_tx_byte)
    );

    always #5 clk = ~clk;

    // ---- SPEC RULE 1: the master may never drive a line high --------
    // Checked on every change of either pin, for the whole run, not
    // just at sample points.
    integer od_violations;
    always @(m_sda_oe or m_sda_o or m_scl_oe or m_scl_o) begin
        if (!rst) begin
            if (m_sda_oe === 1'b1 && m_sda_o !== 1'b0) begin
                $display("  OPEN-DRAIN VIOLATION t=%0t: SDA driven with oe=1, o=%b", $time, m_sda_o);
                od_violations = od_violations + 1;
            end
            if (m_scl_oe === 1'b1 && m_scl_o !== 1'b0) begin
                $display("  OPEN-DRAIN VIOLATION t=%0t: SCL driven with oe=1, o=%b", $time, m_scl_o);
                od_violations = od_violations + 1;
            end
        end
    end

    // ---- SPEC RULE 2: START/STOP must appear as real bus events -----
    reg scl_p, sda_p;
    integer n_start, n_stop;
    integer cyc;
    always @(posedge clk) begin
        if (rst) begin
            scl_p <= 1'b1; sda_p <= 1'b1; n_start <= 0; n_stop <= 0; cyc <= 0;
        end else begin
            if (scl && scl_p && sda_p && !sda) n_start <= n_start + 1;
            if (scl && scl_p && !sda_p && sda) n_stop  <= n_stop + 1;
            scl_p <= scl; sda_p <= sda;
            cyc <= cyc + 1;
        end
    end

    integer errors;
    integer t0, scl_period;
    real scl_hz;
    reg [7:0] status;

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

    // Watchdog: a clock-stretch bug can hang the FSM forever instead of
    // failing an assertion, so the run must be bounded.
    initial begin
        #4000000;
        $display("FAIL: WATCHDOG TIMEOUT - the master never finished a transfer.");
        $finish;
    end

    initial begin
        clk = 0; rst = 1; bus_we = 0; bus_addr = 0; bus_wdata = 0;
        slave_tx_byte = 8'h00; errors = 0; od_violations = 0;
        force_sda_low = 0; force_scl_low = 0;
        #12 rst = 0;
        @(posedge clk);

        // ============ TEST 1: reset-default SCL rate <= 100 kHz ========
        // Deliberately does NOT write CLKDIV - the point is what the
        // peripheral does out of reset, which is what a board would see
        // if firmware forgot to configure it.
        write_reg(8'h48, (7'h43 << 1) | 1'b0); // TXDATA = address, write
        write_reg(8'h40, 32'b0011);            // START + STOP_AFTER
        @(posedge scl);                        // first clock pulse of the byte
        t0 = cyc;
        @(posedge scl);
        scl_period = cyc - t0;
        scl_hz = SOC_CLK_HZ / scl_period;
        $display("TEST1: SCL period = %0d clock cycles -> %0.1f Hz at %0.0f Hz clk (I2C standard mode: 100000 max)",
                 scl_period, scl_hz, SOC_CLK_HZ);
        if (scl_hz > 100000.0) begin
            $display("  MISMATCH: reset-default SCL is FASTER than I2C standard mode allows");
            errors = errors + 1;
        end
        // Guard against "fixed" by making it absurdly slow, too.
        if (scl_hz < 90000.0) begin
            $display("  MISMATCH: reset-default SCL is far below 100 kHz, divider is wrong");
            errors = errors + 1;
        end
        wait (dut.busy == 1'b0);

        // ============ TEST 2: START and STOP really happened ==========
        $display("TEST2: start conditions seen = %0d, stop conditions seen = %0d (expect >=1 each)",
                 n_start, n_stop);
        if (n_start < 1) begin $display("  MISMATCH: no START condition on the bus"); errors = errors + 1; end
        if (n_stop  < 1) begin $display("  MISMATCH: no STOP condition on the bus");  errors = errors + 1; end

        // Speed the clock up for the remaining functional tests.
        write_reg(8'h44, 16'd3);

        // ============ TEST 3: a slave holding SDA low =================
        // tx_byte = 0x00 means the slave pulls SDA low for all eight
        // data bits. The master must read 0x00 (the LINE), and must
        // never fight the slave for the line while it does.
        slave_tx_byte = 8'h00;
        write_reg(8'h48, (7'h43 << 1) | 1'b1); // address byte, read
        write_reg(8'h40, 32'b0001);            // START, no stop
        wait (dut.busy == 1'b1); wait (dut.busy == 1'b0);
        write_reg(8'h40, 32'b1111);            // START + STOP + IS_READ + NACK
        wait (dut.busy == 1'b1); wait (dut.busy == 1'b0);
        bus_addr = 32'hFF000050; @(posedge clk); #1;
        $display("TEST3: RXDATA with slave holding SDA low = %h (expect 00)", bus_rdata[7:0]);
        if (bus_rdata[7:0] !== 8'h00) begin
            $display("  MISMATCH: master did not read the line the slave was pulling low");
            errors = errors + 1;
        end

        // ============ TEST 4: clock stretching ========================
        // An external device holds SCL low across the moment the master
        // wants to release it. The master must wait; the byte must
        // still arrive intact. Without the stretch check the master
        // free-runs on its own divider and the slave loses bits.
        slave_tx_byte = 8'h00;
        write_reg(8'h48, (7'h43 << 1) | 1'b0); // address byte, write
        write_reg(8'h40, 32'b0001);            // START, stay held
        wait (dut.busy == 1'b1); wait (dut.busy == 1'b0);

        fork
            begin : stretcher
                // Grab the line while it is already low (the legal way
                // for a slave to extend a low period) and hold it far
                // past the master's own half-period.
                @(negedge scl);
                force_scl_low = 1;
                repeat (200) @(posedge clk);
                force_scl_low = 0;
            end
            begin : data_byte
                write_reg(8'h48, 8'h5A);
                write_reg(8'h40, 32'b0011);     // START + STOP_AFTER
                wait (dut.busy == 1'b1); wait (dut.busy == 1'b0);
            end
        join
        #1;
        $display("TEST4: slave received %h through a stretched clock (expect 5a)", slave_last_rx);
        if (slave_last_rx !== 8'h5A) begin
            $display("  MISMATCH: byte corrupted by clock stretching");
            errors = errors + 1;
        end

        // ============ TEST 5: the open-drain invariant overall ========
        $display("TEST5: open-drain violations over the whole run = %0d (expect 0)", od_violations);
        if (od_violations != 0) errors = errors + 1;

        if (errors == 0)
            $display("PASS: i2c_master is true open-drain on both lines, emits real START/STOP, reads a slave-held SDA, survives clock stretching, and resets to <=100 kHz.");
        else
            $display("FAIL: %0d problem(s), see above.", errors);
        $finish;
    end
endmodule
