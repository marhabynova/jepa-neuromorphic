module i2c_master (
    input         clk,
    input         rst,

    input         bus_we,
    input         bus_re,
    input  [31:0] bus_addr,
    input  [31:0] bus_wdata,
    output [31:0] bus_rdata,

    // §99: Opus audit (RESEARCH_NOTES.md §99/D2) found the BrainBoard 1500
    // cannot be woken from sleep with SPI alone - its PI4IOE5V6408 GPIO
    // expander (address 0x43) must be driven over I2C first (AKD_SLEEP/
    // AKD_RST pins). This is the hard prerequisite the hardware plan was
    // missing. Open-drain SDA modeled the standard way for synthesizable
    // Verilog: sda_oe=1 actively drives sda_o, sda_oe=0 releases the line
    // for the external pull-up (or the slave) to control; sda_i is the
    // actual line state, always sampled regardless of who's driving.
    // No clock-stretching support (a slave holding SCL low to stall the
    // master) - not needed for the simple register read/writes the
    // BB15 bring-up sequence does, but a real limitation if a future
    // slave needs it.
    output        scl,
    output        sda_o,
    output        sda_oe,
    input         sda_i
);
    // Register map:
    //   0xFF000040  CTRL   (W) bit0=START(pulse) bit1=STOP_AFTER bit2=IS_READ bit3=NACK_AFTER(read only)
    //   0xFF000044  CLKDIV (W) [15:0] quarter-bit-period divider
    //   0xFF000048  TXDATA (W) [7:0]  byte to send (address+R/W for the first byte of a transaction, data after)
    //   0xFF00004C  STATUS (R) bit0=BUSY bit1=ACK_ERROR (no slave ACKed - e.g. nothing at that address)
    //   0xFF000050  RXDATA (R) [7:0]  byte received (for IS_READ bytes)
    //
    // Programming model: write TXDATA, write CTRL with START=1 to send a
    // byte. The FIRST byte of a transaction (while idle) also generates a
    // START condition. A byte written with STOP_AFTER=0 leaves the bus
    // "held" (SCL/SDA mid-transaction, no STOP yet) so the next START
    // pulse sends another byte WITHOUT a fresh START condition (standard
    // multi-byte I2C transaction, e.g. "write register address, then
    // write register data" as one continuous transaction). STOP_AFTER=1
    // generates a real STOP condition after this byte's ACK, returning to
    // idle. No repeated-START support - not needed for the BB15 bring-up
    // sequence (independent START..STOP transactions per register access).
    localparam ADDR_CTRL   = 8'h40;
    localparam ADDR_CLKDIV = 8'h44;
    localparam ADDR_TXDATA = 8'h48;
    localparam ADDR_STATUS = 8'h4C;
    localparam ADDR_RXDATA = 8'h50;

    wire sel_ctrl   = bus_addr[7:0] == ADDR_CTRL;
    wire sel_clkdiv = bus_addr[7:0] == ADDR_CLKDIV;
    wire sel_txdata = bus_addr[7:0] == ADDR_TXDATA;
    wire sel_status = bus_addr[7:0] == ADDR_STATUS;
    wire sel_rxdata = bus_addr[7:0] == ADDR_RXDATA;

    reg [15:0] clkdiv;
    reg [7:0]  txdata_reg;
    reg        busy;
    reg        ack_error;
    reg [7:0]  rxdata_reg;

    // IDLE/HELD both qualify "not busy" for accepting a new start_pulse -
    // HELD means a prior byte finished without STOP_AFTER, bus still owned.
    localparam S_IDLE=4'd0, S_START=4'd1, S_BIT_LOW=4'd2, S_BIT_HIGH=4'd3,
               S_HELD=4'd4, S_STOP_LOW=4'd5, S_STOP_HIGH1=4'd6, S_STOP_HIGH2=4'd7,
               S_START2=4'd8;
    reg [3:0] state;
    reg [15:0] div_cnt;
    reg [3:0] bit_idx;      // 0..7 data bits, 8 = ACK/NACK bit slot
    reg [7:0] shift_reg;
    reg scl_reg, sda_reg, sda_oe_reg;
    reg stop_after, is_read, nack_after;

    wire start_pulse = bus_we && sel_ctrl && bus_wdata[0] && !busy;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            clkdiv <= 16'd10;
        end else if (bus_we) begin
            if (sel_clkdiv)      clkdiv     <= bus_wdata[15:0];
            else if (sel_txdata) txdata_reg <= bus_wdata[7:0];
        end
    end

    reg [31:0] rd_data;
    always @(posedge clk) begin
        if (sel_status)      rd_data <= {30'd0, ack_error, busy};
        else if (sel_rxdata) rd_data <= {24'd0, rxdata_reg};
        else                 rd_data <= 32'd0;
    end
    assign bus_rdata = rd_data;

    wire tick = (div_cnt == clkdiv);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE; busy <= 1'b0; ack_error <= 1'b0;
            scl_reg <= 1'b1; sda_reg <= 1'b1; sda_oe_reg <= 1'b1;
            div_cnt <= 16'd0; bit_idx <= 4'd0; shift_reg <= 8'd0; rxdata_reg <= 8'd0;
        end else begin
            case (state)
                S_IDLE, S_HELD: begin
                    if (start_pulse) begin
                        busy       <= 1'b1;
                        stop_after <= bus_wdata[1];
                        is_read    <= bus_wdata[2];
                        nack_after <= bus_wdata[3];
                        shift_reg  <= txdata_reg;
                        bit_idx    <= 4'd0;
                        div_cnt    <= 16'd0;
                        if (state == S_IDLE) begin
                            // Fresh transaction: generate START (SDA falls
                            // while SCL is high - both already high at idle).
                            state <= S_START;
                        end else begin
                            // Already mid-transaction (held from a prior
                            // non-STOP byte) - go straight to clocking bits,
                            // no new START condition. BUG FIX (caught before
                            // simulation, by re-reading this against the
                            // is_read-dependent logic already present in
                            // S_BIT_HIGH below): a READ byte's first bit must
                            // RELEASE sda (oe=0) for the slave to drive, not
                            // force shift_reg[7] with oe=1 - that's only
                            // correct for a WRITE byte. Uses bus_wdata[2]
                            // directly (this cycle's incoming is_read),
                            // since the `is_read` register itself won't be
                            // updated until after this same nonblocking
                            // assignment settles.
                            scl_reg <= 1'b0;
                            if (bus_wdata[2]) begin
                                sda_oe_reg <= 1'b0;
                            end else begin
                                // BUG FIX (caught via tb_i2c_debug.v's per-
                                // edge trace, not by inspection): this reads
                                // txdata_reg, NOT shift_reg. `shift_reg <=
                                // txdata_reg` is scheduled in THIS SAME
                                // nonblocking-assignment cycle (above), so
                                // shift_reg[7] here would still see its OLD
                                // (pre-update) value, not the byte that was
                                // just loaded - the trace showed exactly
                                // this: sending 0xA5, every bit came out
                                // right except the very first one.
                                sda_reg <= txdata_reg[7]; sda_oe_reg <= 1'b1;
                            end
                            state <= S_BIT_LOW;
                        end
                    end
                end

                S_START: begin
                    if (!tick) begin
                        div_cnt <= div_cnt + 16'd1;
                    end else begin
                        div_cnt <= 16'd0;
                        // BUG FIX (caught via tb_i2c.v's TEST3, not by
                        // inspection): the original code set sda_reg and
                        // scl_reg to their post-START values in this SAME
                        // tick, so SDA and SCL changed at the exact same
                        // simulation time - a START condition (SDA falls
                        // WHILE SCL stays high) never actually appeared as
                        // a distinct bus event. An independent slave model
                        // (tb/i2c_slave_model.v) never saw a start_cond or
                        // stop_cond fire across the whole simulation as a
                        // result - it just free-ran through bit counts
                        // after the first (initial-state) transaction,
                        // which made TEST3's read return garbage despite
                        // TEST1/TEST2 happening to pass by coincidence.
                        // Fix: split into two ticks, matching the already-
                        // correct STOP sequence's own staggering.
                        sda_reg <= 1'b0; sda_oe_reg <= 1'b1; // START: SDA falls, SCL stays high this tick
                        state   <= S_START2;
                    end
                end

                S_START2: begin
                    if (!tick) begin
                        div_cnt <= div_cnt + 16'd1;
                    end else begin
                        div_cnt <= 16'd0;
                        scl_reg <= 1'b0;           // now drop SCL to begin clocking the first bit
                        sda_reg <= shift_reg[7];
                        state   <= S_BIT_LOW;
                    end
                end

                // One (SCL-low, SCL-high) pair per bit slot. Data bits
                // (0-7) drive/sample MSB-first; slot 8 is the ACK/NACK bit
                // - direction depends on is_read (set per-byte at start_pulse).
                S_BIT_LOW: begin
                    if (!tick) begin
                        div_cnt <= div_cnt + 16'd1;
                    end else begin
                        div_cnt <= 16'd0;
                        scl_reg <= 1'b1; // rising edge - this is the sample instant
                        state   <= S_BIT_HIGH;
                    end
                end

                S_BIT_HIGH: begin
                    if (!tick) begin
                        div_cnt <= div_cnt + 16'd1;
                    end else begin
                        div_cnt <= 16'd0;
                        scl_reg <= 1'b0;
                        if (bit_idx == 4'd8) begin
                            // ACK/NACK slot just finished.
                            if (!is_read) ack_error <= sda_i; // slave NACKed a write = sda stayed high
                            if (stop_after) begin
                                sda_reg <= 1'b0; sda_oe_reg <= 1'b1; // prep for STOP (SDA must be low while SCL low, then rise while SCL high)
                                state <= S_STOP_LOW;
                            end else begin
                                busy  <= 1'b0;
                                state <= S_HELD;
                            end
                        end else begin
                            bit_idx <= bit_idx + 4'd1;
                            // BUG FIX (caught via tb_i2c_debug.v's per-edge
                            // trace, not by inspection): the original code
                            // only had an explicit case for "entering the
                            // ACK slot of a READ byte" - entering the ACK
                            // slot of a WRITE byte (bit_idx 7->8, !is_read)
                            // fell through to the generic "drive next write
                            // bit" else branch, keeping the master driving
                            // SDA instead of releasing it for the SLAVE to
                            // ack. Confirmed in the trace: m_oe stayed 1
                            // through the whole ack slot, slave's own oe
                            // never got a chance to matter, ack_error read
                            // back 0 for the WRONG reason (never actually
                            // sampled the slave's ack at all).
                            if (bit_idx == 4'd7 && is_read) begin
                                // Entering the ACK slot of a READ byte: master
                                // drives the ack/nack bit (nack_after), after
                                // having released SDA for the slave during bits 0-7.
                                sda_reg <= nack_after; sda_oe_reg <= 1'b1;
                            end else if (bit_idx == 4'd7 && !is_read) begin
                                // Entering the ACK slot of a WRITE byte:
                                // release SDA - it's the SLAVE's turn to ack.
                                sda_oe_reg <= 1'b0;
                            end else if (is_read) begin
                                // Reading a data bit: release SDA, let the
                                // slave drive; sample happens below.
                                sda_oe_reg <= 1'b0;
                            end else begin
                                sda_reg <= shift_reg[6]; sda_oe_reg <= 1'b1;
                                shift_reg <= {shift_reg[6:0], 1'b0};
                            end
                            state <= S_BIT_LOW;
                        end
                        // Sample incoming data WHILE scl_reg was still 1 (the
                        // just-finished S_BIT_HIGH), for read bits 0-7 only.
                        if (is_read && bit_idx != 4'd8) begin
                            rxdata_reg <= {rxdata_reg[6:0], sda_i};
                        end
                    end
                end

                S_STOP_LOW: begin
                    if (!tick) begin
                        div_cnt <= div_cnt + 16'd1;
                    end else begin
                        div_cnt <= 16'd0;
                        scl_reg <= 1'b1; // SCL back up with SDA still low
                        state   <= S_STOP_HIGH1;
                    end
                end
                S_STOP_HIGH1: begin
                    if (!tick) begin
                        div_cnt <= div_cnt + 16'd1;
                    end else begin
                        div_cnt <= 16'd0;
                        sda_reg <= 1'b1; // STOP: SDA rises while SCL is high
                        state   <= S_STOP_HIGH2;
                    end
                end
                S_STOP_HIGH2: begin
                    if (!tick) begin
                        div_cnt <= div_cnt + 16'd1;
                    end else begin
                        div_cnt <= 16'd0;
                        busy    <= 1'b0;
                        state   <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    assign scl     = scl_reg;
    assign sda_o    = sda_reg;
    assign sda_oe   = sda_oe_reg;
endmodule
