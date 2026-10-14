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
    // missing.
    //
    // TRUE OPEN-DRAIN (§100 - the previous version was not). I2C has no
    // push-pull drivers at all: a device may only pull a line LOW or
    // release it, and the bus's external pull-up resistor provides every
    // high level. The earlier code set sda_oe=1 together with sda_reg=1
    // in six places, and drove SCL push-pull outright - on a real bus
    // that means the master actively fights any slave that is ACKing or
    // clock-stretching, i.e. a driver sourcing current shorted against
    // one sinking it. It only looked fine because the testbench's
    // wired-AND model resolved it in the master's favour.
    //
    // Both lines are now (o, oe, i) triples where `o` is hard-wired to 0
    // and `oe` is asserted ONLY to pull low:
    //     oe=1 -> pull low        oe=0 -> release (pull-up/slave wins)
    // so it is structurally impossible for this module to source a high.
    // synth_top.v turns each triple into one real bidirectional pad.
    //
    // scl_i additionally enables CLOCK STRETCHING: whenever this master
    // has released SCL but the line still reads low, a slave is holding
    // it, and the whole transfer FSM freezes until the slave lets go
    // (see `stretch_hold` below). The old version had no way to even
    // observe this - SCL was output-only.
    output        scl_o,
    output        scl_oe,
    input         scl_i,
    output        sda_o,
    output        sda_oe,
    input         sda_i
);
    // Register map:
    //   0xFF000040  CTRL   (W) bit0=START(pulse) bit1=STOP_AFTER bit2=IS_READ bit3=NACK_AFTER(read only)
    //   0xFF000044  CLKDIV (W) [15:0] HALF-bit-period divider: each of a
    //                               bit's two SCL phases lasts clkdiv+1 clocks,
    //                               so f_SCL = f_clk / (2*(clkdiv+1)). (The old
    //                               comment said "quarter"; the FSM has only two
    //                               phases per bit, so that was never right.)
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
            // §100: the reset default used to be 10, which at this SoC's
            // 12 MHz clock is 12e6 / (2 * (10+1)) = 545 kHz - more than
            // five times the 100 kHz the BrainBoard 1500's I2C bus is
            // specified for, so a board brought up without software
            // writing CLKDIV first would simply not communicate.
            // 59 gives 12e6 / (2 * (59+1)) = 100.0 kHz nominal; with the
            // clock-stretch check costing a couple of clocks per SCL-high
            // phase the real figure lands just UNDER 100 kHz, which is
            // the correct side of a maximum.
            clkdiv <= 16'd59;
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

    // §100: SDA and SCL are driven by other devices on a shared bus,
    // entirely asynchronously to this clock (and, on the planned board,
    // through a TXB0108 level translator), so both inputs can change
    // arbitrarily close to a clock edge. Sampling them directly - which
    // is what the ACK check, the read-data shift and the clock-stretch
    // check all did - risks latching a metastable value. Two flops in
    // series give the first a full clock period to resolve.
    //
    // The 2-cycle delay is fine for both uses:
    //   - data: the slave presents each bit on the preceding SCL
    //     falling edge, a full half-period (clkdiv+1 clocks, 60 at the
    //     reset default) before the master samples, so a 2-clock lag
    //     stays well inside the valid window.
    //   - clock stretch: the lag just makes the master wait 2 extra
    //     clocks before believing SCL has gone high, which lengthens
    //     the SCL high phase slightly - i.e. the bus runs a hair SLOWER
    //     than nominal, the safe direction for a maximum-rate spec.
    reg scl_meta, scl_sync, sda_meta, sda_sync;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // Reset to the idle (released, pulled-up) level, not 0 -
            // coming out of reset believing the bus is held low would
            // make the master think a slave is stretching the clock.
            scl_meta <= 1'b1; scl_sync <= 1'b1;
            sda_meta <= 1'b1; sda_sync <= 1'b1;
        end else begin
            scl_meta <= scl_i;  scl_sync <= scl_meta;
            sda_meta <= sda_i;  sda_sync <= sda_meta;
        end
    end

    // §100: clock stretching. `scl_reg` is now only an INTENT ("release
    // the line" vs "pull it low"); what the bus actually does is scl_i.
    // If this master has released SCL (scl_reg=1) but the line still
    // reads low while a transfer is in progress, some slave is holding
    // it down to buy time - the I2C-specified way for a slow device to
    // stall the master. Freezing the entire transfer FSM (including
    // div_cnt, so the half-period restarts cleanly once the line is
    // free) is exactly the required response. Qualified by `busy` so an
    // idle master is never frozen and can still latch a START request.
    wire stretch_hold = busy && scl_reg && !scl_sync;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE; busy <= 1'b0; ack_error <= 1'b0;
            scl_reg <= 1'b1; sda_reg <= 1'b1; sda_oe_reg <= 1'b1;
            div_cnt <= 16'd0; bit_idx <= 4'd0; shift_reg <= 8'd0; rxdata_reg <= 8'd0;
        end else if (!stretch_hold) begin
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
                            if (!is_read) ack_error <= sda_sync; // slave NACKed a write = sda stayed high
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
                            rxdata_reg <= {rxdata_reg[6:0], sda_sync};
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

    // §100: the only place the open-drain rule has to be enforced, and
    // the reason the whole FSM above could stay untouched. `sda_reg` and
    // `scl_reg` keep their old meaning as the INTENDED line level; what
    // changes is how that intent reaches the pin.
    //
    //   intent 0  -> pull the line low   (oe=1, o=0)
    //   intent 1  -> release the line    (oe=0; the pull-up makes it 1)
    //   not ours  -> release the line    (sda_oe_reg=0, e.g. the slave's
    //                                     ACK slot or a read byte)
    //
    // Driving a hard 0 on `o` and gating only `oe` makes "drive high"
    // unrepresentable rather than merely unused, so no future edit to
    // the FSM can reintroduce the bug. SCL has no release/own
    // distinction - a master always owns the clock - so its oe is just
    // the inverted intent.
    assign scl_o   = 1'b0;
    assign scl_oe  = ~scl_reg;
    assign sda_o   = 1'b0;
    assign sda_oe  = sda_oe_reg && !sda_reg;
endmodule
