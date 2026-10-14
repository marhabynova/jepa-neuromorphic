module spi_master (
    input         clk,
    input         rst,

    // Generic peripheral bus from cpu.v (see cpu.v's port comment) - this
    // module decodes addr[7:0] itself, cpu.v knows nothing about these
    // specific registers.
    input         bus_we,
    input         bus_re,
    input  [31:0] bus_addr,
    input  [31:0] bus_wdata,
    output [31:0] bus_rdata,

    // Physical SPI pins, to be wired straight to top-level FPGA pads in
    // soc_top.v / the board constraint file.
    output        spi_sck,
    output        spi_mosi,
    input         spi_miso,
    output        spi_cs_n,

    // §96: pulses for exactly one cycle when a transfer finishes (busy
    // 1->0) - lets the CPU wait for "Akida done" via an interrupt instead
    // of busy-polling STATUS, the whole point of adding interrupts to
    // cpu.v in the first place.
    output        irq_out
);
    // Register map (word addresses, byte offsets within the 0xFF-prefixed
    // MMIO page - 0x00/0x?? reserved for cpu.v's own single-register debug
    // port, kept separate on purpose):
    //   0xFF000010  CTRL   (W) bit0=START(pulse) bit1=CS_FORCE bit3:2=MODE(CPOL,CPHA)
    //   0xFF000014  CLKDIV (W) [15:0] half-period divider (see sck_tick below)
    //   0xFF000018  TXDATA (W) [7:0]  byte to send - write THEN pulse START
    //   0xFF00001C  STATUS (R) bit0=BUSY
    //   0xFF000020  RXDATA (R) [7:0]  byte received by the most recently
    //                               completed transfer
    localparam ADDR_CTRL   = 8'h10;
    localparam ADDR_CLKDIV = 8'h14;
    localparam ADDR_TXDATA = 8'h18;
    localparam ADDR_STATUS = 8'h1C;
    localparam ADDR_RXDATA = 8'h20;

    reg        cpol, cpha, cs_force;
    reg [15:0] clkdiv;
    reg [7:0]  txdata_reg;
    reg [7:0]  rxdata_reg;
    reg        busy;

    wire sel_ctrl   = bus_addr[7:0] == ADDR_CTRL;
    wire sel_clkdiv = bus_addr[7:0] == ADDR_CLKDIV;
    wire sel_txdata = bus_addr[7:0] == ADDR_TXDATA;
    wire sel_status = bus_addr[7:0] == ADDR_STATUS;
    wire sel_rxdata = bus_addr[7:0] == ADDR_RXDATA;

    wire start_pulse = bus_we && sel_ctrl && bus_wdata[0] && !busy;

    // --- Register writes ---------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cpol <= 1'b0; cpha <= 1'b0; cs_force <= 1'b0; clkdiv <= 16'd4;
        end else if (bus_we) begin
            if (sel_ctrl) begin
                cs_force <= bus_wdata[1];
                cpol     <= bus_wdata[2];
                cpha     <= bus_wdata[3];
            end else if (sel_clkdiv) begin
                clkdiv <= bus_wdata[15:0];
            end else if (sel_txdata) begin
                txdata_reg <= bus_wdata[7:0];
            end
        end
    end

    // --- Synchronous read, SAME convention as dmem.v (unconditional every
    // cycle, no explicit re-gating needed since every field here is
    // idempotent to read repeatedly) ----------------------------------------
    reg [31:0] rd_data;
    always @(posedge clk) begin
        if (sel_status)      rd_data <= {31'd0, busy};
        else if (sel_rxdata) rd_data <= {24'd0, rxdata_reg};
        else                 rd_data <= 32'd0;
    end
    assign bus_rdata = rd_data;

    // --- SCK generation: a free-running half-period tick from clkdiv -------
    reg [15:0] div_cnt;
    reg        sck_reg;
    wire       tick = (div_cnt == clkdiv);

    // --- Transfer FSM --------------------------------------------------------
    // Classic edge-based shift register: on every `tick` while busy, sck
    // toggles. Which toggle direction is the SHIFT edge (present next MOSI
    // bit) vs the SAMPLE edge (latch MISO) depends on CPHA - see the
    // leading/trailing assignment below, standard SPI timing definition.
    // CPHA=0's first bit is a special case (must be valid BEFORE the first
    // leading edge, since there's no preceding trailing edge yet) - handled
    // explicitly in the START branch instead of waiting for a tick.
    reg [7:0] shift_reg;
    reg       first_shift;
    reg [3:0] bits_left;
    reg       cs_n_reg;
    // See the A5 fix below: CPHA=0's last sample lands on a LEADING edge,
    // one half-period before sck would naturally return to cpol - this
    // defers ending the transfer by exactly one more tick (the natural
    // trailing edge) instead of forcing sck_reg out-of-band.
    reg       finishing;

    // §100: spi_miso is driven by an external device off ITS own clock
    // domain (and, on the planned board, through a TXB0108 level
    // translator), so it can change arbitrarily close to a clock edge
    // here. Sampling it straight into rxdata_reg could latch a
    // metastable value. Two flops in series give the first a full clock
    // period to resolve. The 2-cycle delay is harmless because the
    // slave presents each bit a half SCK period (clkdiv+1 clocks) ahead
    // of the sampling edge - at the default clkdiv this is many clocks
    // of margin, and even the fastest setting used in simulation
    // (clkdiv=2, 3 clocks per half period) still samples the bit the
    // slave drove on the previous edge.
    reg miso_meta, miso_sync;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            miso_meta <= 1'b0;
            miso_sync <= 1'b0;
        end else begin
            miso_meta <= spi_miso;
            miso_sync <= miso_meta;
        end
    end

    wire next_sck = sck_reg ^ (tick && busy);
    wire leading_edge  = cpol ? (sck_reg == 1'b1 && !next_sck) : (sck_reg == 1'b0 && next_sck);
    wire trailing_edge = cpol ? (sck_reg == 1'b0 && next_sck) : (sck_reg == 1'b1 && !next_sck);

    wire shift_edge  = busy && tick && (cpha ? leading_edge : trailing_edge);
    wire sample_edge = busy && tick && (cpha ? trailing_edge : leading_edge);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            busy <= 1'b0; div_cnt <= 16'd0; sck_reg <= 1'b0; cs_n_reg <= 1'b1;
            shift_reg <= 8'd0; bits_left <= 4'd0; rxdata_reg <= 8'd0; first_shift <= 1'b0;
            finishing <= 1'b0;
        end else begin
            if (start_pulse) begin
                busy        <= 1'b1;
                cs_n_reg    <= 1'b0;
                div_cnt     <= 16'd0;
                bits_left   <= 4'd8;
                first_shift <= 1'b1;
                // CPHA=0: the MSB must be valid BEFORE the first (leading)
                // edge, since that edge samples it - pre-loading here is
                // correct and sufficient, no special-casing needed below.
                // CPHA=1: the MSB must become valid AT the first (leading)
                // edge, not before - pre-loading it here is still harmless
                // (MOSI just shows it a bit early, before CS even cares),
                // but the first shift_edge (also a leading edge for CPHA=1)
                // must NOT then rotate past it before it's been sampled on
                // the following trailing edge - that's what first_shift
                // guards below.
                shift_reg <= txdata_reg;
                // BUG FIX (Opus audit A4, caught before simulation by
                // re-reading this against the CTRL write-handler above):
                // `cpol` is a REGISTER updated by the OTHER always block
                // in this SAME clock edge (sel_ctrl's bus_we branch) - if
                // software writes mode bits and START together in one
                // CTRL write (which the register map invites, since
                // bit0=START and bits3:2=MODE share a register), this
                // line would read the STALE pre-write cpol, parking sck
                // at the wrong idle level for the mode actually being
                // requested. Fix: read bus_wdata[2] directly, matching
                // what cpol is about to become - consistent since the
                // write handler latches that same bit into cpol this
                // same cycle (same pattern as i2c_master.v's txdata_reg
                // stale-read fix).
                sck_reg   <= bus_wdata[2];
            end else if (busy) begin
                // BUG FIXED: an earlier draft had an unconditional
                // `sck_reg <= cpol` default sitting OUTSIDE this if/else,
                // which snapped sck_reg back to cpol on every non-tick
                // cycle - it never actually HELD a toggled level for a
                // full half-period, so leading/trailing edges fired almost
                // every cycle instead of once per clkdiv+1 cycles (caught
                // via per-cycle simulation trace, not by inspection - the
                // garbage shift_reg/rxdata_reg values made it obvious).
                // Fix: only ASSIGN sck_reg on a tick (toggle); on a
                // non-tick cycle, leaving it unassigned here means it
                // correctly HOLDS via ordinary flip-flop memory.
                if (tick) begin
                    div_cnt <= 16'd0;
                    sck_reg <= ~sck_reg;
                    // The deferred end-of-transfer from CPHA=0's last
                    // sample (see below) - this tick's toggle IS the
                    // natural trailing edge that returns sck to cpol, so
                    // nothing needs to be forced; just let busy drop now
                    // that the real edge has actually happened.
                    if (finishing) begin
                        busy      <= 1'b0;
                        cs_n_reg  <= cs_force ? 1'b0 : 1'b1;
                        finishing <= 1'b0;
                    end
                end else begin
                    div_cnt <= div_cnt + 16'd1;
                end

                // BUG FIXED: an earlier draft tied end-of-transfer/bits_left
                // to shift_edge, which has an off-by-one for CPHA=1 - the
                // first bit is pre-loaded onto MOSI before any clock edge
                // (so it can be valid in time for CPHA=0's leading-edge
                // sample), but for CPHA=1 the first LEADING edge IS a
                // shift_edge, so the old code immediately rotated to bit2
                // before bit1 had even been sampled on the following
                // trailing edge - silently dropping a bit. Caught via
                // simulation (CPHA=1 cases failed a loopback self-test,
                // CPHA=0 passed), not by inspection. Fix: tie bits_left/
                // end-of-transfer to sample_edge instead - sample_edge
                // fires exactly 8 times per transfer regardless of CPHA
                // (unlike shift_edge, which has the pre-load asymmetry),
                // so it's the correct event to count 8 bits against.
                // shift_edge now just unconditionally rotates while busy;
                // one harmless extra rotation after the last useful bit
                // doesn't matter since nothing reads shift_reg once busy
                // drops.
                if (sample_edge) begin
                    rxdata_reg <= {rxdata_reg[6:0], miso_sync};
                    if (bits_left == 4'd1) begin
                        // BUG FIX (Opus audit A5, caught by an
                        // INDEPENDENT slave model, tb/spi_slave_model.v -
                        // a self-consistent loopback test could not have
                        // caught this, see below): for CPHA=0, the last
                        // sample lands on a LEADING edge - half a clock
                        // period before SCK would naturally return to
                        // cpol on the following TRAILING edge. An
                        // earlier draft of this fix ended the transfer
                        // (busy<=0) on THIS same cycle and forced
                        // `sck_reg <= cpol` directly, reasoning that
                        // would at least park SCK at the right level.
                        // That's wrong in a worse way: it OVERWROTE the
                        // very toggle (`sck_reg <= ~sck_reg` above, same
                        // always-block invocation) that makes this edge
                        // the Nth sample in the first place - the
                        // physical SCK pin never actually transitioned
                        // for this bit at all. A self-consistent loopback
                        // test (tb_spi.v) couldn't catch it, because the
                        // master's own sample_edge/shift_edge wires are
                        // computed from `next_sck` (a separate wire,
                        // XOR-derived from tick&&busy), not from the
                        // possibly-overridden sck_reg - so RXDATA still
                        // came back correct from the master's own
                        // internal perspective even though the physical
                        // pin lied. tb/spi_slave_model.v watches the
                        // actual pin and caught it directly: bit_cnt
                        // stalled at 7, last_rx never latched.
                        // Fix: don't touch sck_reg here at all - let
                        // THIS tick's natural toggle (the real Nth
                        // leading edge) happen untouched, and defer
                        // ending the transfer to the FOLLOWING tick (see
                        // the `if (finishing)` branch above), which is
                        // the natural trailing edge that returns sck to
                        // cpol on its own.
                        //
                        // SEPARATE BUG FOUND THE SAME WAY (not yet in the
                        // Opus audit's A1-A8 list - found by this
                        // independent-slave-model work itself, §99's own
                        // point): the ORIGINAL code ended immediately for
                        // CPHA=1 too (busy<=0; cs_n_reg<=... in this same
                        // cycle as the final sample). That looked safe in
                        // isolation (SCK does land back at cpol correctly
                        // for CPHA=1), but cs_n_reg and sck_reg are BOTH
                        // driven by nonblocking assignment from this SAME
                        // always block/clock edge - meaning they update
                        // atomically in the same simulation delta, with
                        // ZERO hold time between the final sample edge
                        // and CS deasserting. tb_spi.v's loopback test
                        // couldn't catch this (the master's own RXDATA
                        // capture never looks at the cs_n pin's value at
                        // all), but tb/spi_slave_model.v's independent
                        // sample logic gates on `!cs_n` at the moment of
                        // the clock edge, and Verilog's nonblocking
                        // update semantics mean it sees cs_n ALREADY
                        // deasserted by the time it evaluates that same
                        // edge - the 8th bit's sample was silently
                        // skipped (bit_cnt stalled at 7, exactly as
                        // observed in simulation). This is also a real
                        // hardware concern, not just a simulation
                        // artifact: a CS line with zero hold time past
                        // the last clock edge is a genuine setup/hold
                        // violation for any real external slave. Fix:
                        // defer ending uniformly for BOTH cpha values via
                        // the same `finishing` mechanism - CPHA=1 just
                        // tolerates one extra (harmless, per the
                        // shift_edge comment above) half-period of
                        // clocking before CS actually drops, giving every
                        // external slave real hold time on the last bit.
                        finishing <= 1'b1;
                        bits_left <= 4'd0;
                    end else begin
                        bits_left <= bits_left - 4'd1;
                    end
                end
                if (shift_edge) begin
                    // CPHA=1's first shift_edge is the leading edge that
                    // merely REVEALS the pre-loaded MSB (which the
                    // following trailing edge will sample) - it must not
                    // rotate yet, or the MSB gets skipped before anything
                    // ever samples it (confirmed by simulation: without
                    // this guard, CPHA=1 transfers came back shifted left
                    // by exactly one bit, e.g. sent 0x7E got 0xFC = 0x7E<<1).
                    // CPHA=0 never takes this branch (first_shift stays
                    // irrelevant since its shift_edge is a trailing edge
                    // that always legitimately follows a completed sample).
                    if (cpha && first_shift) begin
                        first_shift <= 1'b0;
                    end else begin
                        shift_reg   <= {shift_reg[6:0], 1'b0};
                        first_shift <= 1'b0;
                    end
                end
            end else begin
                cs_n_reg <= cs_force ? 1'b0 : 1'b1;
            end
        end
    end

    assign spi_sck  = sck_reg;
    assign spi_mosi = shift_reg[7];
    assign spi_cs_n = cs_n_reg;

    // One-cycle-delayed pulse (fires the cycle right after busy actually
    // drops to 0, avoiding any same-cycle ambiguity with the condition
    // that clears busy) - matches cpu.v's expectation of a short pulse per
    // completed transfer, which it latches internally so it's never missed.
    reg irq_pulse;
    always @(posedge clk or posedge rst) begin
        if (rst) irq_pulse <= 1'b0;
        else     irq_pulse <= (busy && sample_edge && bits_left == 4'd1);
    end
    assign irq_out = irq_pulse;
endmodule
