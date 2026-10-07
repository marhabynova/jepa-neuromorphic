// Independent behavioural SPI slave, written from the SPI timing
// definition directly (CPOL = idle clock level, CPHA = whether the first
// bit must be valid before the first clock edge (CPHA=0) or only becomes
// valid AT it (CPHA=1); data sampled on one edge, shifted/prepared on the
// other, alternating every half-period) - NOT derived from reading
// spi_master.v's internals, so it can catch master bugs a self-consistent
// loopback structurally cannot (see i2c_slave_model.v, same methodology,
// and RESEARCH_NOTES.md §99's whole reason for building this: the real
// A4/A5 bugs in spi_master.v were specifically bugs loopback couldn't see).
//
// CPOL/CPHA are parameters (not sensed from the bus) because SPI has no
// in-band mode negotiation - both sides must already agree on the
// physical-layer configuration before any transfer starts, unlike I2C's
// address byte.
module spi_slave_model #(
    parameter CPOL = 0,
    parameter CPHA = 0
) (
    input        sck,
    input        mosi,
    input        cs_n,
    output       miso,
    input  [7:0] tx_byte,
    output [7:0] last_rx,
    output       rx_valid
);
    // Which physical edge (posedge/negedge of sck) is the SAMPLE edge is
    // determined purely by CPOL×CPHA, independent of idle level: for
    // CPOL==CPHA (modes 0 and 3) the sample edge is sck's rising edge;
    // for CPOL!=CPHA (modes 1 and 2) it's the falling edge. Derived
    // directly from the mode table, not copied from any existing module.
    localparam SAMPLE_ON_POSEDGE = (CPOL == CPHA);

    reg [7:0] shift_in;
    reg [7:0] shift_out;
    reg [3:0] bit_cnt;
    reg [7:0] last_rx_r;
    reg       rx_valid_r;
    // Same subtlety as spi_master.v's own first_shift guard: for CPHA=1,
    // the FIRST shift-type edge merely REVEALS the MSB that was already
    // preloaded at CS-fall (continuously driven on MISO since then) - it
    // must not rotate the register yet, or bit1 never gets sampled
    // (confirmed necessary by deriving the mode-1/mode-3 timing by hand,
    // the same way spi_master.v's own comment describes discovering it
    // via simulation - this model independently needs the identical fix
    // because it's an inherent property of CPHA=1 timing, not an
    // implementation detail of the master).
    reg       first_shift;

    assign miso     = cs_n ? 1'b0 : shift_out[7];
    assign last_rx  = last_rx_r;
    assign rx_valid = rx_valid_r;

    always @(negedge cs_n) begin
        bit_cnt     <= 4'd0;
        shift_out   <= tx_byte;
        first_shift <= 1'b1;
    end

    always @(posedge cs_n) begin
        rx_valid_r <= 1'b0;
    end

    always @(posedge sck) begin
        if (!cs_n) begin
            if (SAMPLE_ON_POSEDGE) begin
                shift_in <= {shift_in[6:0], mosi};
                if (bit_cnt == 4'd7) begin
                    last_rx_r  <= {shift_in[6:0], mosi};
                    rx_valid_r <= 1'b1;
                end
                bit_cnt <= bit_cnt + 4'd1;
            end else begin
                // BUG FIX (caught via per-edge trace, not by inspection):
                // this guard must only suppress the FIRST shift for
                // CPHA=1 (where the shift edge coincides with the
                // leading edge that merely reveals the preloaded MSB -
                // same reasoning as spi_master.v's own first_shift
                // guard). For CPHA=0, the shift edge is the TRAILING
                // edge, and even the very first one legitimately needs
                // to rotate (to prepare bit6 for the 2nd sample) - the
                // first bit was already made valid by the CS-fall
                // preload alone, nothing "reveals" it here. Originally
                // missing the `CPHA &&` qualifier (copied the shape of
                // spi_master.v's guard but not the full condition)
                // silently ate CPHA=0's second bit on every transfer -
                // slave->master byte came back corrupted (e.g. sent
                // 0x3C, master received 0x1e = 0x3C with bit6 dropped).
                if (CPHA && first_shift) begin
                    first_shift <= 1'b0;
                end else begin
                    shift_out   <= {shift_out[6:0], 1'b0};
                    first_shift <= 1'b0;
                end
            end
        end
    end

    always @(negedge sck) begin
        if (!cs_n) begin
            if (!SAMPLE_ON_POSEDGE) begin
                shift_in <= {shift_in[6:0], mosi};
                if (bit_cnt == 4'd7) begin
                    last_rx_r  <= {shift_in[6:0], mosi};
                    rx_valid_r <= 1'b1;
                end
                bit_cnt <= bit_cnt + 4'd1;
            end else begin
                if (CPHA && first_shift) begin
                    first_shift <= 1'b0;
                end else begin
                    shift_out   <= {shift_out[6:0], 1'b0};
                    first_shift <= 1'b0;
                end
            end
        end
    end
endmodule
