// Independent behavioural I2C slave, written from the I2C timing
// definition directly (START=SDA-falls-while-SCL-high, bits sampled on
// SCL's rising edge, slave outputs prepared on SCL's falling edge,
// STOP=SDA-rises-while-SCL-high) - NOT derived from reading
// i2c_master.v's internal state, so it can catch master bugs a
// self-consistent loopback structurally cannot.
module i2c_slave_model #(
    parameter [6:0] MY_ADDR = 7'h43
) (
    input        rst,        // BUG FOUND DURING BRING-UP (not in i2c_master.v -
                              // in THIS model): `scl` is X at simulation start,
                              // and Verilog counts an X->1 transition as a
                              // posedge - without an explicit reset tied to the
                              // same timeline as the rest of the testbench, that
                              // spurious pre-reset "posedge" consumed one bit
                              // slot, shifting every subsequent bit count by one
                              // and firing the address-ack transition a full
                              // bit early. Caught via per-edge internal $display
                              // tracing, not by inspection.
    input        scl,
    input        sda_i,     // bus state as driven by everyone else (the master)
    output       sda_o,      // always 0 - open-drain, only meaningful when sda_oe=1
    output       sda_oe,     // 1 = this slave is pulling SDA low right now
    output [7:0] last_rx,
    output       rx_valid,   // pulses for one $monitor-visible delta after a write byte+ack
    input  [7:0] tx_byte     // byte this slave sends back when addressed for a read
);
    localparam ST_WAIT_ADDR = 0, ST_ADDR_ACK = 1, ST_DATA = 2, ST_DATA_ACK = 3;

    reg [1:0]  st;
    reg [2:0]  bit_cnt;      // 0..7 within the current byte
    reg [7:0]  shift_in;
    reg        addressed, is_read;
    reg        drive_low;    // what THIS slave wants to pull low right now
    reg [7:0]  last_rx_r;
    reg        rx_valid_r;
    reg        scl_prev, sda_prev;

    wire start_cond = scl && scl_prev && sda_prev && !sda_i;
    wire stop_cond  = scl && scl_prev && !sda_prev && sda_i;
    // Can't index a concatenation expression directly in this Verilog
    // dialect ({a,b}[x:y] is a syntax error in iverilog) - a named wire
    // avoids repeating {shift_in[6:0],sda_i} everywhere it's indexed.
    wire [7:0] next_shift = {shift_in[6:0], sda_i};

    initial begin
        st = ST_WAIT_ADDR; bit_cnt = 0; shift_in = 0; addressed = 0; is_read = 0;
        drive_low = 0; last_rx_r = 0; rx_valid_r = 0; scl_prev = 1; sda_prev = 1;
    end

    // START/STOP detection - level-sensitive, independent of the clocked
    // bit logic below (a real slave's START/STOP detector works the same
    // way: watch SDA transitions while SCL is high, regardless of what
    // bit-phase the rest of the state machine thinks it's in).
    always @(scl or sda_i or rst) begin
        if (rst) begin
            // Takes priority over start_cond/stop_cond regardless of
            // simulation-ordering between always blocks (unlike a separate
            // always@(posedge rst) block would) - directly in the same
            // block that reacts to scl/sda transitions, so there's no
            // race window where a spurious pre-reset edge could be
            // processed before this clears it.
            st <= ST_WAIT_ADDR; bit_cnt <= 0; shift_in <= 0; addressed <= 0; is_read <= 0;
            drive_low <= 0; scl_prev <= 1; sda_prev <= 1;
        end else if (start_cond) begin
            st <= ST_WAIT_ADDR; bit_cnt <= 0; shift_in <= 0; addressed <= 0; drive_low <= 0;
            scl_prev <= scl; sda_prev <= sda_i;
        end else if (stop_cond) begin
            st <= ST_WAIT_ADDR; bit_cnt <= 0; drive_low <= 0;
            scl_prev <= scl; sda_prev <= sda_i;
        end else begin
            scl_prev <= scl; sda_prev <= sda_i;
        end
    end

    // Sample on SCL's rising edge (bit value is valid by then). Guarded by
    // `rst` directly (not just a separate always block) for the same
    // reason as above - a spurious posedge on `scl` while still in reset
    // must not advance bit_cnt/state.
    always @(posedge scl or posedge rst) begin
      if (rst) begin
        bit_cnt <= 0; st <= ST_WAIT_ADDR; rx_valid_r <= 1'b0;
      end else begin
        rx_valid_r <= 1'b0;
        case (st)
            ST_WAIT_ADDR: begin
                shift_in <= next_shift;
                if (bit_cnt == 3'd7) begin
                    addressed <= (next_shift[7:1] == MY_ADDR);
                    is_read   <= next_shift[0];
                    st <= ST_ADDR_ACK;
                end
                bit_cnt <= bit_cnt + 3'd1;
            end
            ST_ADDR_ACK: begin
                st <= ST_DATA; bit_cnt <= 0;
            end
            ST_DATA: begin
                if (!is_read) begin
                    shift_in <= next_shift;
                    if (bit_cnt == 3'd7) begin
                        last_rx_r  <= next_shift;
                        rx_valid_r <= 1'b1;
                    end
                end
                bit_cnt <= bit_cnt + 3'd1;
                if (bit_cnt == 3'd7) st <= ST_DATA_ACK;
            end
            ST_DATA_ACK: begin
                // For a read, this clock also samples the MASTER's ack/nack
                // (not acted on by this simple model - a real slave would
                // stop sending more bytes on a master NACK; this model
                // only ever sends one byte per address phase anyway).
                st <= ST_DATA; bit_cnt <= 0;
            end
        endcase
      end
    end

    // Prepare the NEXT thing this slave drives on SCL's falling edge -
    // standard I2C convention (outputs change while the clock is low).
    // Same rst-guard reasoning as the posedge block above.
    always @(negedge scl or posedge rst) begin
      if (rst) begin
        drive_low <= 1'b0;
      end else begin
        case (st)
            ST_ADDR_ACK: drive_low <= addressed; // ACK (pull low) only if address matched
            ST_DATA:     drive_low <= is_read && addressed ? ~tx_byte[3'd7 - bit_cnt] : 1'b0;
            default:     drive_low <= 1'b0;
        endcase
      end
    end

    assign sda_oe   = drive_low;
    assign sda_o    = 1'b0;
    assign last_rx  = last_rx_r;
    assign rx_valid = rx_valid_r;
endmodule
