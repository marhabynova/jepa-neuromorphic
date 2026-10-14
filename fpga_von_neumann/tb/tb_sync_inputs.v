`timescale 1ns/1ps

// §100: clock-domain-crossing conformance test.
//
// What a 2-flop synchronizer must do, stated independently of how it is
// written:
//   (a) The value the design uses must be the input as it was TWO clock
//       edges ago - not one, and not combinational. One flop is not a
//       synchronizer: it can still go metastable, and because its
//       output fans out, different downstream gates can resolve the
//       same metastable bit differently in the same cycle.
//   (b) A pulse on the input that is shorter than a clock period and
//       falls entirely between two edges must never reach the design.
//       (A real synchronizer may legitimately swallow such a pulse;
//       what it must NOT do is let it through combinationally.)
//   (c) The input must be allowed to change at ANY time, including
//       right next to a clock edge, without the design seeing a value
//       that was never on the wire.
//
// Reset release has its own rule, from the same body of practice:
//   (d) Asserting reset must be asynchronous - it has to work with no
//       clock at all.
//   (e) RELEASING reset must be synchronous - every flop in the design
//       must leave reset on the same clock edge, otherwise a state
//       machine can start from a state its reset values never describe.
//
// Simulation cannot model metastability itself, so what is actually
// checkable is the STRUCTURE that prevents it: that the design's view
// of each asynchronous input really is two registered stages behind the
// pin. That is exactly what was missing before - spi_miso, i2c sda_i,
// i2c scl_i and gpio_in all went straight into logic.
module tb_sync_inputs;
    reg clk, rst;
    integer errors;

    always #5 clk = ~clk;

    // ---------------- device under test: gpio ------------------------
    reg  [7:0] gpio_in;
    wire [7:0] gpio_out_w;
    wire [31:0] gpio_rdata;
    gpio gpio_u (
        .clk(clk), .rst(rst), .bus_we(1'b0), .bus_re(1'b0),
        .bus_addr(32'h0), .bus_wdata(32'h0), .bus_rdata(gpio_rdata),
        .gpio_out(gpio_out_w), .gpio_in(gpio_in)
    );

    // ---------------- device under test: spi_master ------------------
    reg miso;
    wire spi_sck, spi_mosi, spi_cs_n, spi_irq;
    wire [31:0] spi_rdata;
    spi_master spi_u (
        .clk(clk), .rst(rst), .bus_we(1'b0), .bus_re(1'b0),
        .bus_addr(32'h0), .bus_wdata(32'h0), .bus_rdata(spi_rdata),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(miso),
        .spi_cs_n(spi_cs_n), .irq_out(spi_irq)
    );

    // ---------------- device under test: i2c_master ------------------
    reg sda_line, scl_line;
    wire i2c_scl_o, i2c_scl_oe, i2c_sda_o, i2c_sda_oe;
    wire [31:0] i2c_rdata;
    i2c_master i2c_u (
        .clk(clk), .rst(rst), .bus_we(1'b0), .bus_re(1'b0),
        .bus_addr(32'h0), .bus_wdata(32'h0), .bus_rdata(i2c_rdata),
        .scl_o(i2c_scl_o), .scl_oe(i2c_scl_oe), .scl_i(scl_line),
        .sda_o(i2c_sda_o), .sda_oe(i2c_sda_oe), .sda_i(sda_line)
    );

    // ---------------- device under test: reset_sync ------------------
    reg  ext_rst;
    wire rst_out;
    reset_sync rsync (.clk(clk), .rst_in(ext_rst), .rst_out(rst_out));

    task chk(input [255:0] name, input [31:0] got, input [31:0] expect);
        begin
            if (got !== expect) begin
                $display("  MISMATCH %0s: got %h, expected %h", name, got, expect);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        clk = 0; rst = 1; errors = 0;
        gpio_in = 8'h00; miso = 1'b0; sda_line = 1'b1; scl_line = 1'b1;
        ext_rst = 1'b1;
        #12 rst = 0; ext_rst = 0;
        @(posedge clk); @(posedge clk); @(posedge clk);

        // ===== (a) two registered stages, not one, not combinational ==
        // Change every asynchronous input at a deliberately awkward
        // moment - 1 ns after a clock edge, i.e. nowhere near the next
        // one - then step edge by edge and watch the value march
        // through the chain.
        @(posedge clk); #1;
        gpio_in  = 8'hA5;
        miso     = 1'b1;
        sda_line = 1'b0;
        scl_line = 1'b0;

        // Still nothing: the value has not even reached the first flop.
        chk("gpio stage2 before any edge", gpio_u.gpio_in_sync, 8'h00);
        chk("spi  stage2 before any edge", spi_u.miso_sync,     1'b0);
        chk("i2c  sda stage2 before edge", i2c_u.sda_sync,      1'b1);
        chk("i2c  scl stage2 before edge", i2c_u.scl_sync,      1'b1);

        @(posedge clk); #1;
        // One edge: first stage has it, the design must NOT yet.
        chk("gpio stage1 after 1 edge", gpio_u.gpio_in_meta, 8'hA5);
        chk("gpio stage2 after 1 edge", gpio_u.gpio_in_sync, 8'h00);
        chk("spi  stage1 after 1 edge", spi_u.miso_meta,     1'b1);
        chk("spi  stage2 after 1 edge", spi_u.miso_sync,     1'b0);
        chk("i2c  sda stage1 after 1 edge", i2c_u.sda_meta,  1'b0);
        chk("i2c  sda stage2 after 1 edge", i2c_u.sda_sync,  1'b1);
        chk("i2c  scl stage1 after 1 edge", i2c_u.scl_meta,  1'b0);
        chk("i2c  scl stage2 after 1 edge", i2c_u.scl_sync,  1'b1);

        @(posedge clk); #1;
        // Two edges: now, and only now, the design sees it.
        chk("gpio stage2 after 2 edges", gpio_u.gpio_in_sync, 8'hA5);
        chk("spi  stage2 after 2 edges", spi_u.miso_sync,     1'b1);
        chk("i2c  sda stage2 after 2 edges", i2c_u.sda_sync,  1'b0);
        chk("i2c  scl stage2 after 2 edges", i2c_u.scl_sync,  1'b0);

        // ===== (b) a sub-clock-period glitch must not reach the design
        // A 2 ns pulse on a 10 ns clock, placed entirely between edges.
        @(posedge clk); #1;
        gpio_in = 8'h00;
        @(posedge clk); @(posedge clk); #1;   // let 0x00 settle through
        chk("gpio settled to 00", gpio_u.gpio_in_sync, 8'h00);
        #2 gpio_in = 8'hFF;                   // glitch starts mid-period
        #2 gpio_in = 8'h00;                   // and ends before the edge
        @(posedge clk); @(posedge clk); #1;
        chk("glitch did not reach the design", gpio_u.gpio_in_sync, 8'h00);

        // ===== (c) change RIGHT at a clock edge ======================
        // The whole reason synchronizers exist. The design must end up
        // with a clean 0 or 1 - never X - two edges later.
        @(posedge clk);
        gpio_in = 8'h3C;                      // zero-delay, same instant
        @(posedge clk); @(posedge clk); #1;
        if (gpio_u.gpio_in_sync !== 8'h3C && gpio_u.gpio_in_sync !== 8'h00) begin
            $display("  MISMATCH edge-aligned change produced %h, neither old nor new value",
                     gpio_u.gpio_in_sync);
            errors = errors + 1;
        end
        if (^gpio_u.gpio_in_sync === 1'bx) begin
            $display("  MISMATCH edge-aligned change left X in the design");
            errors = errors + 1;
        end

        // ===== (d) reset ASSERT is asynchronous ======================
        // Raise it between clock edges and require it to take effect
        // immediately, with no edge in between.
        @(posedge clk); #3;
        ext_rst = 1'b1;
        #1;
        chk("reset asserted asynchronously", rst_out, 1'b1);

        // ===== (e) reset RELEASE is synchronous ======================
        // Drop the pin between edges: rst_out must stay asserted until
        // clock edges push the release through, and must then fall ON
        // an edge (checked by sampling just before and just after).
        @(posedge clk); #3;
        ext_rst = 1'b0;
        #1;
        chk("reset still held right after async release", rst_out, 1'b1);
        @(posedge clk); #1;
        chk("reset still held 1 edge after release", rst_out, 1'b1);
        @(posedge clk); #1;
        chk("reset released 2 edges after release", rst_out, 1'b0);

        if (errors == 0)
            $display("PASS: gpio_in, spi_miso, i2c sda_i/scl_i all cross into the design through two registered stages, and reset asserts asynchronously but releases synchronously.");
        else
            $display("FAIL: %0d problem(s), see above.", errors);
        $finish;
    end
endmodule
