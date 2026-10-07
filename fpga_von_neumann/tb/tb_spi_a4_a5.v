`timescale 1ns/1ps

// Regression for Opus audit A4/A5 (RESEARCH_NOTES.md §99), both in
// spi_master.v. Deliberately does NOT use tb_spi.v's do_transfer() helper,
// which always writes mode separately before START even when testing the
// "same write" case - that accidentally made the stale-cpol bug (A4)
// unobservable (the stale value already matched). This test writes mode+
// START packed into ONE CTRL write, with the mode actually CHANGING from
// the previous transfer, which is the only way to tell stale-cpol apart
// from fresh-cpol.
module tb_spi_a4_a5;
    reg clk, rst;
    reg bus_we, bus_re;
    reg [31:0] bus_addr, bus_wdata;
    wire [31:0] bus_rdata;
    wire spi_sck, spi_mosi, spi_cs_n;

    spi_master dut (
        .clk(clk), .rst(rst),
        .bus_we(bus_we), .bus_re(bus_re), .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_rdata(bus_rdata),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_miso(spi_mosi), .spi_cs_n(spi_cs_n)
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

    reg [7:0] rx;
    integer errors;

    initial begin
        #5000 begin
            $display("WATCHDOG TIMEOUT - busy=%b bits_left=%0d sck=%b", dut.busy, dut.bits_left, dut.sck_reg);
            $finish;
        end
    end

    initial begin
        clk = 0; rst = 1; bus_we = 0; bus_re = 0; bus_addr = 0; bus_wdata = 0;
        errors = 0;
        #12 rst = 0;
        @(posedge clk);

        // --- A4: establish CPOL=1 via a mode-only write (START=0), run one
        // full transfer in that mode so `cpol` register settles at 1 ---
        write_reg(8'h14, 16'd2);                 // CLKDIV
        write_reg(8'h10, {28'b0, 1'b0, 1'b1, 2'b00}); // CTRL: CPHA=0,CPOL=1, START=0
        write_reg(8'h18, 8'h3C);                  // TXDATA
        write_reg(8'h10, {28'b0, 1'b0, 1'b1, 2'b01}); // CTRL: same mode, START=1
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        @(posedge clk);
        $display("warm-up transfer (CPOL=1,CPHA=0): rx=%h (expect 3c), sck after=%b (expect 1=cpol)", dut.rxdata_reg, spi_sck);
        if (dut.rxdata_reg !== 8'h3C) errors = errors + 1;
        if (spi_sck !== 1'b1) begin errors = errors + 1; $display("  A5 FAIL: sck parked at %b, expected cpol=1", spi_sck); end

        // --- A4's real test: switch to CPOL=0,CPHA=0 with mode+START packed
        // into a SINGLE CTRL write (no separate mode-only write first). The
        // `cpol` register is still 1 (stale) at the instant this write's
        // start_pulse branch evaluates sck_reg's initial value - the fix
        // must read bus_wdata[2] (=0, the NEW mode) directly, not the stale
        // register, or the first half-period glitches and corrupts the bit.
        write_reg(8'h18, 8'h91);                  // TXDATA
        write_reg(8'h10, {28'b0, 1'b0, 1'b0, 2'b01}); // CTRL: CPOL=0,CPHA=0 AND START=1, same write
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        @(posedge clk);
        $display("combined mode+START transfer (CPOL=0,CPHA=0): rx=%h (expect 91), sck after=%b (expect 0=cpol)", dut.rxdata_reg, spi_sck);
        if (dut.rxdata_reg !== 8'h91) begin errors = errors + 1; $display("  A4 FAIL: byte corrupted by stale-cpol glitch"); end
        if (spi_sck !== 1'b0) begin errors = errors + 1; $display("  A5 FAIL: sck parked at %b, expected cpol=0", spi_sck); end

        // --- A5 cross-check: CPOL=1,CPHA=0 (opposite idle level) in its own
        // combined mode+START write, confirm sck parks at 1 this time too.
        write_reg(8'h18, 8'h5A);
        write_reg(8'h10, {28'b0, 1'b0, 1'b1, 2'b01}); // CTRL: CPOL=1,CPHA=0 AND START=1
        wait (dut.busy == 1'b1);
        wait (dut.busy == 1'b0);
        @(posedge clk);
        $display("combined mode+START transfer (CPOL=1,CPHA=0): rx=%h (expect 5a), sck after=%b (expect 1=cpol)", dut.rxdata_reg, spi_sck);
        if (dut.rxdata_reg !== 8'h5A) begin errors = errors + 1; $display("  A4 FAIL: byte corrupted by stale-cpol glitch"); end
        if (spi_sck !== 1'b1) begin errors = errors + 1; $display("  A5 FAIL: sck parked at %b, expected cpol=1", spi_sck); end

        if (errors == 0)
            $display("PASS: A4 (mode+START same write) and A5 (sck idle parking) both verified correct.");
        else
            $display("FAIL: %0d error(s), see above.", errors);

        $finish;
    end
endmodule
