`timescale 1ns/1ps

// Loopback self-test: MISO wired directly to MOSI. If the shift timing is
// correct, whatever byte is sent should also be exactly what comes back in
// RXDATA, for any of the 4 SPI modes. Does NOT prove compatibility with a
// real external device (that needs real hardware), but DOES prove the
// MOSI-placement-vs-MISO-sampling edge timing is internally consistent,
// which is the part a design mistake would actually break.
module tb_spi;
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

    // cpol_bit/cpha_bit passed explicitly (not packed into a 2-bit "mode")
    // to avoid any ambiguity about which register bit is which - matches
    // spi_master.v's CTRL layout directly: bit2=CPOL, bit3=CPHA.
    task do_transfer(input [7:0] tx_byte, input cpol_bit, input cpha_bit, input [15:0] div, output [7:0] rx_byte);
        begin
            write_reg(8'h14, div);                                       // CLKDIV
            write_reg(8'h10, {28'b0, cpha_bit, cpol_bit, 2'b00});          // CTRL: set mode, START=0
            write_reg(8'h18, tx_byte);                                     // TXDATA
            write_reg(8'h10, {28'b0, cpha_bit, cpol_bit, 2'b01});          // CTRL: same mode, START=1
            wait (dut.busy == 1'b1);
            wait (dut.busy == 1'b0);
            @(posedge clk);
            rx_byte = dut.rxdata_reg;
        end
    endtask

    reg [7:0] rx;
    integer errors;

    initial begin
        #5000 begin
            $display("WATCHDOG TIMEOUT - busy=%b bits_left=%0d sck=%b cs_n=%b div_cnt=%0d",
                      dut.busy, dut.bits_left, dut.sck_reg, dut.cs_n_reg, dut.div_cnt);
            $finish;
        end
    end

    initial begin
        clk = 0; rst = 1; bus_we = 0; bus_re = 0; bus_addr = 0; bus_wdata = 0;
        errors = 0;
        #12 rst = 0;
        @(posedge clk);

        // Mode 0 (CPOL=0,CPHA=0)
        do_transfer(8'hA5, 1'b0, 1'b0, 16'd2, rx); // mode0 CPOL=0,CPHA=0
        $display("mode0 sent=8'hA5 got=%h (expect a5)", rx);
        if (rx !== 8'hA5) errors = errors + 1;

        do_transfer(8'h3C, 1'b0, 1'b0, 16'd2, rx); // mode0 CPOL=0,CPHA=0
        $display("mode0 sent=8'h3C got=%h (expect 3c)", rx);
        if (rx !== 8'h3C) errors = errors + 1;

        // Mode 1 (CPOL=0,CPHA=1)
        do_transfer(8'h7E, 1'b0, 1'b1, 16'd2, rx); // mode1 CPOL=0,CPHA=1
        $display("mode1 sent=8'h7E got=%h (expect 7e)", rx);
        if (rx !== 8'h7E) errors = errors + 1;

        // Mode 2 (CPOL=1,CPHA=0)
        do_transfer(8'h91, 1'b1, 1'b0, 16'd2, rx); // mode2 CPOL=1,CPHA=0
        $display("mode2 sent=8'h91 got=%h (expect 91)", rx);
        if (rx !== 8'h91) errors = errors + 1;

        // Mode 3 (CPOL=1,CPHA=1)
        do_transfer(8'h5A, 1'b1, 1'b1, 16'd2, rx); // mode3 CPOL=1,CPHA=1
        $display("mode3 sent=8'h5A got=%h (expect 5a)", rx);
        if (rx !== 8'h5A) errors = errors + 1;

        // STATUS register sanity: busy should read 0 after transfers finish
        bus_addr = 32'hFF00001C; bus_re = 1; @(posedge clk); bus_re = 0; @(posedge clk);
        $display("STATUS.busy = %0d (expect 0)", bus_rdata[0]);
        if (bus_rdata[0] !== 1'b0) errors = errors + 1;

        if (errors == 0)
            $display("PASS: spi_master loopback correct across all 4 modes.");
        else
            $display("FAIL: %0d mismatch(es), see above.", errors);

        $finish;
    end
endmodule
