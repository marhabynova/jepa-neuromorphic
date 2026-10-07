`timescale 1ns/1ps

// §99: swept-irq-timing regression for the Opus audit's A1/A2 findings
// (RESEARCH_NOTES.md §99, recommendation C1) - direct cpu.v instantiation
// (no peripherals needed), irq driven straight by this testbench, timed by
// watching cpu.v's own internal control signals rather than guessing
// cycle counts. See tools/asm_irq_edge_cases.py for the program structure.
//
// A1 (DI defeated by a same-cycle interrupt): inject irq exactly on DI's
// own exec_now cycle. Pre-fix, the old if-else priority in cpu.v's update
// block dispatched the interrupt anyway (DI lost the race) - handler
// invocation count would already be 1 by "marker A" (right after DI's
// observation window), instead of only becoming 1 later, after the
// explicit re-EI.
//
// A2 (IRET coincident with a pending+enabled interrupt loses the IRQ and
// corrupts epc): inject a second irq exactly on the handler's own EI
// cycle (handler = EI;IRET back to back, so pending+enabled both become
// true exactly on IRET's own cycle - the race). Pre-fix, the old priority
// cleared irq_pending and clobbered epc with a bogus value as if the
// interrupt had been dispatched, even though PC actually went to epc
// (IRET's destination) - the interrupt was silently dropped and never
// fired again. Post-fix, it stays pending, correctly fires again shortly
// after IRET returns (invocation count reaches 3).
module tb_irq_edge_cases;
    reg clk, rst;
    reg irq;
    wire [31:0] mmio_out;
    wire mmio_valid;
    wire mmio_we, mmio_re;
    wire [31:0] mmio_addr, mmio_wdata;

    cpu dut (
        .clk(clk), .rst(rst),
        .mmio_out(mmio_out), .mmio_valid(mmio_valid),
        .mmio_we(mmio_we), .mmio_re(mmio_re),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_rdata(32'd0),
        .irq(irq)
    );

    always #5 clk = ~clk;

    // Counts handler invocations by watching the FETCH of IRQ_VECTOR
    // (addr 4) - independent of whatever the handler itself does with
    // registers, so the same handler body works for counting across both
    // phases without needing a dedicated marker register.
    integer invocations;
    always @(posedge clk) begin
        if (!rst && dut.state == 2'd0 && dut.pc == 32'd4)
            invocations <= invocations + 1;
    end

    integer errors;

    initial begin
        #20000 begin
            $display("WATCHDOG TIMEOUT - pc=%0d state=%0d invocations=%0d", dut.pc, dut.state, invocations);
            $finish;
        end
    end

    initial begin
        clk = 0; rst = 1; irq = 0; invocations = 0; errors = 0;
        #12 rst = 0;

        // --- A1: wait for DI's own exec_now cycle, inject irq on it ---
        wait (dut.irq_dis && dut.exec_now);
        irq = 1;
        @(posedge clk); #1;
        irq = 0;
        $display("[A1] injected irq on DI's own exec cycle (invocations=%0d so far, expect 0)", invocations);
        if (invocations !== 0) begin
            errors = errors + 1;
            $display("  A1 FAIL: DI lost the race - handler already ran despite DI, invocations=%0d", invocations);
        end

        // marker A: x21==111 means the post-DI observation window has closed.
        wait (dut.rf.regs[21] == 111);
        $display("[A1] marker A reached, invocations=%0d (expect still 0 - deferred, not lost)", invocations);
        if (invocations !== 0) begin
            errors = errors + 1;
            $display("  A1 FAIL: handler ran during the DI window, expected still 0 at marker A");
        end

        // marker B: x22==222 means re-EI's window has closed - the
        // deferred interrupt should have fired exactly once by now.
        wait (dut.rf.regs[22] == 222);
        $display("[A1] marker B reached, invocations=%0d (expect 1 - deferred interrupt finally delivered)", invocations);
        if (invocations !== 1) begin
            errors = errors + 1;
            $display("  A1 FAIL: expected exactly 1 invocation by marker B, got %0d", invocations);
        end

        // --- A2: trigger a fresh (non-racy) interrupt somewhere in loop4
        // to get a second handler invocation running, then watch for ITS
        // own EI cycle and inject a second irq exactly then.
        irq = 1;
        @(posedge clk); #1;
        irq = 0;

        wait (dut.irq_en && dut.exec_now);  // the handler's own EI, unambiguous from here on
        irq = 1;
        @(posedge clk); #1;
        irq = 0;
        // This edge is the handler's own IRET FETCH cycle (pc=8, state=
        // S_FETCH) - irq_pending/irq_enable are now both 1 (latched by
        // this same edge), but the race hasn't happened yet: dispatch
        // decisions only happen on a returning_to_fetch cycle, which for
        // IRET is its EXEC cycle, one edge further on.
        $display("[A2] injected irq on handler's own EI cycle (now at its IRET fetch: pc=%0d state=%0d irq_pending=%b irq_enable=%b epc=%0d)", dut.pc, dut.state, dut.irq_pending, dut.irq_enable, dut.epc);
        if (dut.pc !== 32'd8) begin
            errors = errors + 1;
            $display("  A2 TEST SETUP FAIL: expected to be at IRET's fetch (pc=8), got pc=%0d - race timing assumption broke", dut.pc);
        end

        @(posedge clk); #1;   // IRET's own EXEC cycle (the actual race: iret vs irq_taken)
        @(posedge clk); #1;   // one more edge - whatever the update block decided this past cycle is now visible
        $display("[A2] right after IRET's own exec cycle: pc=%0d epc=%0d irq_pending=%b invocations=%0d (expect epc UNCHANGED=72, irq_pending still 1=preserved)", dut.pc, dut.epc, dut.irq_pending, invocations);
        if (dut.epc !== 32'd72) begin
            errors = errors + 1;
            $display("  A2 FAIL: epc corrupted by the IRET/interrupt race - got %0d, expected 72 (IRET's own return address, untouched)", dut.epc);
        end
        if (dut.irq_pending !== 1'b1) begin
            errors = errors + 1;
            $display("  A2 FAIL: irq_pending was cleared by the race even though IRET (not the interrupt) actually took the PC this cycle - the interrupt was silently dropped");
        end

        // marker D: the preserved (not lost) second irq should have been
        // dispatched again shortly after IRET returned - by now it may
        // already have happened (dispatch can fire on literally the next
        // returning_to_fetch cycle once pending+enabled are both true),
        // so this just confirms it was NOT lost outright, not that it
        // fires at any particular instruction count.
        wait (dut.rf.regs[24] == 444);
        $display("[A2] marker D reached, invocations=%0d (expect 3 - the deferred interrupt was preserved, not dropped)", invocations);
        if (invocations !== 3) begin
            errors = errors + 1;
            $display("  A2 FAIL: expected exactly 3 invocations at marker D, got %0d - interrupt was silently lost", invocations);
        end

        if (errors == 0)
            $display("PASS: A1 (DI priority) and A2 (IRET priority, no lost/corrupted interrupt) both verified correct.");
        else
            $display("FAIL: %0d error(s), see above.", errors);

        $finish;
    end
endmodule
