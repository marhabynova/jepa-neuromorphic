// §100: reset-release synchronizer - asynchronous assert, SYNCHRONOUS
// deassert.
//
// The external reset pin is a free-running asynchronous signal. Feeding
// it straight to every flip-flop's async reset (what synth_top did) is
// fine on the way IN - an asynchronous assert is exactly what you want,
// it works even with no clock running. The problem is on the way OUT:
// when the pin is released, different flip-flops across the die come
// out of reset on DIFFERENT clock edges, because the release instant has
// no defined relationship to the clock and the reset net has real skew
// across a 5280-LC device. A state machine whose state register leaves
// reset one cycle before its counter does starts from a combination
// that no reset value ever describes.
//
// Standard fix: assert asynchronously, release synchronously. The two
// flops below are held at 1 the instant rst_in rises (async), and after
// rst_in falls they shift a 0 through, so rst_out drops on a clock edge
// - the same edge for every flop that uses it. The second flop is also
// what gives any metastability from the asynchronous falling edge a
// full clock period to settle before it reaches the design.
//
// Kept as its own module rather than inlined in synth_top so it can be
// simulated and tested directly (synth_top itself instantiates the
// iCE40 SB_IO primitive and cannot be run under iverilog).
module reset_sync (
    input  clk,
    input  rst_in,   // asynchronous, active high, straight from the pin
    output rst_out   // async assert, synchronous deassert
);
    reg [1:0] sync;

    always @(posedge clk or posedge rst_in) begin
        if (rst_in) sync <= 2'b11;
        else        sync <= {sync[0], 1'b0};
    end

    assign rst_out = sync[1];
endmodule
