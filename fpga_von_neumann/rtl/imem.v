module imem (
    input         clk,
    input  [31:0] addr,
    output reg [31:0] instr
);
    // 256 words = 1KB instruction space. Loaded from program.hex at
    // elaboration time (sim-only path for now; FPGA mapping will need a
    // block-RAM init file / $readmemh equivalent in the vendor flow).
    //
    // SYNCHRONOUS READ (RESEARCH_NOTES.md §68/§74): `instr` now reflects
    // mem[addr] as of the PREVIOUS clock edge, not combinationally - this
    // is the read timing real FPGA block RAM has, and what yosys's
    // memory_bram pass needs to infer it instead of flattening this into
    // 256*32 individual flip-flops (§68's finding: that flattening alone
    // exceeded an iCE40 UP5K's whole LUT budget). cpu.v's FETCH/EXEC/MEM
    // FSM accounts for this 1-cycle latency.
    reg [31:0] mem [0:255];

    initial begin
        $readmemh("program.hex", mem);
    end

    always @(posedge clk)
        instr <= mem[addr[9:2]]; // word-aligned, byte address / 4
endmodule
