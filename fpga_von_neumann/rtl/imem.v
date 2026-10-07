module imem (
    input         clk,
    input  [31:0] addr,
    output reg [31:0] instr
);
    // 1024 words = 4KB instruction space (§97: grown from 256/1KB - the
    // audit in RESEARCH_NOTES.md §95 flagged the old size as "smoke-test
    // scale only," too small for anything beyond toy programs; iCE40 UP5K
    // has 30 4Kbit EBR blocks total, and this + dmem's matching 4KB only
    // use 16 of them - comfortable headroom still left). Loaded from
    // program.hex at elaboration time (sim-only path for now; FPGA mapping
    // will need a block-RAM init file / $readmemh equivalent in the vendor
    // flow).
    //
    // SYNCHRONOUS READ (RESEARCH_NOTES.md §68/§74): `instr` now reflects
    // mem[addr] as of the PREVIOUS clock edge, not combinationally - this
    // is the read timing real FPGA block RAM has, and what yosys's
    // memory_bram pass needs to infer it instead of flattening this into
    // individual flip-flops (§68's finding: that flattening alone exceeded
    // an iCE40 UP5K's whole LUT budget). cpu.v's FETCH/EXEC/MEM FSM
    // accounts for this 1-cycle latency.
    reg [31:0] mem [0:1023];

    initial begin
        $readmemh("program.hex", mem);
    end

    always @(posedge clk)
        instr <= mem[addr[11:2]]; // word-aligned, byte address / 4
endmodule
