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

    // The program image used to be hard-coded as "program.hex" in the
    // current working directory, which meant EVERY cpu-level testbench
    // shared one mutable file: running tb_soc_spi after tb_cpu silently
    // ran the wrong program unless the matching tools/asm_*.py had been
    // re-run first. (The audit confirmed tb_cpu/tb_soc_spi/tb_soc_irq/
    // tb_jalr_lui/tb_irq_edge_cases all fail against a checked-in
    // program.hex that belongs to a different test.) IMEM_HEX lets the
    // Makefile hand each testbench its OWN immutable hex file
    // (-DIMEM_HEX=\"build/hex/program_spi.hex\"), so test order can never
    // change a result. The default keeps the old behaviour for anyone
    // invoking iverilog by hand, and yosys takes the same default (the
    // initial block is a simulation/BRAM-init construct either way).
`ifndef IMEM_HEX
  `define IMEM_HEX "program.hex"
`endif

    initial begin
        $readmemh(`IMEM_HEX, mem);
    end

    always @(posedge clk)
        instr <= mem[addr[11:2]]; // word-aligned, byte address / 4
endmodule
