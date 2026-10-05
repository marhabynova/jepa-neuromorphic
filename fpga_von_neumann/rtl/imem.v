module imem (
    input  [31:0] addr,
    output [31:0] instr
);
    // 256 words = 1KB instruction space. Loaded from program.hex at
    // elaboration time (sim-only path for now; FPGA mapping will need a
    // block-RAM init file / $readmemh equivalent in the vendor flow).
    reg [31:0] mem [0:255];

    initial begin
        $readmemh("program.hex", mem);
    end

    assign instr = mem[addr[9:2]]; // word-aligned, byte address / 4
endmodule
