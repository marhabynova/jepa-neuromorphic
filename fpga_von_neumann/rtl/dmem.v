module dmem (
    input         clk,
    input  [31:0] addr,
    input  [31:0] wr_data,
    input         we,
    // §102: per-byte write enable so LB/LH/LBU/LHU/SB/SH don't need a
    // read-modify-write cycle - cpu.v derives this from mem_size/addr[1:0]
    // and pre-aligns wr_data into the matching lane(s); a word store just
    // sets all 4 bits (its previous, only, behaviour). Each bit gates an
    // independent nonblocking assignment to its own byte slice of the
    // same word, so unselected bytes are left completely untouched.
    input  [3:0]  wstrobe,
    output reg [31:0] rd_data
);
    // SYNCHRONOUS READ (§68/§74, same rationale as imem.v): `rd_data` now
    // reflects mem[addr] as of the previous edge, not combinationally.
    // Write was already synchronous (unchanged) - only read needed fixing.
    // 1024 words = 4KB data space (§97: grown from 256/1KB, matching
    // imem.v's grow for the same reason - see imem.v's comment).
    reg [31:0] mem [0:1023];
    integer i;

    initial begin
        for (i = 0; i < 1024; i = i + 1)
            mem[i] = 32'd0;
    end

    always @(posedge clk) begin
        rd_data <= mem[addr[11:2]];
        if (we) begin
            if (wstrobe[0]) mem[addr[11:2]][7:0]   <= wr_data[7:0];
            if (wstrobe[1]) mem[addr[11:2]][15:8]  <= wr_data[15:8];
            if (wstrobe[2]) mem[addr[11:2]][23:16] <= wr_data[23:16];
            if (wstrobe[3]) mem[addr[11:2]][31:24] <= wr_data[31:24];
        end
    end
endmodule
