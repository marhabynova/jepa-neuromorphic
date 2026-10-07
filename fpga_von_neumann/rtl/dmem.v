module dmem (
    input         clk,
    input  [31:0] addr,
    input  [31:0] wr_data,
    input         we,
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
        if (we)
            mem[addr[11:2]] <= wr_data;
    end
endmodule
