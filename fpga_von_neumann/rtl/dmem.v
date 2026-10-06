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
    reg [31:0] mem [0:255]; // 256 words = 1KB data space
    integer i;

    initial begin
        for (i = 0; i < 256; i = i + 1)
            mem[i] = 32'd0;
    end

    always @(posedge clk) begin
        rd_data <= mem[addr[9:2]];
        if (we)
            mem[addr[9:2]] <= wr_data;
    end
endmodule
