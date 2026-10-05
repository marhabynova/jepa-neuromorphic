module dmem (
    input         clk,
    input  [31:0] addr,
    input  [31:0] wr_data,
    input         we,
    output [31:0] rd_data
);
    reg [31:0] mem [0:255]; // 256 words = 1KB data space
    integer i;

    initial begin
        for (i = 0; i < 256; i = i + 1)
            mem[i] = 32'd0;
    end

    assign rd_data = mem[addr[9:2]];

    always @(posedge clk) begin
        if (we)
            mem[addr[9:2]] <= wr_data;
    end
endmodule
