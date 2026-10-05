module regfile (
    input         clk,
    input         we,
    input  [4:0]  rd_addr1,
    input  [4:0]  rd_addr2,
    input  [4:0]  wr_addr,
    input  [31:0] wr_data,
    output [31:0] rd_data1,
    output [31:0] rd_data2
);
    reg [31:0] regs [31:0];
    integer i;

    initial begin
        for (i = 0; i < 32; i = i + 1)
            regs[i] = 32'd0;
    end

    // x0 is hardwired to zero per RV32I spec, reads bypass the array.
    assign rd_data1 = (rd_addr1 == 5'd0) ? 32'd0 : regs[rd_addr1];
    assign rd_data2 = (rd_addr2 == 5'd0) ? 32'd0 : regs[rd_addr2];

    always @(posedge clk) begin
        if (we && wr_addr != 5'd0)
            regs[wr_addr] <= wr_data;
    end
endmodule
