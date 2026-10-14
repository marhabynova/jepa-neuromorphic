module alu (
    input  [31:0] a,
    input  [31:0] b,
    input  [3:0]  op,
    output [31:0] result
);
    localparam ALU_ADD = 4'b0000;
    localparam ALU_SUB = 4'b0001;
    localparam ALU_AND = 4'b0010;
    localparam ALU_OR  = 4'b0011;
    localparam ALU_XOR = 4'b0100;
    localparam ALU_SLT = 4'b0101;
    localparam ALU_SLL = 4'b0110;
    localparam ALU_SRL = 4'b0111;
    // §100: the two ops RV32I needs that this ALU never had. SRA is a
    // genuinely different operation from SRL (sign-replicating vs
    // zero-filling), and SLTU is a different comparison from SLT
    // (0xFFFFFFFF is the LARGEST value unsigned, the SMALLEST signed) -
    // neither can be emulated by the other, which is why control.v
    // mapping SRAI->SRL and SLTIU->SLT would have been silently wrong.
    localparam ALU_SRA  = 4'b1000;
    localparam ALU_SLTU = 4'b1001;

    reg [31:0] result_r;
    assign result = result_r;

    always @(*) begin
        case (op)
            ALU_ADD: result_r = a + b;
            ALU_SUB: result_r = a - b;
            ALU_AND: result_r = a & b;
            ALU_OR:  result_r = a | b;
            ALU_XOR: result_r = a ^ b;
            ALU_SLT: result_r = ($signed(a) < $signed(b)) ? 32'd1 : 32'd0;
            ALU_SLL: result_r = a << b[4:0];
            ALU_SRL: result_r = a >> b[4:0];                   // logical: zero-fill
            ALU_SRA: result_r = $signed(a) >>> b[4:0];          // arithmetic: sign-fill
            ALU_SLTU: result_r = (a < b) ? 32'd1 : 32'd0;       // unsigned compare
            default: result_r = 32'd0;
        endcase
    end
endmodule
