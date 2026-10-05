module cpu (
    input  clk,
    input  rst,
    // Memory-mapped I/O: any SW whose target address has top byte 0xFF
    // (i.e. addr[31:24] == 8'hFF) is diverted here instead of dmem - the
    // real-hardware analogue of writing to an LED/UART register instead of
    // RAM. mmio_out holds the last word written; mmio_valid pulses high for
    // exactly the cycle of a qualifying store, so an off-chip consumer can
    // edge-detect it. Added so cpu has externally observable outputs (see
    // RESEARCH_NOTES.md §46 - without this, yosys `synth` sweeps the whole
    // design away as unobserved dead logic).
    output reg [31:0] mmio_out,
    output             mmio_valid
);
    reg [31:0] pc;
    wire [31:0] instr;

    wire [6:0] opcode = instr[6:0];
    wire [4:0] rd     = instr[11:7];
    wire [2:0] funct3 = instr[14:12];
    wire [4:0] rs1    = instr[19:15];
    wire [4:0] rs2    = instr[24:20];
    wire [6:0] funct7 = instr[31:25];

    wire reg_write, mem_write, mem_to_reg, alu_src, branch, jump;
    wire [3:0] alu_op;

    control ctrl (
        .opcode(opcode), .funct3(funct3), .funct7(funct7),
        .reg_write(reg_write), .mem_write(mem_write), .mem_to_reg(mem_to_reg),
        .alu_src(alu_src), .branch(branch), .jump(jump), .alu_op(alu_op)
    );

    // Immediate generation for all formats we use (I/S/B/J), sign-extended.
    wire [31:0] imm_i = {{20{instr[31]}}, instr[31:20]};
    wire [31:0] imm_s = {{20{instr[31]}}, instr[31:25], instr[11:7]};
    wire [31:0] imm_b = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
    wire [31:0] imm_j = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};

    reg [31:0] imm;
    always @(*) begin
        case (opcode)
            7'b0100011: imm = imm_s; // STORE
            7'b1100011: imm = imm_b; // BRANCH
            7'b1101111: imm = imm_j; // JAL
            default:    imm = imm_i; // ITYPE / LOAD
        endcase
    end

    wire [31:0] rd_data1, rd_data2;
    wire [31:0] wb_data;

    regfile rf (
        .clk(clk), .we(reg_write),
        .rd_addr1(rs1), .rd_addr2(rs2), .wr_addr(rd),
        .wr_data(wb_data),
        .rd_data1(rd_data1), .rd_data2(rd_data2)
    );

    wire [31:0] alu_b = alu_src ? imm : rd_data2;
    wire [31:0] alu_result;
    alu alu_u (.a(rd_data1), .b(alu_b), .op(alu_op), .result(alu_result));

    wire mmio_sel = mem_write && (alu_result[31:24] == 8'hFF);
    assign mmio_valid = mmio_sel;

    always @(posedge clk or posedge rst) begin
        if (rst)
            mmio_out <= 32'd0;
        else if (mmio_sel)
            mmio_out <= rd_data2;
    end

    wire [31:0] dmem_rd;
    dmem dmem_u (
        .clk(clk), .addr(alu_result), .wr_data(rd_data2),
        .we(mem_write && !mmio_sel), .rd_data(dmem_rd)
    );

    wire [31:0] pc_plus4 = pc + 32'd4;
    assign wb_data = jump ? pc_plus4 : (mem_to_reg ? dmem_rd : alu_result);

    // Dedicated branch comparator (separate from the ALU path) so BEQ/BNE/
    // BLT/BGE can all be expressed cleanly from funct3. BLTU/BGEU (unsigned)
    // are not implemented yet - TODO before this instruction set is called
    // "complete".
    wire branch_eq = (rd_data1 == rd_data2);
    wire branch_lt = ($signed(rd_data1) < $signed(rd_data2));
    reg  branch_cond;
    always @(*) begin
        case (funct3)
            3'b000:  branch_cond = branch_eq;    // BEQ
            3'b001:  branch_cond = !branch_eq;   // BNE
            3'b100:  branch_cond = branch_lt;    // BLT
            3'b101:  branch_cond = !branch_lt;   // BGE
            default: branch_cond = 1'b0;
        endcase
    end
    wire branch_taken = branch && branch_cond;

    wire [31:0] pc_next = (jump || branch_taken) ? (pc + imm) : pc_plus4;

    imem imem_u (.addr(pc), .instr(instr));

    always @(posedge clk or posedge rst) begin
        if (rst)
            pc <= 32'd0;
        else
            pc <= pc_next;
    end
endmodule
