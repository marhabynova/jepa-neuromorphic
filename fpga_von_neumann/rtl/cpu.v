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
    // design away as unobserved dead logic). KEPT UNCHANGED (still fires on
    // every MMIO-region store, same address 0xFF000000 the existing
    // regression test checks) so tb_cpu.v's PASS condition stays valid.
    output reg [31:0] mmio_out,
    output             mmio_valid,

    // Generic peripheral bus (added for SPI/Akida integration, SoC-level
    // design): fires in PARALLEL with the single-register debug port above
    // for ANY load/store whose address has top byte 0xFF - a peripheral
    // instantiated outside cpu.v (see soc_top.v) decodes addr[7:0] itself to
    // pick its own registers, so cpu.v stays a generic core that doesn't
    // need to know what's attached. mmio_re is asserted for exactly the
    // EXEC cycle of a LOAD targeting this region (same cycle the address is
    // first presented) - a peripheral should register mmio_rdata off the
    // CURRENT mmio_addr every clock edge unconditionally, the same
    // synchronous-read convention dmem.v already uses, so it's ready by the
    // time the CPU's extra MEM cycle consumes it.
    output             mmio_we,
    output             mmio_re,
    output [31:0]      mmio_addr,
    output [31:0]      mmio_wdata,
    input  [31:0]      mmio_rdata,

    // §96: minimal interrupt mechanism (no CSRs - this core has none - so
    // not spec-compliant RISC-V M-mode trap handling, just enough to let a
    // peripheral wake the CPU without being polled). A peripheral should
    // PULSE this for exactly one cycle per event (see spi_master.v's
    // irq_out) - cpu.v latches it into an internal irq_pending flip-flop
    // (below) so a short pulse is never missed even if it arrives mid-
    // instruction, when the FSM isn't at a fetch boundary yet.
    input              irq
);
    // §68/§74: imem/dmem reads became SYNCHRONOUS (1-cycle latency, the
    // timing real FPGA block RAM has) so yosys's memory_bram pass can infer
    // real BRAM instead of flattening them into individual flip-flops
    // (§68 found that flattening alone exceeded an iCE40 UP5K's whole LUT
    // budget). That retired the old single-cycle datapath: this is now a
    // small FETCH/EXEC/MEM FSM instead. Nothing overlaps (no pipelining),
    // so there are no hazards to handle - just an explicit 1-cycle wait
    // after presenting an address, before using what comes back.
    localparam S_FETCH = 2'd0, S_EXEC = 2'd1, S_MEM = 2'd2, S_WB = 2'd3;
    reg [1:0] state;
    reg [31:0] pc;
    wire [31:0] instr;

    wire [6:0] opcode = instr[6:0];
    wire [4:0] rd     = instr[11:7];
    wire [2:0] funct3 = instr[14:12];
    wire [4:0] rs1    = instr[19:15];
    wire [4:0] rs2    = instr[24:20];
    wire [6:0] funct7 = instr[31:25];

    wire reg_write, mem_write, mem_to_reg, alu_src, branch, jump, jalr, lui;
    wire auipc, illegal;
    wire iret, irq_en, irq_dis;
    wire [3:0] alu_op;
    wire [1:0] mem_size;
    wire load_unsigned;

    control ctrl (
        .opcode(opcode), .funct3(funct3), .funct7(funct7),
        .reg_write(reg_write), .mem_write(mem_write), .mem_to_reg(mem_to_reg),
        .alu_src(alu_src), .branch(branch), .jump(jump), .jalr(jalr), .lui(lui),
        .auipc(auipc),
        .iret(iret), .irq_en(irq_en), .irq_dis(irq_dis),
        .alu_op(alu_op), .illegal(illegal),
        .mem_size(mem_size), .load_unsigned(load_unsigned)
    );

    // Immediate generation for all formats we use (I/S/B/J/U), sign-extended
    // (U-type's bottom 12 bits are architecturally zero, not sign-extended -
    // LUI just places imm[31:12] directly, no extension needed there).
    wire [31:0] imm_i = {{20{instr[31]}}, instr[31:20]};
    wire [31:0] imm_s = {{20{instr[31]}}, instr[31:25], instr[11:7]};
    wire [31:0] imm_b = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
    wire [31:0] imm_j = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};
    wire [31:0] imm_u = {instr[31:12], 12'b0};

    reg [31:0] imm;
    always @(*) begin
        case (opcode)
            7'b0100011: imm = imm_s; // STORE
            7'b1100011: imm = imm_b; // BRANCH
            7'b1101111: imm = imm_j; // JAL
            7'b0010111: imm = imm_u; // AUIPC (§100) - rd <= pc + imm_u
            default:    imm = imm_i; // ITYPE / LOAD
        endcase
    end

    // `instr` (and everything decoded from it) stays stable across EXEC,
    // MEM and WB, because `pc` - imem's read address - only changes on the
    // edge that returns to FETCH. `returning_to_fetch` marks exactly that
    // edge: EXEC->FETCH for everything except loads, WB->FETCH for loads
    // (§103 added S_WB between S_MEM and S_FETCH - see load_data_raw_reg
    // below). Every other transition (FETCH->EXEC, EXEC->MEM, MEM->WB)
    // holds pc so addresses stay valid for memory that's still catching up
    // to a synchronous read.
    wire returning_to_fetch = (state == S_EXEC && !mem_to_reg) || (state == S_WB);
    wire reg_write_now = reg_write && returning_to_fetch;
    wire exec_now = (state == S_EXEC);

    wire [31:0] rd_data1, rd_data2;
    wire [31:0] wb_data;

    regfile rf (
        .clk(clk), .we(reg_write_now),
        .rd_addr1(rs1), .rd_addr2(rs2), .wr_addr(rd),
        .wr_data(wb_data),
        .rd_data1(rd_data1), .rd_data2(rd_data2)
    );

    wire [31:0] alu_b = alu_src ? imm : rd_data2;
    // §100: AUIPC is the one instruction whose left operand is the PC
    // rather than a register. Feeding it in here (one 32-bit mux) reuses
    // the ALU's existing adder instead of adding a second 32-bit adder,
    // and AUIPC's result then travels the ordinary alu_result writeback
    // path with no change to the wb_data mux.
    wire [31:0] alu_a = auipc ? pc : rd_data1;
    wire [31:0] alu_result;
    alu alu_u (.a(alu_a), .b(alu_b), .op(alu_op), .result(alu_result));

    // Gated by exec_now (§74 fix): without it, mem_write/mmio_sel would
    // stay combinationally true for one extra S_FETCH cycle too (instr -
    // and everything decoded from it - doesn't update until the NEXT
    // FETCH->EXEC edge), causing a harmless-but-wrong duplicate dmem write
    // and a second mmio_valid pulse for the same logical store.
    wire mmio_region = (alu_result[31:24] == 8'hFF);
    wire mmio_sel = mem_write && exec_now && mmio_region;
    assign mmio_valid = mmio_sel;

    always @(posedge clk or posedge rst) begin
        if (rst)
            mmio_out <= 32'd0;
        else if (mmio_sel)
            mmio_out <= rd_data2;
    end

    // Generic peripheral bus - same mmio_region gate as the debug port
    // above, so a store to 0xFF000000 (the debug port's address) still also
    // appears here; harmless, since no peripheral in soc_top.v decodes that
    // specific offset. addr/wdata are just alu_result/rd_data2 passed
    // through - cpu.v does no further decoding, that's the peripheral's job.
    assign mmio_we    = mmio_sel;
    assign mmio_re    = mem_to_reg && exec_now && mmio_region;
    assign mmio_addr  = alu_result;
    assign mmio_wdata = rd_data2;

    // §102: byte/half store alignment. dmem.v is still one 32-bit word
    // per address (addr[11:2]) with no read-modify-write - instead, the
    // store data is pre-replicated into every lane here and wstrobe picks
    // which lane(s) actually get written, so a byte store only needs to
    // present its 8 bits at whichever of the 4 positions addr[1:0]
    // selects (dmem then writes only that one byte of the word).
    reg [3:0]  dmem_wstrobe;
    reg [31:0] dmem_wdata;
    always @(*) begin
        case (mem_size)
            2'b00: begin // byte
                dmem_wstrobe = 4'b0001 << alu_result[1:0];
                dmem_wdata   = {rd_data2[7:0], rd_data2[7:0], rd_data2[7:0], rd_data2[7:0]};
            end
            2'b01: begin // half - addr[1] picks the lower or upper 16 bits
                dmem_wstrobe = alu_result[1] ? 4'b1100 : 4'b0011;
                dmem_wdata   = {rd_data2[15:0], rd_data2[15:0]};
            end
            default: begin // word
                dmem_wstrobe = 4'b1111;
                dmem_wdata   = rd_data2;
            end
        endcase
    end

    wire [31:0] dmem_rd;
    dmem dmem_u (
        .clk(clk), .addr(alu_result), .wr_data(dmem_wdata),
        .we(mem_write && exec_now && !mmio_sel), .wstrobe(dmem_wstrobe),
        .rd_data(dmem_rd)
    );

    // LOAD from an MMIO-region address reads mmio_rdata (the attached
    // peripheral's registered output) instead of dmem_rd. mmio_region is
    // driven by alu_result, which stays stable across EXEC->MEM exactly
    // like dmem's address does, so this mux is safe to use combinationally
    // here even though we're now in S_MEM.
    wire [31:0] load_data_raw = mmio_region ? mmio_rdata : dmem_rd;

    // §103: load_data_raw is REGISTERED here (captured at the S_MEM->S_WB
    // edge, one full cycle after it's valid) instead of being consumed
    // combinationally in the same cycle it arrives. §102's byte/half
    // extraction mux below sat directly on top of the BRAM's own
    // clock-to-Q delay in a single cycle (BRAM Q -> extract mux -> wb_data
    // -> regfile D), and icetime's independent timing check (not nextpnr's
    // own optimistic self-report - see RESEARCH_NOTES.md §99c) measured
    // that combined chain at 83.94ns (11.91MHz), ~0.6ns over the 12MHz/
    // 83.33ns target, consistently across every nextpnr seed tried. This
    // register breaks that single cycle into two: BRAM Q -> register
    // (S_MEM->S_WB), then register -> extract mux -> wb_data -> regfile D
    // (S_WB->S_FETCH) - same total load latency class (one more cycle,
    // loads already being the slowest instruction), zero change to any
    // value ever written, since alu_result[1:0]/mem_size/load_unsigned are
    // all still stable through S_WB for the same reason they were already
    // stable through S_MEM (instr/pc don't move until S_WB's own
    // returning_to_fetch edge).
    reg [31:0] load_data_raw_reg;
    always @(posedge clk) begin
        if (state == S_MEM)
            load_data_raw_reg <= load_data_raw;
    end

    reg [31:0] load_data;
    always @(*) begin
        case (mem_size)
            2'b00: begin // byte
                case (alu_result[1:0])
                    2'b00: load_data = load_unsigned ? {24'd0, load_data_raw_reg[7:0]}   : {{24{load_data_raw_reg[7]}},  load_data_raw_reg[7:0]};
                    2'b01: load_data = load_unsigned ? {24'd0, load_data_raw_reg[15:8]}  : {{24{load_data_raw_reg[15]}}, load_data_raw_reg[15:8]};
                    2'b10: load_data = load_unsigned ? {24'd0, load_data_raw_reg[23:16]} : {{24{load_data_raw_reg[23]}}, load_data_raw_reg[23:16]};
                    default: load_data = load_unsigned ? {24'd0, load_data_raw_reg[31:24]} : {{24{load_data_raw_reg[31]}}, load_data_raw_reg[31:24]};
                endcase
            end
            2'b01: begin // half
                if (alu_result[1])
                    load_data = load_unsigned ? {16'd0, load_data_raw_reg[31:16]} : {{16{load_data_raw_reg[31]}}, load_data_raw_reg[31:16]};
                else
                    load_data = load_unsigned ? {16'd0, load_data_raw_reg[15:0]}  : {{16{load_data_raw_reg[15]}},  load_data_raw_reg[15:0]};
            end
            default: load_data = load_data_raw_reg; // word
        endcase
    end

    wire [31:0] pc_plus4 = pc + 32'd4;
    // jump covers both JAL and JALR (both write rd<=pc+4, cpu.v's pc_next
    // mux below is what actually differs between them); lui writes imm_u
    // directly, no ALU involvement.
    assign wb_data = jump ? pc_plus4 : (lui ? imm_u : (mem_to_reg ? load_data : alu_result));

    // Dedicated branch comparator (separate from the ALU path) so all six
    // conditional branches can be expressed cleanly from funct3.
    // §100: BLTU/BGEU used to be missing - funct3 110/111 hit the
    // `default: branch_cond = 0` arm, so those branches were NEVER taken,
    // no matter what the operands were. (control.v also kept br=1 for
    // them, so they weren't flagged either - they just fell through
    // silently, which is exactly how an unsigned loop bound turns into an
    // off-by-everything bug.) `branch_ltu` is a genuinely different
    // comparison from `branch_lt`: for 0xFFFFFFFF vs 0x00000001 the
    // signed compare says less-than, the unsigned one says greater-than.
    wire branch_eq  = (rd_data1 == rd_data2);
    wire branch_lt  = ($signed(rd_data1) < $signed(rd_data2));
    wire branch_ltu = (rd_data1 < rd_data2);
    reg  branch_cond;
    always @(*) begin
        case (funct3)
            3'b000:  branch_cond = branch_eq;    // BEQ
            3'b001:  branch_cond = !branch_eq;   // BNE
            3'b100:  branch_cond = branch_lt;    // BLT
            3'b101:  branch_cond = !branch_lt;   // BGE
            3'b110:  branch_cond = branch_ltu;   // BLTU
            3'b111:  branch_cond = !branch_ltu;  // BGEU
            default: branch_cond = 1'b0;         // 010/011 unassigned (control.v flags them illegal)
        endcase
    end
    wire branch_taken = branch && branch_cond;

    // JALR's target is rs1+imm (already computed as alu_result, since
    // control.v sets asrc=1/aop=ADD for it), with the LSB cleared per the
    // RV32I spec (RISC-V jump targets are always 2-byte-aligned minimum;
    // we're 4-byte aligned throughout anyway, this just matches spec).
    // JAL's target is still pc+imm (PC-relative, no ALU involvement).
    wire [31:0] jalr_target = {alu_result[31:1], 1'b0};
    wire [31:0] pc_next_normal = jalr ? jalr_target :
                                (jump || branch_taken) ? (pc + imm) : pc_plus4;

    // §96: minimal interrupt mechanism. irq_pending latches the (possibly
    // single-cycle) `irq` input so it's never missed mid-instruction;
    // irq_taken is checked only at a fetch boundary (returning_to_fetch),
    // same discipline as every other pc_next decision here - interrupts
    // never land mid-instruction. epc holds the address execution would
    // have continued at, so IRET can resume exactly there. IRQ_VECTOR is a
    // fixed address (word 1 / byte 4) - a program wanting to use
    // interrupts must place its handler's first instruction there (e.g.
    // via an unconditional JAL at address 0 that jumps over this slot to
    // the real main program, the same convention real microcontrollers'
    // reset/vector tables use).
    localparam IRQ_VECTOR = 32'd4;
    reg irq_pending;
    reg irq_enable;
    reg [31:0] epc;
    // BUG FIX (Opus audit A1, RESEARCH_NOTES.md §99): DI (irq_dis) and a
    // pending interrupt both want to act on the SAME cycle whenever DI is
    // the currently-executing instruction - DI isn't a load, so exec_now
    // and returning_to_fetch are both true on DI's own EXEC cycle, exactly
    // the condition that also makes irq_taken eligible. Without the
    // exclusion below, the (unrelated) if-else priority in the update
    // block further down would dispatch the pending interrupt INSTEAD of
    // respecting DI - the one instruction whose entire purpose is to
    // block exactly that. Excluding DI's own cycle here (not just in the
    // update block) keeps every consumer of irq_taken - this wire, the
    // pc_next mux below, and irq_dispatch - consistently blind to the
    // interrupt for that one cycle; it stays pending (not cleared) for
    // whenever EI re-enables it later.
    wire irq_taken = irq_pending && irq_enable && !(exec_now && irq_dis);
    // BUG FIX (Opus audit A2): IRET is ALSO returning_to_fetch on its own
    // EXEC cycle (not a load), so if a handler calls EI before IRET and a
    // new interrupt is already pending by the time IRET executes,
    // irq_taken can be true on that exact cycle too. The pc_next mux
    // below already prioritizes iret correctly (checked first in the
    // ternary - PC goes to epc, not IRQ_VECTOR). But the SEPARATE update
    // block further down must not independently treat this as "the
    // interrupt was dispatched" - that would clear irq_pending (silently
    // dropping the interrupt that never actually got its handler run) and
    // clobber epc with pc_next_normal (a meaningless value - IRET's own
    // "next instruction" address, not a real return point for anything).
    // irq_dispatch excludes iret so bookkeeping only fires when the
    // interrupt is genuinely the thing that happened this cycle.
    wire irq_dispatch = returning_to_fetch && irq_taken && !iret;

    wire [31:0] pc_next = iret ? epc :
                          (returning_to_fetch && irq_taken) ? IRQ_VECTOR :
                          pc_next_normal;

    imem imem_u (.clk(clk), .addr(pc), .instr(instr));

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_FETCH;
            pc <= 32'd0;
        end else begin
            case (state)
                S_FETCH: state <= S_EXEC;
                S_EXEC:  state <= mem_to_reg ? S_MEM : S_FETCH; // loads need 1 more cycle
                S_MEM:   state <= S_WB; // §103: extra cycle to register load_data_raw
                S_WB:    state <= S_FETCH;
                default: state <= S_FETCH;
            endcase
            if (returning_to_fetch)
                pc <= pc_next;
            // else: hold pc - FETCH->EXEC and EXEC->MEM keep the address
            // stable for memory that's still catching up.
        end
    end

    // §100: observability for control.v's `illegal`. An unimplemented
    // encoding now retires as an architectural NOP (reg_write/mem_write
    // are both forced low in control.v) instead of executing as some
    // other instruction - but "did nothing" and "did the right thing"
    // look identical from the outside, so this sticky flag plus the PC
    // of the FIRST offender lets a testbench prove the instruction was
    // rejected rather than mis-executed. No output port and no soc_top
    // wiring: a real board has no pin to spare for it (§99c's pin-budget
    // lesson) and yosys sweeps both registers away during synthesis, so
    // this costs zero LCs. It is deliberately NOT a trap - this core has
    // no CSRs and no trap infrastructure (see control.v's §96 note).
    reg illegal_seen;
    reg [31:0] illegal_pc;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            illegal_seen <= 1'b0;
            illegal_pc   <= 32'd0;
        end else if (exec_now && illegal && !illegal_seen) begin
            illegal_seen <= 1'b1;
            illegal_pc   <= pc;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            irq_pending <= 1'b0;
            irq_enable  <= 1'b0;
            epc         <= 32'd0;
        end else begin
            // Latch any pulse, clear only once this specific interrupt is
            // actually taken (not just whenever irq happens to be low this
            // cycle - `irq` and the clear can coincide on the same edge).
            // Uses irq_dispatch (not the raw returning_to_fetch&&irq_taken)
            // so an IRET coincident with a new pending interrupt (A2) does
            // NOT clear it here - the interrupt never actually got
            // dispatched this cycle (PC went to epc, not IRQ_VECTOR), so
            // it must stay pending for whenever it's next checked.
            if (irq && !irq_dispatch)
                irq_pending <= 1'b1;
            else if (irq_dispatch)
                irq_pending <= 1'b0;

            if (irq_dispatch) begin
                epc         <= pc_next_normal; // where normal execution would have gone
                irq_enable  <= 1'b0;           // no nested interrupts without an explicit EI
            end else if (exec_now && irq_dis) begin
                irq_enable <= 1'b0;
            end else if (exec_now && (irq_en || iret)) begin
                // IRET re-enables interrupts as part of returning - standard
                // convention in most real architectures' return-from-trap.
                irq_enable <= 1'b1;
            end
        end
    end
endmodule
