# fpga_von_neumann - reproducible simulation + synthesis flow
#
# Why this file exists (RESEARCH_NOTES.md §100): until now every test was
# run by hand with ad-hoc iverilog invocations, and all six CPU-level
# testbenches shared ONE mutable `program.hex` in the working directory.
# Whichever tools/asm_*.py had been run last decided what the next test
# actually executed - an independent audit confirmed that tb_cpu,
# tb_soc_spi, tb_soc_irq, tb_jalr_lui and tb_irq_edge_cases all FAIL
# against a checked-in program.hex belonging to a different test.
#
#   make test    - regenerate every program image, run every testbench,
#                  print PASS/FAIL per test, exit non-zero on any failure
#   make synth   - yosys synth_ice40 + nextpnr-ice40 (UP5K/SG48/12MHz) +
#                  icepack + icetime, all output under build/
#   make clean   - remove build/
#
# Each CPU test gets its OWN hex file under build/hex and imem.v is
# compiled with -DIMEM_HEX pointing at it, so test order can never change
# a result again.

SHELL     := /bin/bash
RTL       := $(wildcard rtl/*.v)
# synth_top.v instantiates the iCE40 SB_IO pad primitive for the I2C
# bidirectional pins, which iverilog has no model for. It is a
# synthesis-only wrapper that no testbench instantiates (every bench
# drives soc_top or a sub-module directly), so simulation simply leaves
# it out; `make synth` still elaborates it through yosys.
SIM_RTL   := $(filter-out rtl/synth_top.v,$(RTL))
TB_DIR    := tb
MODELS    := tb/i2c_slave_model.v tb/spi_slave_model.v
TOOLS_DIR := tools
BUILD     := build
HEX       := $(BUILD)/hex
SIM       := $(BUILD)/sim
SYN       := $(BUILD)/synth

IVERILOG  := iverilog
VVP       := vvp
YOSYS     := yosys
NEXTPNR   := nextpnr-ice40
ICEPACK   := icepack
ICETIME   := icetime
PYTHON    := python3

SYNTH_TOP := synth_top
# No .pcf: this project has NO verified UPduino v3.1 pinout (see
# RESEARCH_NOTES.md §99c/§100). nextpnr free-places the I/O instead, which
# gives honest LUT/BRAM/IO/Fmax numbers but a bitstream that must NOT be
# programmed into a real board. Set PCF=<file> once a real pin map exists.
PCF       :=

# --- Test table -------------------------------------------------------
# CPU-level tests: <testbench>:<asm script>:<hex file it writes>
CPU_TESTS := \
	tb_cpu:asm.py:program.hex \
	tb_jalr_lui:asm_jalr_lui_test.py:program_jalr_lui.hex \
	tb_rv32i_ext:asm_rv32i_ext_test.py:program_rv32i_ext.hex \
	tb_illegal_instr:asm_rv32i_ext_test.py:program_rv32i_ext.hex \
	tb_soc_spi:asm_spi_test.py:program_spi.hex \
	tb_soc_irq:asm_irq_test.py:program_irq.hex \
	tb_soc_multi_periph:asm_multi_periph_test.py:program_multi_periph.hex \
	tb_irq_edge_cases:asm_irq_edge_cases.py:program_irq_edge_cases.hex

# Peripheral-level tests: no CPU, no program image needed.
PERIPH_TESTS := \
	tb_spi \
	tb_spi_a4_a5 \
	tb_spi_slave_model \
	tb_i2c \
	tb_i2c_open_drain

# Trace-only benches kept from debugging sessions: they print per-edge
# traces and have no PASS/FAIL verdict of their own. They are still
# compiled and run (a crash or a compile error is a real regression), but
# their output is not graded.
TRACE_TESTS := \
	tb_i2c_debug \
	tb_i2c_debug2 \
	tb_i2c_debug3 \
	tb_i2c_debug4 \
	tb_spi_debug

.PHONY: all test hex synth clean

all: test

# --- Program images ---------------------------------------------------
hex:
	@mkdir -p $(HEX)
	@set -e; for entry in $(CPU_TESTS); do \
		script=$${entry#*:}; script=$${script%%:*}; \
		hexfile=$${entry##*:}; \
		( cd $(HEX) && PYTHONPATH=$(CURDIR)/$(TOOLS_DIR) $(PYTHON) $(CURDIR)/$(TOOLS_DIR)/$$script >/dev/null ); \
		test -s $(HEX)/$$hexfile || { echo "ERROR: $$script did not produce $$hexfile"; exit 1; }; \
	done
	@echo "program images regenerated into $(HEX)/"

# --- Simulation -------------------------------------------------------
test: hex
	@mkdir -p $(SIM)
	@fail=0; \
	printf '%-26s %s\n' "TEST" "RESULT"; \
	printf '%-26s %s\n' "--------------------------" "------"; \
	for entry in $(CPU_TESTS); do \
		tb=$${entry%%:*}; hexfile=$${entry##*:}; \
		if ! $(IVERILOG) -g2005 -D IMEM_HEX='"$(CURDIR)/$(HEX)/'$$hexfile'"' \
		      -o $(SIM)/$$tb.out $(SIM_RTL) $(TB_DIR)/$$tb.v > $(SIM)/$$tb.compile.log 2>&1; then \
			printf '%-26s %s\n' "$$tb" "FAIL (compile)"; cat $(SIM)/$$tb.compile.log; fail=1; continue; fi; \
		$(VVP) $(SIM)/$$tb.out > $(SIM)/$$tb.log 2>&1 || true; \
		if grep -q '^PASS' $(SIM)/$$tb.log; then printf '%-26s %s\n' "$$tb" "PASS"; \
		else printf '%-26s %s\n' "$$tb" "FAIL"; sed 's/^/    | /' $(SIM)/$$tb.log; fail=1; fi; \
	done; \
	for tb in $(PERIPH_TESTS); do \
		if ! $(IVERILOG) -g2005 -o $(SIM)/$$tb.out $(SIM_RTL) $(MODELS) $(TB_DIR)/$$tb.v \
		      > $(SIM)/$$tb.compile.log 2>&1; then \
			printf '%-26s %s\n' "$$tb" "FAIL (compile)"; cat $(SIM)/$$tb.compile.log; fail=1; continue; fi; \
		$(VVP) $(SIM)/$$tb.out > $(SIM)/$$tb.log 2>&1 || true; \
		if grep -q '^PASS' $(SIM)/$$tb.log; then printf '%-26s %s\n' "$$tb" "PASS"; \
		else printf '%-26s %s\n' "$$tb" "FAIL"; sed 's/^/    | /' $(SIM)/$$tb.log; fail=1; fi; \
	done; \
	for tb in $(TRACE_TESTS); do \
		if ! $(IVERILOG) -g2005 -o $(SIM)/$$tb.out $(SIM_RTL) $(MODELS) $(TB_DIR)/$$tb.v \
		      > $(SIM)/$$tb.compile.log 2>&1; then \
			printf '%-26s %s\n' "$$tb" "FAIL (compile)"; cat $(SIM)/$$tb.compile.log; fail=1; continue; fi; \
		if $(VVP) $(SIM)/$$tb.out > $(SIM)/$$tb.log 2>&1; then printf '%-26s %s\n' "$$tb" "ran (trace-only)"; \
		else printf '%-26s %s\n' "$$tb" "FAIL (runtime)"; sed 's/^/    | /' $(SIM)/$$tb.log; fail=1; fi; \
	done; \
	echo; \
	if [ $$fail -ne 0 ]; then echo "RESULT: at least one test FAILED"; exit 1; \
	else echo "RESULT: all graded testbenches PASS"; fi

# --- Synthesis + place-and-route --------------------------------------
# Targets the iCE40 UP5K in the SG48 package at 12 MHz, the combination
# the UPduino v3.1 board uses.
synth:
	@mkdir -p $(SYN)
	@echo "*** WARNING: building WITHOUT a pin-constraint file. The pinout for"
	@echo "*** the real board has never been verified offline, so nextpnr"
	@echo "*** free-places every I/O. Resource and Fmax numbers below are real;"
	@echo "*** the resulting .bin is NOT safe to program into hardware."
	$(YOSYS) -p "read_verilog $(RTL); synth_ice40 -top $(SYNTH_TOP) -json $(SYN)/$(SYNTH_TOP).json" \
		> $(SYN)/yosys.log 2>&1 || { tail -40 $(SYN)/yosys.log; exit 1; }
	@grep -A40 'Printing statistics' $(SYN)/yosys.log | tail -40 || true
	$(NEXTPNR) --up5k --package sg48 --freq 12 \
		--json $(SYN)/$(SYNTH_TOP).json $(if $(PCF),--pcf $(PCF),) --asc $(SYN)/$(SYNTH_TOP).asc \
		> $(SYN)/nextpnr.log 2>&1 || { tail -40 $(SYN)/nextpnr.log; exit 1; }
	$(ICEPACK) $(SYN)/$(SYNTH_TOP).asc $(SYN)/$(SYNTH_TOP).bin
	$(ICETIME) -d up5k -mtr $(SYN)/$(SYNTH_TOP).rpt $(SYN)/$(SYNTH_TOP).asc > $(SYN)/icetime.log 2>&1 || true
	@echo "=== nextpnr utilisation / timing ==="
	@grep -E 'ICESTORM_LC|ICESTORM_RAM|SB_IO|SB_GB|Max frequency|Info: Device utilisation' $(SYN)/nextpnr.log || true
	@echo "=== icetime (independent cross-check) ==="
	@grep -E 'Total path delay|MHz' $(SYN)/icetime.log || true

clean:
	rm -rf $(BUILD)
