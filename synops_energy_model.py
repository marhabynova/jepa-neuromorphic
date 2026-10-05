"""
SynOps energy model (RESEARCH_NOTES.md §15/§9/§11, planned since early in the
project, never implemented until now). Pure arithmetic, no GPU run needed --
uses REAL architecture dims from run_phase_a.py/grid_sweep.py (the actual
N-MNIST collapse experiment, §16-17) and the REAL measured spike rate from
that experiment (§16: avg spike_rate = 0.0556 across the confirmed-collapse
run), not a made-up or retracted earlier number.

Two separate, clearly-labeled numbers, not blended into one:
  1. "As actually run" (GPU): both DenseJEPA and SpikingJEPA cost the SAME
     dense MACs on a GPU, since GPU hardware computes the full matmul
     regardless of whether downstream values get thresholded into a spike.
     Sparsity gives ZERO speedup here -- this is the honest answer to
     "does this save me GPU cost right now, on my own hardware".
  2. "If deployed on real event-driven hardware" (Loihi-1 proxy): layer 1
     (fc1, FLAT_DIM->HIDDEN) is conservatively treated as full-dense (its
     input is the raw frame, not a measured spike train -- no sparsity
     claimed there without evidence). Layers 2-3 (fc2, fc3) are gated by the
     REAL measured aggregate spike_rate from §16.

Energy constants, both cited, both explicitly NOT Loihi 2 (no public pJ/SOP
figure exists for Loihi 2 -- checked directly against Intel's own Loihi 2
technology brief during this session, confirmed absent):
  - MAC_ENERGY_PJ = 4.6 pJ (32-bit floating point multiply 3.7pJ + add 0.9pJ,
    Horowitz, ISSCC 2014 -- the standard citation used throughout the
    efficient-DNN-hardware literature, e.g. Sze et al. survey). This is an
    ARITHMETIC-ONLY floor: real GPU energy is dominated by data movement
    (RESEARCH_NOTES.md §1, same Horowitz source: 32-bit DRAM access ~640pJ,
    ~4 orders of magnitude more than a 32-bit add) -- so this UNDERSTATES
    real dense-GPU energy, which if anything makes the spiking comparison
    more conservative, not less.
  - SOP_ENERGY_PJ = 23.6 pJ (Loihi 1, Davies et al. 2018, "Loihi: A
    Neuromorphic Manycore Processor with On-Chip Learning", pre-silicon
    SDF+SPICE simulation at 0.75V). Loihi 2 is reported "5x faster for
    synaptic operations" (Intel Loihi 2 technical brief, footnote 2) but
    speed != energy-per-op, and no energy figure was published alongside
    that speed claim -- using the Loihi 1 number as a labeled proxy, NOT
    extrapolating a Loihi 2 number that was never published.
"""

# --- Real architecture dims, from run_phase_a.py / grid_sweep.py (§16-17) ---
FLAT_DIM = 34 * 34 * 2  # N-MNIST sensor size, real_data.py
HIDDEN_DIM = 128
REPR_DIM = 64
N_TIME_BINS = 20  # real_data.py, sequence length fed to the encoder

# --- Real measured result, from RESEARCH_NOTES.md §16 (confirmed-collapse run) ---
MEASURED_SPIKE_RATE = 0.0556  # 5.56%, pooled across spk1+spk2+spk3

# --- Published energy constants (see docstring for full citation) ---
MAC_ENERGY_PJ = 4.6   # Horowitz ISSCC 2014, 32-bit FP multiply+add
SOP_ENERGY_PJ = 23.6  # Davies et al. 2018, Loihi 1 (NOT Loihi 2 -- unavailable)

# --- Per-timestep MACs per layer (encoder only, matches both architectures
#     layer-for-layer: fc1 in->hid, fc2 hid->hid, fc3 hid->out) ---
macs_l1 = FLAT_DIM * HIDDEN_DIM
macs_l2 = HIDDEN_DIM * HIDDEN_DIM
macs_l3 = HIDDEN_DIM * REPR_DIM
macs_per_timestep = macs_l1 + macs_l2 + macs_l3
macs_per_sample = macs_per_timestep * N_TIME_BINS

print("=== Architecture (real dims from §16-17 N-MNIST experiment) ===")
print(f"FLAT_DIM={FLAT_DIM} HIDDEN_DIM={HIDDEN_DIM} REPR_DIM={REPR_DIM} N_TIME_BINS={N_TIME_BINS}")
print(f"MACs per timestep: L1={macs_l1:,} L2={macs_l2:,} L3={macs_l3:,} total={macs_per_timestep:,}")
print(f"MACs per sample (x{N_TIME_BINS} timesteps): {macs_per_sample:,}")

# Opus audit 2026-10-02 caught a 1000x unit bug here: 1 uJ = 1e6 pJ, not 1e9.
# All uJ figures below are now divided by 1e6 (correct), not 1e9 (wrong).
print("\n=== Condition 1: AS ACTUALLY RUN (GPU, both architectures) ===")
energy_gpu_pj = macs_per_sample * MAC_ENERGY_PJ
print(f"Dense JEPA on GPU:   {macs_per_sample:,} MACs/sample -> {energy_gpu_pj:,.0f} pJ/sample ({energy_gpu_pj/1e6:.2f} uJ/sample)")
print(f"Spiking JEPA on GPU: {macs_per_sample:,} MACs/sample -> {energy_gpu_pj:,.0f} pJ/sample ({energy_gpu_pj/1e6:.2f} uJ/sample)")
print("-> IDENTICAL. GPU computes the full dense matmul regardless of spike thresholding downstream.")
print("-> Running SpikingJEPA on this laptop's GPU saves ZERO compute/energy vs DenseJEPA.")

macs_l1_total = macs_l1 * N_TIME_BINS
synops_l23_total = (macs_l2 + macs_l3) * N_TIME_BINS * MEASURED_SPIKE_RATE

print(f"\n=== Condition 2a: neuromorphic, MIXED pricing (L1 at MAC energy, L2-3 at SOP energy) ===")
print("Opus audit flagged this as internally inconsistent -- L1 is 92% of ops but priced at the")
print("CHEAPER-per-op currency (MAC, not SOP) inside a 'deployed on neuromorphic hardware' total.")
print("Shown for comparison only, NOT the headline number.")
energy_neuromorphic_mixed_pj = macs_l1_total * MAC_ENERGY_PJ + synops_l23_total * SOP_ENERGY_PJ
print(f"Layer 1 (priced as dense MAC):    {macs_l1_total:,.0f} ops -> {macs_l1_total*MAC_ENERGY_PJ:,.0f} pJ")
print(f"Layers 2-3 (priced as SynOps, {MEASURED_SPIKE_RATE:.1%} spike rate): {synops_l23_total:,.0f} ops -> {synops_l23_total*SOP_ENERGY_PJ:,.0f} pJ")
print(f"Total: {energy_neuromorphic_mixed_pj:,.0f} pJ/sample ({energy_neuromorphic_mixed_pj/1e6:.2f} uJ/sample)")

print(f"\n=== Condition 2b: neuromorphic, CONSISTENT pricing (every op priced at SOP energy, incl. L1) ===")
print("This is the honest version: on real neuromorphic hardware, EVERY synaptic operation --")
print("spiking or not -- costs the SOP energy, not the GPU MAC energy. L1 does 100% of its ops")
print("(no measured sparsity there), so it pays full SOP price for all of them.")
synops_l1_total = macs_l1_total  # same op count, priced differently
energy_neuromorphic_consistent_pj = synops_l1_total * SOP_ENERGY_PJ + synops_l23_total * SOP_ENERGY_PJ
print(f"Layer 1 (100% of ops, SOP price): {synops_l1_total:,.0f} ops -> {synops_l1_total*SOP_ENERGY_PJ:,.0f} pJ")
print(f"Layers 2-3 ({MEASURED_SPIKE_RATE:.1%} of ops, SOP price): {synops_l23_total:,.0f} ops -> {synops_l23_total*SOP_ENERGY_PJ:,.0f} pJ")
print(f"Total: {energy_neuromorphic_consistent_pj:,.0f} pJ/sample ({energy_neuromorphic_consistent_pj/1e6:.2f} uJ/sample)")

ratio_mixed = energy_gpu_pj / energy_neuromorphic_mixed_pj
ratio_consistent = energy_gpu_pj / energy_neuromorphic_consistent_pj
print(f"\n=== Comparison ===")
print(f"GPU (as actually run):                        {energy_gpu_pj/1e6:.2f} uJ/sample")
print(f"Neuromorphic, mixed pricing (2a):              {energy_neuromorphic_mixed_pj/1e6:.2f} uJ/sample -> ratio {ratio_mixed:.2f}x")
print(f"Neuromorphic, consistent SOP pricing (2b):     {energy_neuromorphic_consistent_pj/1e6:.2f} uJ/sample -> ratio {ratio_consistent:.2f}x")
print(f"\n-> Under the HONEST consistent-pricing model (2b), neuromorphic deployment of THIS architecture")
print(f"   is {1/ratio_consistent:.1f}x WORSE than the GPU floor-estimate, not better -- because SOP energy (23.6pJ)")
print(f"   exceeds MAC energy (4.6pJ) per operation, and L1's 92%-of-ops share pays that higher price")
print(f"   on 100% of its operations (no sparsity there) with no offsetting discount.")
print("\nCAVEATS (read before citing this number anywhere):")
print("1. Loihi-1 proxy, NOT Loihi-2 -- no public Loihi-2 pJ/SOP figure exists (checked against Intel's own brief).")
print("2. The GPU number is ARITHMETIC-ONLY (MAC energy alone) -- real GPU energy is dominated by data movement")
print("   (RESEARCH_NOTES.md §1), so 29.49 uJ/sample is a FLOOR; real GPU energy is higher still.")
print("3. Uses the REAL measured spike rate from §16, but that run is CONFIRMED COLLAPSED (§16-17) -- a collapsed")
print("   encoder's spike rate reflects a degenerate/non-useful representation, NOT a validated efficient one.")
print("4. Per-sample inference-only estimate. Training (backward pass, optimizer state, EMA target update) not included.")
print("5. Net conclusion unchanged in direction either way: this toy architecture does NOT demonstrate a neuromorphic")
print("   energy win, under 2a it's a wash (~1.06x) and under the more honest 2b it's actively worse (~0.2x).")
