"""
§36i continuation: trace what c_mem/g_l actually resolve to and whether
membrane_capacitance (default HXTransformedModelParameter(), unlike
i_synin_gm/synapse_dac_bias's bare HXParameter(500) defaults already proven
to have ZERO effect in mock mode - see RESEARCH_NOTES.md §56) behaves the
same way or differently. Run inside the EBRAINS Collab kernel where hxsnn/
GT2/x2/T2/IN_DIM2/HID2/OUT2 are already defined from prior cells.
"""
import inspect

# 1. Find where integration_step_code is actually built (the literal
#    per-step formula string exec'd inside cuba_aelif_integration).
import hxtorch.spiking.functional as F
print("=== Searching for the integration_step_code source ===")
for name in dir(F):
    obj = getattr(F, name)
    if inspect.isclass(obj) and 'cuba' in name.lower():
        print(name)
for name in dir(F):
    obj = getattr(F, name)
    if inspect.isclass(obj) and ('step' in name.lower() or 'code' in name.lower()):
        print(name)
print()

# 2. Inspect default membrane_capacitance resolved value on a real LIF.
lif_probe = hxsnn.LIF(4, hxsnn.Experiment(dt=1e-6, mock=True), tau_mem=4.48e-6, tau_syn=4.48e-6)
print("=== membrane_capacitance attribute ===")
print(type(lif_probe.membrane_capacitance), lif_probe.membrane_capacitance)
try:
    print("model_value:", lif_probe.membrane_capacitance.model_value)
except Exception as e:
    print("no .model_value:", e)
print()
print("=== leak_conductance attribute (default None per signature) ===")
print(type(lif_probe.leak_conductance), lif_probe.leak_conductance)
