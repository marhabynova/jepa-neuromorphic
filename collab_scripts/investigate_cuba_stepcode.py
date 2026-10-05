"""
§36i continuation part 2: get the literal per-step integration formula
text that CuBaStepCode generates for our exact flag configuration
(leaky=True, fire=True, refractory=False since refractory_time=0,
exponential=False, no adaptation) - this is the piece cuba_aelif_integration's
own source doesn't show directly (it's built by CuBaStepCode elsewhere and
exec'd as a string each timestep).
"""
import inspect
import hxtorch.spiking.functional as F

print("=== CuBaStepCode class source ===")
print(inspect.getsource(F.CuBaStepCode))
