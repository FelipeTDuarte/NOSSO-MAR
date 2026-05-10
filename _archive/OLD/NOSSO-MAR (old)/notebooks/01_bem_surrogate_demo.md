# Demo: BEM Surrogate (Module 2)

Open Jupyter and run:

```python
import numpy as np
from src.nosso_mar.modules.fsi.bem_surrogate import BEMSurrogate

cfg   = {}   # uses random weights (replace with trained checkpoint)
sur   = BEMSurrogate(cfg)
omega = np.linspace(0.2, 3.0, 64)
props = {"radius":3.,"draft":3.,"mass":5e4,"volume":85.,"Bpto":3e4,"cog":0.,"depth":50.}
out   = sur.predict(props, omega)

from src.nosso_mar.utils.visualisation import plot_bem_response
plot_bem_response(omega, out["added_mass"].numpy(), out["radiation_damping"].numpy(),
                  out["excitation_force"].numpy(), out["rao"].numpy())
```
