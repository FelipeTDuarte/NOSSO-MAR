# Data Assimilation

## EnKF  (`assimilation/enkf.py`)
Stochastic EnKF, perturbed observations, covariance inflation, optional localisation.
Forecast model = any NOSSO-MAR module.run().

## 4D-Var  (`assimilation/four_dvar.py`)
AD through neural operators replaces classical adjoint. Minimiser: L-BFGS.

## Observation operators
| Sensor | Class |
|---|---|
| Wave buoy | WaveBuoyObsOperator |
| Satellite altimeter | SatelliteSwathOperator |

## State vector
WaveStateVector = [η(x,y), u, v, h, Hs, Tp] — serialised as flat tensor for EnKF.
