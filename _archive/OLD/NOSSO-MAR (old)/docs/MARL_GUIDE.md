# MARL Guide

## Algorithms
| Algorithm | Agent | Use case |
|---|---|---|
| MADDPG | WECLayoutAgent | Layout optimisation |
| MAPPO  | PTOControlAgent | PTO damping control |

## WaveFarmEnv
- Obs: local wave spectrum + position + neighbour positions
- Action: Δ(x,y) [m] — incremental, bounded
- Reward: total farm power (cooperative) + multi-objective terms

## Training
```bash
python scripts/train_marl_farm.py --config configs/marl/maddpg_farm.yaml
```
