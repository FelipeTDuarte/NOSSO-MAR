# Demo: WEC Farm MARL Layout Optimisation

```python
import torch
from src.nosso_mar.reinforcement.environment.wave_farm_env import WaveFarmEnv
from src.nosso_mar.reinforcement.agents.layout_agent import WECLayoutAgent

cfg = {"n_wec":3,"farm_bounds":[[0,2000],[0,2000]],"episode_length":50}
env = WaveFarmEnv(cfg)

agents = [WECLayoutAgent(f"wec_{i}", env.observation_space_dim(), 2, 3, {})
          for i in range(3)]

obs = env.reset()
for step in range(50):
    obs_t   = [torch.tensor(o, dtype=torch.float32) for o in obs]
    actions = [a.select_action(o)[0] for a, o in zip(agents, obs_t)]
    obs, rewards, done, info = env.step(actions)
    print(f"Step {step}: total_power = {info['total_power']:.2e} W")
    if done: break
```
