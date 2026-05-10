import torch
from src.nosso_mar.reinforcement.environment.wave_farm_env import WaveFarmEnv
from src.nosso_mar.reinforcement.agents.layout_agent import WECLayoutAgent


def test_env_reset_step():
    env = WaveFarmEnv({"n_wec":3,"farm_bounds":[[0,1000],[0,1000]],"episode_length":5})
    obs = env.reset()
    assert len(obs) == 3
    next_obs, rewards, done, info = env.step([torch.zeros(2)]*3)
    assert "total_power" in info

def test_layout_agent_action():
    agent = WECLayoutAgent("w0", obs_dim=20, action_dim=2, n_agents=3, config={})
    act, _ = agent.select_action(torch.randn(1, 20))
    assert act.shape == (1, 2)
