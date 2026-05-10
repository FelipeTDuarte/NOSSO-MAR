"""
Train MARL agents for WEC farm layout and PTO optimisation.

Usage:
    python scripts/train_marl_farm.py --config configs/marl/maddpg_farm.yaml
"""
import argparse, yaml, logging, torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.nosso_mar.reinforcement.environment.wave_farm_env import WaveFarmEnv
    from src.nosso_mar.reinforcement.agents.layout_agent import WECLayoutAgent
    from src.nosso_mar.reinforcement.algorithms.maddpg import MADDPG

    env     = WaveFarmEnv(cfg["env"])
    n_wec   = cfg["env"]["n_wec"]
    obs_dim = env.observation_space_dim()
    act_dim = env.action_space_dim()

    agents  = [WECLayoutAgent(f"wec_{i}", obs_dim, act_dim, n_wec,
                               cfg.get("agent_cfg", {}))
               for i in range(n_wec)]
    maddpg  = MADDPG(agents, cfg=cfg.get("maddpg_cfg", {}))

    n_episodes   = cfg.get("n_episodes",  5000)
    n_steps_ep   = cfg.get("episode_len", 200)
    update_every = cfg.get("update_every", 50)

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        for t in range(n_steps_ep):
            obs_t   = [torch.tensor(o, dtype=torch.float32) for o in obs]
            actions = maddpg.act(obs_t)
            next_obs, rewards, done, info = env.step(actions)
            maddpg.store(obs, actions, rewards, next_obs, [done]*n_wec)
            obs        = next_obs
            ep_reward += sum(rewards) / n_wec
            if done: break

        if ep % update_every == 0:
            losses = maddpg.update()
            logger.info(f"Ep {ep}/{n_episodes} | reward={ep_reward:.2f} | {losses}")

        if ep % 500 == 0:
            maddpg.save(f"outputs/maddpg_ep{ep}.pt")

    logger.info("MARL training complete.")


if __name__ == "__main__":
    main()
