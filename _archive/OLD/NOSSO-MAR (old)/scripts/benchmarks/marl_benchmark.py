"""Benchmark MARL environment step speed."""
import time, torch


def main():
    from src.nosso_mar.reinforcement.environment.wave_farm_env import WaveFarmEnv
    cfg = {"n_wec": 5, "farm_bounds": [[0,2000],[0,2000]], "episode_length": 200}
    env = WaveFarmEnv(cfg)
    env.reset()
    actions = [torch.zeros(2) for _ in range(5)]

    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        env.step(actions)
    ms = (time.perf_counter() - t0) / N * 1000
    print(f"WaveFarmEnv.step(): {ms:.3f} ms  (dummy NO, N={N})")


if __name__ == "__main__":
    main()
