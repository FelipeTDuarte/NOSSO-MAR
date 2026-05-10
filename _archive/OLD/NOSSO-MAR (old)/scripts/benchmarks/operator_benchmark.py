"""
Benchmark neural operator inference speed vs classical solvers.
Reports speedup factor per fitness evaluation for GA integration.

Usage:
    python scripts/benchmarks/operator_benchmark.py --n_samples 100
"""
import argparse, time, torch, numpy as np


def benchmark_fno2d(n_samples: int, grid: int = 128) -> float:
    from src.nosso_mar.operators.fno.fno2d import FNO2d
    model = FNO2d({"in_channels":6,"out_channels":3,"width":64,
                   "modes_x":16,"modes_y":16,"n_layers":4})
    model.eval()
    x = torch.randn(1, 6, grid, grid)
    # Warm-up
    with torch.no_grad():
        for _ in range(5): model(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_samples): model(x)
    return (time.perf_counter() - t0) / n_samples


def benchmark_wno(n_samples: int, grid: int = 128) -> float:
    from src.nosso_mar.operators.wno.wavelet_neural_operator import WaveletNeuralOperator
    model = WaveletNeuralOperator({"in_channels":6,"out_channels":3,"width":64,
                                    "levels":3,"n_layers":4})
    model.eval()
    x = torch.randn(1, 6, grid, grid)
    with torch.no_grad():
        for _ in range(5): model(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_samples): model(x)
    return (time.perf_counter() - t0) / n_samples


def benchmark_deeponet(n_samples: int, n_omega: int = 64) -> float:
    from src.nosso_mar.operators.deeponet.deeponet import DeepONet
    model = DeepONet({"branch_input_dim":7,"trunk_input_dim":1,
                      "hidden_dim":256,"n_hidden":5,"output_dim":4,"p":256})
    model.eval()
    b = torch.randn(1, 7)
    t = torch.linspace(0.2, 3.0, n_omega).unsqueeze(-1).unsqueeze(0)
    with torch.no_grad():
        for _ in range(5): model(b, t)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_samples): model(b, t)
    return (time.perf_counter() - t0) / n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--grid",      type=int, default=128)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"NOSSO-MAR Operator Benchmark  (n={args.n_samples}, grid={args.grid}x{args.grid})")
    print(f"{'='*55}")

    for name, fn in [("FNO2d",    lambda: benchmark_fno2d(args.n_samples, args.grid)),
                     ("WNO",      lambda: benchmark_wno(args.n_samples, args.grid)),
                     ("DeepONet", lambda: benchmark_deeponet(args.n_samples))]:
        ms = fn() * 1000
        print(f"  {name:<12}  {ms:.3f} ms / inference")

    # Reference SWAN estimate (from literature: ~60 s per run on single core)
    swan_s = 60.0
    fno_s  = benchmark_fno2d(10, args.grid)
    print(f"\n  Estimated speedup vs SWAN: {swan_s/fno_s:.0f}x  (FNO2d)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
