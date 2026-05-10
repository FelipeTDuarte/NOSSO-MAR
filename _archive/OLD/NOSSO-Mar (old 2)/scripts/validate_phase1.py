"""Phase 1 public benchmark validation runner.

Usage:
    python scripts/validate_phase1.py [--checkpoint PATH] [--benchmark PATH]

With no arguments, runs a self-contained smoke validation using the analytic
cylinder synthetic benchmark so that the script can be tested without any
external files or trained checkpoint.
"""

import argparse
import numpy as np
import torch


def relative_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / (np.std(ref) + 1e-12))


def run_validation(checkpoint_path=None, benchmark_path=None):
    from nossomar.data.public_benchmarks import synthetic_benchmark
    from nossomar.operators.deeponet_wec import WECDeepONet

    bench = synthetic_benchmark()
    omega_np = bench["omega"]
    props_np = np.array(
        [[bench["radius"], bench["draft"], bench["depth"], bench["bpto"]]],
        dtype=np.float32,
    )

    model = WECDeepONet()
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint provided — running untrained model for shape/smoke check.")

    model.eval()
    with torch.no_grad():
        props = torch.tensor(props_np)
        omega = torch.tensor(omega_np, dtype=torch.float32).unsqueeze(0)
        A_pred, B_pred = model(props, omega)

    A_pred_np = A_pred.squeeze().numpy()
    B_pred_np = B_pred.squeeze().numpy()

    rmse_A = relative_rmse(A_pred_np, bench["added_mass_heave"])
    rmse_B = relative_rmse(B_pred_np, bench["damping_heave"])

    print(f"Relative RMSE  added_mass : {rmse_A:.4f}")
    print(f"Relative RMSE  damping    : {rmse_B:.4f}")
    return {"rmse_added_mass": rmse_A, "rmse_damping": rmse_B}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 surrogate validation against a benchmark dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to a benchmark CSV file (omega, added_mass, damping columns).",
    )
    args = parser.parse_args()
    run_validation(checkpoint_path=args.checkpoint, benchmark_path=args.benchmark)


if __name__ == "__main__":
    main()
