from nossomar.training.train_wec import train_wec_surrogate
from nossomar.data.wec_dataset import generate_analytic_dataset


def test_loss_decreases():
    ds = generate_analytic_dataset(n_cases=200, seed=0)
    history = train_wec_surrogate(ds, epochs=10, lr=1e-3, batch_size=32)
    # The minimum loss over all epochs must be below the first epoch loss.
    # This is robust to single-epoch fluctuations in stochastic training.
    assert min(history) < history[0], (
        f"Loss never improved: min={min(history):.4f}, start={history[0]:.4f}"
    )
