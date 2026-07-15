"""Unit test for temperature scaling (plan Fix 32).

Builds logits with a KNOWN miscalibration and checks fit_temperature recovers it.
If a model's logits are the well-calibrated logits multiplied by k, the optimal
temperature is k. Run: python test_calibration.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from hetero_gnn import expected_calibration_error


def fit_temperature_on_logits(logits, ys, max_iter=200):
    """Same optimiser as hetero_gnn.fit_temperature, minus the data loader."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), ys)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def main():
    torch.manual_seed(0)
    n = 4000

    # Well-calibrated base logits: P(y=1) = sigmoid(z)
    z = torch.randn(n) * 1.5
    p = torch.sigmoid(z)
    ys = torch.bernoulli(p).long()
    base = torch.stack([torch.zeros(n), z], dim=1)   # logit difference = z

    print(f"{'true k':>8} {'recovered T':>12} {'ECE before':>11} {'ECE after':>10}")
    ok = True
    for k in [0.5, 1.0, 2.0, 3.0]:
        logits = base * k                            # optimal temperature is k
        T = fit_temperature_on_logits(logits, ys)

        pb = F.softmax(logits, 1)[:, 1].numpy()
        pa = F.softmax(logits / T, 1)[:, 1].numpy()
        y = ys.numpy()
        eb = expected_calibration_error(pb, y)
        ea = expected_calibration_error(pa, y)
        print(f"{k:8.2f} {T:12.3f} {eb:11.4f} {ea:10.4f}")
        if abs(T - k) / k > 0.15:
            ok = False
            print(f"    ^ FAIL: expected T ~= {k}")
        # Only demand an improvement where there was real miscalibration to fix.
        # On an already-calibrated model (k=1) ECE just jitters at the 1e-3
        # level, which is sampling noise rather than damage.
        if eb > 0.02 and ea > eb:
            ok = False
            print("    ^ FAIL: calibration made ECE worse")
        if eb <= 0.02 and ea > 0.02:
            ok = False
            print("    ^ FAIL: calibration broke an already-calibrated model")

    # AUC must be untouched: temperature is a monotone rescaling
    logits = base * 3.0
    T = fit_temperature_on_logits(logits, ys)
    from sklearn.metrics import roc_auc_score
    a1 = roc_auc_score(ys.numpy(), F.softmax(logits, 1)[:, 1].numpy())
    a2 = roc_auc_score(ys.numpy(), F.softmax(logits / T, 1)[:, 1].numpy())
    print(f"\nAUC before={a1:.6f} after={a2:.6f}  (must be identical)")
    if abs(a1 - a2) > 1e-9:
        ok = False
        print("  FAIL: temperature changed the ranking")

    print("\nVERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
