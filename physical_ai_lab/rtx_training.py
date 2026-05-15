"""RTX training workloads for simulation-first robot learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class TrainingRunConfig:
    """Configuration for the synthetic Synria reaching-policy trainer."""

    samples: int = 4096
    epochs: int = 25
    batch_size: int = 256
    learning_rate: float = 0.001
    seed: int = 7
    device: str = "auto"
    report_path: Path = Path("reports/training/synria_reach_policy.json")
    checkpoint_path: Path = Path("runs/rtx_training/synria_reach_policy.pt")


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested


def _make_reach_dataset(torch: Any, samples: int, device: str, seed: int) -> tuple[Any, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    q_goal = (torch.rand((samples, 6), generator=generator, device=device) - 0.5) * 2.4
    q_current = q_goal + torch.randn((samples, 6), generator=generator, device=device) * 0.22
    q_current = torch.clamp(q_current, -1.7, 1.7)

    link_lengths = torch.tensor([0.12, 0.22, 0.18, 0.11, 0.07, 0.04], device=device)
    cumulative = torch.cumsum(q_goal[:, :3], dim=1)
    x = torch.sum(torch.cos(cumulative) * link_lengths[:3], dim=1)
    y = torch.sum(torch.sin(cumulative) * link_lengths[:3], dim=1)
    z = 0.16 + torch.sum(torch.sin(q_goal[:, 1:4]) * link_lengths[1:4], dim=1)
    target_xyz = torch.stack((x, y, z), dim=1)

    features = torch.cat((q_current, target_xyz), dim=1)
    labels = torch.clamp(q_goal - q_current, -0.45, 0.45)
    return features, labels


def train_synria_reach_policy(config: TrainingRunConfig) -> dict[str, object]:
    """Train a compact MLP policy on synthetic reaching data using torch."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised only on missing optional deps
        raise RuntimeError(
            "PyTorch is required for RTX training. Install the ml/GPU stack."
        ) from exc

    device = _resolve_device(torch, config.device)
    torch.manual_seed(config.seed)

    features, labels = _make_reach_dataset(torch, config.samples, device, config.seed)
    split = int(config.samples * 0.8)
    train_ds = TensorDataset(features[:split], labels[:split])
    val_x = features[split:]
    val_y = labels[split:]

    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    model = nn.Sequential(
        nn.Linear(9, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 6),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    started = perf_counter()
    last_train_loss = 0.0
    for _epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch_x.shape[0]
        last_train_loss = total_loss / len(train_ds)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_s = perf_counter() - started

    model.eval()
    with torch.no_grad():
        val_pred = model(val_x)
        val_loss = float(loss_fn(val_pred, val_y).detach().cpu())
        mean_abs_joint_error = float(torch.mean(torch.abs(val_pred - val_y)).detach().cpu())

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": 9,
            "output_dim": 6,
            "config": {
                "samples": config.samples,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
            },
        },
        config.checkpoint_path,
    )

    report = {
        "workload": "synria-synthetic-reaching-policy",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "samples": config.samples,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "train_loss": round(last_train_loss, 6),
        "validation_loss": round(val_loss, 6),
        "mean_abs_joint_delta_error_rad": round(mean_abs_joint_error, 6),
        "elapsed_seconds": round(elapsed_s, 3),
        "checkpoint_path": str(config.checkpoint_path),
    }
    config.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
