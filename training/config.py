from dataclasses import dataclass


@dataclass(frozen=True)
class SelfPlayConfig:
    simulations: int = 48
    max_plies: int = 200
    temperature_moves: int = 12
    temperature: float = 1.0
    dirichlet_noise: bool = True
    pawn_prior_fraction: float | None = None


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    value_loss_weight: float = 1.0
    weight_decay: float = 1e-4
    channels: int = 64
    residual_blocks: int = 4
