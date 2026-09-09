"""RSL-RL PPO training configuration for the Synria pick-and-place tasks.

Hyperparameters are tuned for a 6DOF manipulation task on an RTX 5090
(4096 parallel environments, 30-second episodes, 2-step decimation).

Reference baseline: Isaac Lab's Franka-Cabinet and Franka-Lift tasks, adjusted
for the Synria arm's shorter reach (0.65 m) and lighter payload (1 kg).

Three identical configs are registered — one per game — so each task can be
trained independently or together in a sweep. Inherit from
``SynriaPickPlaceBasePPORunnerCfg`` and override ``experiment_name`` to add a
new task variant without duplicating hyperparameters.

Usage inside Isaac Lab training script::

    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import (
        SynriaChessPickPlacePPORunnerCfg,
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# RSL-RL imports — gated; this module is import-safe without Isaac Lab.
# ---------------------------------------------------------------------------

try:
    from isaaclab.utils import configclass  # type: ignore[import-not-found]
    from isaaclab_rl.rsl_rl import (  # type: ignore[import-not-found]
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
    )

    _ISAACLAB_RL_AVAILABLE = True
except ImportError:  # pragma: no cover - hardware-gated import
    _ISAACLAB_RL_AVAILABLE = False

    def configclass(cls: type) -> type:  # type: ignore[no-redef]
        """No-op stand-in for ``isaaclab.utils.configclass`` on non-RTX machines."""
        return cls

    # Minimal stubs so the module and its subclasses parse on non-RTX machines.
    # Stubs must accept keyword arguments (matching the real RSL-RL dataclass API)
    # and store them as instance attributes so tests can read back config values.

    class _KwargStub:
        """Base stub: accepts any kwargs and stores them as instance attributes."""

        def __init__(self, **kwargs: object) -> None:
            # Apply class-level defaults first, then instance overrides.
            for k, v in vars(type(self)).items():
                if not k.startswith("_"):
                    setattr(self, k, v)
            for k, v in kwargs.items():
                setattr(self, k, v)

    class RslRlOnPolicyRunnerCfg(_KwargStub):  # type: ignore[no-redef]
        """Stub runner config."""

        experiment_name: str = "synria_pickplace_base"
        run_name: str = ""
        num_steps_per_env: int = 24
        max_iterations: int = 1500
        save_interval: int = 50
        empirical_normalization: bool = True
        logger: str = "tensorboard"
        neptune_project: str = ""
        device: str = "cuda"

        def to_dict(self) -> dict[str, object]:
            return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    class RslRlPpoActorCriticCfg(_KwargStub):  # type: ignore[no-redef]
        """Stub actor-critic network config."""

        init_noise_std: float = 1.0
        actor_hidden_dims: list = [512, 256, 128]  # noqa: RUF012
        critic_hidden_dims: list = [512, 256, 128]  # noqa: RUF012
        activation: str = "elu"

    class RslRlPpoAlgorithmCfg(_KwargStub):  # type: ignore[no-redef]
        """Stub PPO algorithm config."""

        value_loss_coef: float = 1.0
        use_clipped_value_loss: bool = True
        clip_param: float = 0.2
        entropy_coef: float = 0.005
        num_learning_epochs: int = 5
        num_mini_batches: int = 4
        learning_rate: float = 3e-4
        schedule: str = "adaptive"
        gamma: float = 0.99
        lam: float = 0.95
        desired_kl: float = 0.01
        max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Shared network architecture
# ---------------------------------------------------------------------------

#: MLP hidden dims for both actor and critic.
#: [512, 256, 128] gives ~500 k parameters per head — appropriate for the
#: 14-float observation space (joint state + ee pose + task encoding).
_ACTOR_CRITIC_CFG = RslRlPpoActorCriticCfg(
    # 0.5 rad: 1.0 made exploration so violent the arm never found contact
    # rewards, and entropy pressure kept noise pinned at ~0.96 all run.
    init_noise_std=0.5,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
)

#: PPO algorithm hyperparameters tuned for dense-reward manipulation.
#: GAE lambda 0.95 + gamma 0.99 stabilise long (30 s) pick-place episodes.
_PPO_ALGORITHM_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    # 0.0002, was 0.001: at v17-20k the noise std sat at 1.11 while the
    # skill was largely learned — training rollouts paid a heavy
    # exploration tax (cycles 0.89 noisy vs a far stronger deterministic
    # policy underneath). Endgame anneal: let the std collapse and
    # harvest the learned behavior; learning continues at lower entropy.
    # E10: 0.0005, was 0.0 (E4). E4's zero-entropy harvest collapsed std to
    # 0.02 — correct for harvesting a learned skill, but now two NEW skills
    # (scratch-context transport and deliberate release) must be DISCOVERED,
    # and attach finally makes their +30 completions reliably reachable.
    entropy_coef=0.0005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=3e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


# ---------------------------------------------------------------------------
# Base runner config
# ---------------------------------------------------------------------------


@configclass
class SynriaPickPlaceBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Base RSL-RL PPO runner config for all Synria pick-and-place variants.

    Must be decorated with ``@configclass``: the parent is a dataclass whose
    generated ``__init__`` writes the *parent* defaults (``MISSING`` sentinels
    included) onto the instance, shadowing any un-decorated subclass overrides.

    Subclasses override ``experiment_name`` to separate TensorBoard runs and
    checkpoint directories by game.

    Tuning notes:
    - ``num_steps_per_env=24`` balances credit assignment (30 s episodes at
      60 Hz / decimation 2 = 900 policy steps → 24 gives ~38 rollouts/episode)
      with GPU memory: 4096 envs × 24 steps × 14-dim obs ≈ 1.4 M floats.
    - ``max_iterations=1500`` targets ~6 B environment steps on 4096 envs,
      matching the Isaac Lab Franka-Lift training budget. Adjust down for
      smoke-test / quick validation (use ``--max_iterations 10``).
    - ``save_interval=50`` keeps disk use manageable; final checkpoint is
      always saved regardless of interval.
    """

    #: RSL-RL sub-configs — set at class level so ``configclass`` picks them up.
    policy: RslRlPpoActorCriticCfg = _ACTOR_CRITIC_CFG
    algorithm: RslRlPpoAlgorithmCfg = _PPO_ALGORITHM_CFG

    #: Training run metadata
    experiment_name: str = "synria_pickplace_base"
    run_name: str = ""  # auto-generated from timestamp if empty

    #: Observation routing (rsl-rl 3.x): actor and critic both consume the
    #: env's "policy" observation group.
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["policy"]}  # noqa: RUF012

    #: Rollout and training loop
    num_steps_per_env: int = 24
    max_iterations: int = 1500

    #: Checkpointing
    save_interval: int = 50
    empirical_normalization: bool = True

    #: Logging
    logger: str = "tensorboard"
    neptune_project: str = "synria-pickplace"


# ---------------------------------------------------------------------------
# Per-game runner configs
# ---------------------------------------------------------------------------


@configclass
class SynriaLudoPickPlacePPORunnerCfg(SynriaPickPlaceBasePPORunnerCfg):
    """PPO config for the Synria + Ludo pick-and-place task."""

    experiment_name: str = "synria_ludo_pickplace_v0"


@configclass
class SynriaChessPickPlacePPORunnerCfg(SynriaPickPlaceBasePPORunnerCfg):
    """PPO config for the Synria + chess pick-and-place task."""

    experiment_name: str = "synria_chess_pickplace_v0"


@configclass
class SynriaCheckersPickPlacePPORunnerCfg(SynriaPickPlaceBasePPORunnerCfg):
    """PPO config for the Synria + checkers pick-and-place task."""

    experiment_name: str = "synria_checkers_pickplace_v0"


# ---------------------------------------------------------------------------
# Registry: task_id → runner config
# ---------------------------------------------------------------------------

#: Map gym task ID → PPO runner config.
#: Used by the training entry point to look up the right config.
TASK_TRAIN_CFG: dict[str, type[SynriaPickPlaceBasePPORunnerCfg]] = {
    "Synria-Ludo-PickPlace-v0": SynriaLudoPickPlacePPORunnerCfg,
    "Synria-Chess-PickPlace-v0": SynriaChessPickPlacePPORunnerCfg,
    "Synria-Checkers-PickPlace-v0": SynriaCheckersPickPlacePPORunnerCfg,
}

__all__ = [
    "SynriaPickPlaceBasePPORunnerCfg",
    "SynriaLudoPickPlacePPORunnerCfg",
    "SynriaChessPickPlacePPORunnerCfg",
    "SynriaCheckersPickPlacePPORunnerCfg",
    "TASK_TRAIN_CFG",
]
