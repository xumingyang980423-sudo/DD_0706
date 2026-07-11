from __future__ import annotations

import runpy
import sys
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
ISAAC_LAB_ROOT = Path(r"E:\Issac_sim\IsaacLab")
RL_GAMES_SCRIPT_DIR = ISAAC_LAB_ROOT / "scripts" / "reinforcement_learning" / "rl_games"
RL_GAMES_TRAIN = RL_GAMES_SCRIPT_DIR / "train.py"

sys.path.insert(0, str(SANDBOX_ROOT / "source" / "isaaclab_stage3"))
sys.path.insert(0, str(RL_GAMES_SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import isaaclab_stage3  # noqa: F401,E402
import torch  # noqa: E402
from rlgames_checkpoint_utils import inject_sigma_hydra_override  # noqa: E402


def _no_compile(fn=None, *args, **kwargs):
    if fn is None:
        return lambda wrapped: wrapped
    return fn


torch.compile = _no_compile

inject_sigma_hydra_override("Stage3Train")
runpy.run_path(str(RL_GAMES_TRAIN), run_name="__main__")
