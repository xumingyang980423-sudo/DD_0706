from __future__ import annotations

import runpy
import sys
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
ISAAC_LAB_ROOT = Path(r"E:\Issac_sim\IsaacLab")
RL_GAMES_SCRIPT_DIR = ISAAC_LAB_ROOT / "scripts" / "reinforcement_learning" / "rl_games"
RL_GAMES_PLAY = RL_GAMES_SCRIPT_DIR / "play.py"

sys.path.insert(0, str(SANDBOX_ROOT / "source" / "isaaclab_stage3"))
sys.path.insert(0, str(RL_GAMES_SCRIPT_DIR))

import isaaclab_stage3  # noqa: F401,E402
import torch  # noqa: E402


def _no_compile(fn=None, *args, **kwargs):
    if fn is None:
        return lambda wrapped: wrapped
    return fn


torch.compile = _no_compile

runpy.run_path(str(RL_GAMES_PLAY), run_name="__main__")
