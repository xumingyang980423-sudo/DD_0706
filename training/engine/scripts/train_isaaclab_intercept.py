from __future__ import annotations

import runpy
import sys
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
ISAAC_LAB_ROOT = Path(r"E:\Issac_sim\IsaacLab")
RSL_RL_SCRIPT_DIR = ISAAC_LAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
RSL_RL_TRAIN = RSL_RL_SCRIPT_DIR / "train.py"

sys.path.insert(0, str(SANDBOX_ROOT / "source" / "isaaclab_stage3"))
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

import isaaclab_stage3  # noqa: F401,E402

runpy.run_path(str(RSL_RL_TRAIN), run_name="__main__")

