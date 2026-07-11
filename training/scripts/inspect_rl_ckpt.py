import sys
import torch

p = sys.argv[1]
c = torch.load(p, map_location="cpu", weights_only=False)
m = c.get("model", c)
print(f"file={p}")
print(f"optimizer={type(c.get('optimizer')).__name__}")
for k in sorted(m.keys()):
    t = m[k]
    if hasattr(t, "shape"):
        print(f"  {k}: {tuple(t.shape)}")
if isinstance(c.get("optimizer"), dict):
    for gname, group in c["optimizer"].items():
        if isinstance(group, dict):
            for pname, state in group.items():
                if isinstance(state, dict):
                    for sk, sv in state.items():
                        if hasattr(sv, "shape"):
                            print(f"  opt[{gname}][{pname}][{sk}]: {tuple(sv.shape)}")
