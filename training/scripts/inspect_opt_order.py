import sys
import torch

c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = c["model"]
o = c["optimizer"]
keys = list(m.keys())
print("model keys order:")
for k in keys:
    print(" ", k, tuple(m[k].shape))
print("optimizer state:")
for i in range(len(o["state"])):
    ea = o["state"][i]["exp_avg"]
    print(f"  [{i}] {tuple(ea.shape)}")
