from pathlib import Path
import numpy as np,pandas as pd,soundfile as sf,yaml

cfg=yaml.safe_load(open("config/base.yaml"))
print(cfg)

root=Path(cfg["data"]["root"])

rows=[]


pd_dir=Path(cfg["data"]["groups"][0])

print(pd_dir)

ehc_dir=Path(cfg["data"]["groups"][1])
print(ehc_dir)

