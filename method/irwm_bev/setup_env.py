"""Create a compatible overlay venv without changing existing environments."""
from pathlib import Path
import json
import subprocess
import sys
import sysconfig

root = Path(__file__).resolve().parents[3]
source = root/'ENV/wmmpc-full-diffusion'
dest = root/'ENV/wmmpc-irwm-bev'
assert str(Path(sys.prefix).resolve()) == str(source.resolve()), (sys.prefix,source)
if dest.exists():
    raise FileExistsError(dest)
subprocess.run([sys.executable,'-m','venv','--without-pip',str(dest)],check=True)
source_packages = Path(sysconfig.get_paths()['purelib'])
target_packages = dest/'lib/python3.10/site-packages'
# Explicit read-only package reuse; installing in the overlay takes precedence.
(target_packages/'wmmpc_base.pth').write_text(str(source_packages)+'\n')
metadata = {'base_environment':str(source),'new_environment':str(dest),
            'package_reuse':'explicit .pth; base is not modified',
            'compatible_entrypoints':['method.full_diffusion','method.irwm_bev'],
            'legacy_mmcv_irwm':'not installed; B2D adaptation uses native PyTorch'}
(dest/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
with (dest/'requirements.lock.txt').open('w') as f:
    subprocess.run([str(dest/'bin/python'),'-m','pip','freeze'],stdout=f,check=True)
subprocess.run([str(dest/'bin/python'),'-c',
    'import torch, torchvision, diffusers, timm, cv2, laspy; from method.full_diffusion.garage import load_config; c=load_config(); print(torch.__version__, c.min_x,c.max_x,c.min_y,c.max_y)'],check=True)
print(json.dumps(metadata),flush=True)
