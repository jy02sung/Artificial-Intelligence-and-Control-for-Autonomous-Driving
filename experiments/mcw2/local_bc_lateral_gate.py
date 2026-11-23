"""Gate the last targeted run against the conservative BC on their common screening seeds."""
from pathlib import Path
import os,time,json,sys
from . import local_bc_run as r
from .local_bc_ppo import D
if __name__=='__main__':
    while not (D/'kinlat3_screen.json').exists():time.sleep(3)
    c=r.summary('kinlat3','screen');a,k=c['arrival'],c['collision']
    passed=(a>=74 and k<=4) or (a>=73 and k<=2)
    print(dict(screen=c, advance=passed),flush=True)
    if passed:
        import subprocess
        env=os.environ.copy();env['BC_GROUP']='kinlat3';env['BC_PPO_FOCUS']='1'
        args={'base':['baseline'],'train':['train','48177','0.000008','12','1'],'select':['select','48177','12']}[sys.argv[1]]
        subprocess.run([sys.executable,'-u','-m','experiments.mcw2.local_bc_run',*args],check=True,env=env)
