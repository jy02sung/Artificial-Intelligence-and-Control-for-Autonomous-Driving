"""Evaluate only the three approved checkpoints; never starts new training."""
import argparse,ctypes,json,subprocess,sys,time
from utils.curriculum_data import OUT,ROOT,save

def live(pid):
    h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    if h:ctypes.windll.kernel32.CloseHandle(h)
    return bool(h)

def run(name):
    folder=OUT/'runs'/name
    for step in (10240,20480,50000):
        while not (folder/f'policy-{step}.pt').exists():
            lock=folder/'active.lock'
            if not lock.exists() or not live(int(lock.read_text())):raise RuntimeError(f'Training stopped before {name}/{step}; no automatic restart')
            time.sleep(20)
        print(dict(name=name,evaluation_step=step),flush=True)
        subprocess.run([sys.executable,'-X','utf8','-u','-m','utils.evaluate_merge_adapter','--name',name,'--step',str(step)],cwd=ROOT,check=True)
        save(OUT/f'{name}-evaluation-watch.json',dict(last_complete=step,done=step==50000))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['F1-seed11','F2-seed11']);run(p.parse_args().name)
