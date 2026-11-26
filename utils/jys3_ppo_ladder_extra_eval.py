"""Two extra deterministic-evaluation workers for a running clean-ladder manager."""
import argparse
import ctypes
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import jys3_ppo_clean_ladder as ladder


def candidates():
    # Take work from the back of the main queue to avoid repeating its active shards.
    for arm,seeds in reversed(list(ladder.SEEDS.items())):
        for seed in reversed(seeds):
            out=ladder.folder(arm,seed)
            if not out.is_dir():
                continue
            marks=list(ladder.marks(arm))
            panels=[]
            if all(ladder.panel(arm,seed,n,"screen").exists() for n in marks):
                steps={ladder.choose(arm,seed)}
                if arm=="M2":
                    steps.add(ladder.choose(arm,seed,250000))
                panels.extend((n,h) for n in sorted(steps,reverse=True) for h in reversed(ladder.HOLDS))
            panels.extend((n,"screen") for n in reversed(marks))
            for n,h in panels:
                if ladder.panel(arm,seed,n,h).exists() or not (ladder.source(arm,seed)/f"policy_{n}.pt").is_file():
                    continue
                for rep in reversed(range(ladder.old.PANELS[h][1])):
                    if not (out/"parts"/f"{n}_{h}_{rep}.json").exists():
                        yield arm,seed,n,h,rep


def evaluate(key):
    a,s,n,h,rep=key
    out=ladder.folder(a,s)
    parts=out/"parts"
    parts.mkdir(parents=True,exist_ok=True)
    target=parts/f"{n}_{h}_{rep}.json"
    if target.exists():
        return False
    with TemporaryDirectory(dir=out,prefix="helper-") as tmp:
        ladder.c.evaluate(a,s,n,h,rep,directory=Path(tmp),checkpoint=ladder.source(a,s)/f"policy_{n}.pt",
                          lateral=a in ("M3","M4","M5","M5-FT"))
        src=Path(tmp)/"parts"/target.name
        rows=ladder.read(src)
        base=ladder.old.PANELS[h][0]
        assert [r["seed"] for r in rows]==[base+rep*100+i for i in range(1,17)]
        try:
            os.link(src,target)  # Publish a complete file without replacing existing work.
        except FileExistsError:
            assert rows==ladder.read(target)
            return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager-pid",required=True,type=int)
    pid=parser.parse_args().manager_pid
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:
        return
    try:
        running={}
        with ProcessPoolExecutor(2) as pool:
            while kernel.WaitForSingleObject(ctypes.c_void_p(handle),0)==258 and not (ladder.ROOT/"finished.json").exists():
                for key in candidates():
                    if len(running)>=2:
                        break
                    if key not in running.values():
                        running[pool.submit(evaluate,key)]=key
                if running:
                    done,_=wait(running,timeout=5,return_when=FIRST_COMPLETED)
                    for job in done:
                        key=running.pop(job)
                        print("published" if job.result() else "already available",*key,flush=True)
                else:
                    time.sleep(5)
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


if __name__=="__main__":
    main()
