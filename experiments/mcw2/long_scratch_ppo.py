"""Fixed-budget 600k PPO learning curve from the scratch DAgger2 student."""
import csv
import json
import math
import random
import multiprocessing as mp
from types import SimpleNamespace
from time import perf_counter
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from . import local_scratch_driver as d

BASE=d.OUT
OUT=BASE.parent/"scratch_parallel_ppo600k_20261124"
MILESTONES=range(50000,600001,50000)
def schedule():
    rng=random.Random(d.SEED+3)
    for group in range(20):
        cases=d.cases(198000000-group*1000000,40)
        for i in range(0,len(cases),16):
            block=cases[i:i+16]
            rng.shuffle(block)
            yield from block

N = 4

def worker(pipe, config, schedule):
    from experiments.mcw2 import local_scratch_driver as d
    d.torch.set_num_threads(1)
    args = SimpleNamespace(**(d.base.x.SPEC['args'] | {'seed': d.SEED}))
    cfg = Path(config)
    road, nav = d.m.load_network_metadata(cfg.parent/'network.net.xml',cfg.parent/'highway.rou.xml',args.nav_horizon)
    env = d.m.make_env(cfg,road,nav,args,*d.m.build_configs(args))
    env.eval_mode = False
    cases = iter(schedule)
    def reset():
        route, seed = next(cases)
        return env.reset(seed=seed,options={'route':route})[0]
    try:
        pipe.send(reset())
        while True:
            action = pipe.recv()
            if action is None: break
            obs, rew, term, cut, _ = env.step(action)
            done = term or cut
            pipe.send((reset() if done else obs, rew, done))
    finally:
        env.close()
        pipe.close()

def train():
    device='cpu'
    from experiments.mcw2 import local_scratch_driver as d
    torch = d.torch
    torch.set_num_threads(1)
    start=perf_counter()
    folder=OUT
    folder.mkdir(parents=True,exist_ok=True)
    args = d.base.x.SPEC['args']
    cfg=d.m.prepare_scenario(args['stage'],folder/'map',color_by_type=args['color_by_type'])
    hp=d.base.x.SPEC['hparams'] | dict(seed=d.SEED,device=device,lr=2.5e-5)
    agent=d.m.PPO(obs_dim=47,act_dim=2,buffer=d.m.RolloutBuffer(),**hp)
    parent=d.weights('dagger2')
    agent.load(str(parent))
    assert next(agent.policy.parameters()).device.type==device
    agent.policy.train()
    torch.manual_seed(d.SEED+3)
    np.random.seed(d.SEED+3)
    order=list(schedule())
    d.base.x.write_json(OUT/'protocol.json',dict(workers=N,device=device,seed=d.SEED,lr=hp['lr'],
        parent=d.digest(parent),collision=args['collision_penalty'],eval_episodes=1600,total_steps=600000,
        milestones=list(MILESTONES),rollout=2048,last_rollout_per_milestone=848,timeout_terminal=True))
    ctx=mp.get_context('spawn')
    workers=[]; pipes=[]; rows=[]
    try:
        for i in range(N):
            conn,child=ctx.Pipe()
            proc=ctx.Process(target=worker,args=(child,str(cfg),order[i::N]))
            proc.start();child.close()
            workers.append(proc);pipes.append(conn)
        obs=np.stack([conn.recv() for conn in pipes])
        ready=perf_counter()
        steps=0
        while steps<600000:
            target=(steps//50000+1)*50000
            batch=min(2048,target-steps)
            t0=perf_counter()
            chunks=[d.m.RolloutBuffer() for _ in pipes]
            for _ in range(batch//N):
                with torch.no_grad():
                    actions,logp,values=agent.policy.sample(torch.as_tensor(obs,device=device))
                    sample=torch.cat((actions.reshape(N,2),logp.reshape(N,1),values.reshape(N,1)),dim=1).cpu().numpy()
                for i,conn in enumerate(pipes):conn.send(sample[i,:2])
                states=[]
                for i,conn in enumerate(pipes):
                    state,reward,done=conn.recv()
                    chunks[i].add(obs[i],sample[i,:2],sample[i,2],reward,sample[i,3],done)
                    states.append(state)
                obs=np.stack(states)
            with torch.no_grad():
                last=agent.policy(torch.as_tensor(obs,device=device))[-1].reshape(N).cpu().numpy()
            gae=[buf.compute_gae(last[i],agent.gamma,agent.gae_lambda) for i,buf in enumerate(chunks)]
            agent.buffer.clear()
            for buf in chunks:
                for field in ('obs','actions','log_probs','rewards','values','dones'):
                    getattr(agent.buffer,field).extend(getattr(buf,field))
            assert len(agent.buffer)==batch
            t1=perf_counter()
            metrics=agent.update(*[np.concatenate(v) for v in zip(*gae)])
            if device=='cuda':torch.cuda.synchronize()
            t2=perf_counter()
            assert all(map(math.isfinite,metrics.values()))
            rows.append([t1-t0,t2-t1])
            steps+=batch
            if steps==target or steps==32768:
                tmp=OUT/'pending.pt'
                agent.save(str(tmp))
                tmp.replace(OUT/f'{steps}.pt')
            d.base.x.write_json(OUT/'progress.json',dict(steps=steps,lr=hp['lr'],seconds=perf_counter()-start))
    finally:
        for conn in pipes:
            try:conn.send(None)
            except (BrokenPipeError,OSError):pass
            conn.close()
        for proc in workers:proc.join(15)

def init_eval(name):
    d.OUT=OUT
    d.init(name,False)

def evaluate():
    cases=d.cases(200000000,100)
    rows_summary=[]
    for target in MILESTONES:
        name=str(target)
        while not (OUT/(name+".pt")).exists():
            time.sleep(5)
        dest=OUT/(name+"_eval.json")
        if dest.exists():
            data=json.loads(dest.read_bytes())
        else:
            with ProcessPoolExecutor(6,initializer=init_eval,initargs=(name,)) as pool:
                rows=[]
                for row in pool.map(d.episode,[(r,s,-1,None) for r,s in cases],chunksize=2):
                    rows.append(row)
                    if len(rows)%80==0:
                        d.base.x.write_json(OUT/"eval_progress.json",dict(steps=target,finished=len(rows),total=len(cases)))
            assert [(r["route"],r["seed"]) for r in rows]==cases
            data=dict(rows=rows,counts=dict(Counter(r["outcome"] for r in rows)),sha256=d.digest(OUT/(name+".pt")))
            d.base.x.write_json(dest,data)
        rows_summary.append(dict(steps=target,**{k:data["counts"].get(k,0) for k in ("arrival","collision","timeout")}))
        with (OUT/"rates.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=rows_summary[0].keys())
            writer.writeheader();writer.writerows(rows_summary)
        print(rows_summary[-1],flush=True)
if __name__=="__main__":
    train() if sys.argv[1]=="train" else evaluate()
