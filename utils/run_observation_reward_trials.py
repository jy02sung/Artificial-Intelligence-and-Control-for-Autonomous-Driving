"""Six isolated 200k trials on one frozen traffic scenario; never changes main."""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as E
import numpy as np
from utils import av_hv_release_check as a
from utils.ic_campaign import Checkpoints, RolloutBuffer
from env.observation_reward_trials import TrialEnv, CONDITIONS
from env.navigation_observation import RouteNavigation
import env.ic_experiment as ic

ROOT=a.ROOT
OUT=ROOT/'results/observation_reward_trials_20261114'
BRANCH='codex/av-observation-reward-20261114'
DOC=ROOT/'docs/experiments/ic-observation-reward-results-20261114.md'
SEED=11
EVAL_SEEDS=(19101,19102,19103)
SOURCE_FILES=('env/sumo_env.py','env/ic_experiment.py','env/navigation_observation.py','env/navigation_clarity.py','env/observation_reward_trials.py','env/mdp_config.py','env/road_config.py','train.py','algorithms/ppo.py','utils/buffer.py','utils/networks.py','utils/run_observation_reward_trials.py')

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');os.replace(tmp,path)

def read(path):return json.loads(path.read_text(encoding='utf8'))

def policy_hash(agent):
    h=hashlib.sha256()
    for key,tensor in agent.policy.state_dict().items():h.update(key.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def make_env(condition,evaluation=False,none=False):
    inp=OUT/'inputs'/('none' if none else 'reference')
    tree=E.parse(inp/'network.net.xml')
    edges={e.get('id'):e for e in tree.findall('edge') if not e.get('function')}
    routes={r.get('id'):r.get('edges').split() for r in E.parse(inp/'highway.rou.xml').findall('route') if r.get('id').startswith('route_')}
    connections=[(c.get('from'),c.get('to'),int(c.get('fromLane')),int(c.get('toLane'))) for c in tree.findall('connection') if c.get('from') in edges and c.get('to') in edges]
    lengths={k:max(float(l.get('length')) for l in e.findall('lane')) for k,e in edges.items()}
    internals={l.get('id'):float(l.get('length')) for e in tree.findall('edge') if e.get('function')=='internal' for l in e.findall('lane')}
    link_lengths={}
    for c in tree.findall('connection'):
        k=(c.get('from'),c.get('to'));link_lengths[k]=max(link_lengths.get(k,0.),internals.get(c.get('via'),0.))
    road=dict(a.b.ROAD_NEW,routes=routes,connections=connections,edges=[(k,e.get('from'),e.get('to'),len(e.findall('lane'))) for k,e in edges.items()],solid_lane_edges=[k for k,e in edges.items() if any(l.get('changeLeft') or l.get('changeRight') for l in e.findall('lane'))])
    nav=RouteNavigation({k:len(e.findall('lane')) for k,e in edges.items()},connections,lengths,horizon=300.,link_lengths=link_lengths)
    ic.ZIP=();ic.RAMPS={k for k,e in edges.items() if e.get('type','').endswith('_link')}
    m=read(a.r.sources('baseline',22)/'manifest.json')
    sim=dict(m['simulation'],max_steps=54000 if evaluation else 4500,warmup_steps=4500)
    env=TrialEnv(str(inp/'run.sumocfg'),road,dict(m['ego'],depart_lane='best',depart_speed='max'),sim,a.b.OBSERVATION,m['action'],m['reward'],dict(a.b.TRAFFIC,prefill={'enabled':False}),SEED,nav,condition)
    env.insert_wait_steps=54000
    return env

class Log(Checkpoints):
    def __init__(self,directory,agent,budget,env):
        super().__init__(directory,agent,budget)
        self.env=env;self.started=time.time();self.episodes=0
        self.next_checkpoint=min(100000,budget)
        if self.writer is None:raise RuntimeError('TensorBoard required for experiment provenance')

    def progress(self,step):
        if (OUT/'STOP').exists():raise RuntimeError('Stopped because another condition failed')
        save(self.directory/'progress.json',dict(stage='training',pid=os.getpid(),steps=step,target=self.budget,episodes=self.episodes,wall_seconds=time.time()-self.started,route_overrides=self.env.route_override_count))

    def log_episode(self,step,ep_return,ep_length):
        super().log_episode(step,ep_return,ep_length);self.episodes+=1;self.progress(step)

    def log_train(self,step,metrics,verbose=False):
        for f,h in read(self.directory/'manifest.json')['source_hashes'].items():
            assert a.b.digest(ROOT/f)==h,('source changed during training',f)
        assert all(np.isfinite(v) for k,v in metrics.items() if k!='ep_rew_mean' and isinstance(v,(int,float)))
        # The base checkpoint class advances by20k; set the intended100k boundary explicitly.
        next_before=self.next_checkpoint
        super().log_train(step,metrics,False)
        if step>=next_before:self.next_checkpoint=next_before+100000
        self.progress(step)
        assert self.env.route_override_count==0

def evaluate(condition,checkpoint,directory,smoke=False):
    m=read(directory/'manifest.json')
    agent=a.b.PPO(obs_dim=63,act_dim=2,**m['hp']);agent.load(str(checkpoint))
    rows=[]
    for none in (False,True):
        env=make_env(condition,evaluation=True,none=none)
        if smoke:env.max_steps=50  # Pipeline verification only; never a performance score.
        try:
            routes=['route_01'] if smoke else sorted(env.routes)
            seeds=EVAL_SEEDS[:1] if none or smoke else EVAL_SEEDS
            for route in routes:
                for seed in seeds:
                    row=dict(condition=condition,checkpoint=checkpoint.name,route=route,seed=seed,traffic='none' if none else 'reference')
                    try:obs,_=env.reset(seed=seed,options={'route':route})
                    except RuntimeError as exc:
                        if '삽입 실패' not in str(exc):raise
                        rows.append(dict(row,outcome='insertion_failure',error=str(exc)));continue
                    trace=a.b.TimeoutTrace();teleports=0
                    while True:
                        if env.step_count%5==0:trace.sample(env)
                        obs,_,done,cut,info=env.step(agent.predict(obs,deterministic=True))
                        teleports+=a.b.traci.simulation.getStartingTeleportNumber()
                        if done or cut:break
                    rows.append(dict(row,outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit',seconds=env.step_count*.2,teleports=teleports,route_overrides=env.route_override_count,diagnostic=trace.result() if cut else None))
                    save(directory/'progress.json',dict(stage='evaluation',pid=os.getpid(),checkpoint=checkpoint.name,completed=len(rows),target=2 if smoke else 64))
        finally:env.close()
    save(directory/(checkpoint.stem+'-evaluation.json'),rows)
    return rows

def train(condition,smoke=False):
    a.b.torch.set_num_threads(1)
    directory=OUT/('smoke' if smoke else 'runs')/condition
    if (directory/'complete.json').exists():return read(directory/'complete.json')
    if directory.exists():raise RuntimeError('Incomplete run preserved; no implicit resume: '+str(directory))
    directory.mkdir(parents=True)
    manifest=read(OUT/'manifest.json')
    for f,h in manifest['source_hashes'].items():assert a.b.digest(ROOT/f)==h,('source changed',f)
    budget=32 if smoke else 200000
    hp=dict(read(a.r.sources('baseline',22)/'manifest.json')['hp'],seed=SEED,device='cpu')
    env=make_env(condition);agent=a.b.PPO(obs_dim=63,act_dim=2,buffer=RolloutBuffer(),**hp)
    save(directory/'manifest.json',dict(condition=condition,hp=hp,budget=budget,initial_tensor_sha=policy_hash(agent),simulation=dict(step_length=.2,warmup_steps=4500,max_steps=4500),reward=env.rw,action_safety=env.lc_safety_enabled,observation_dim=63,source_commit=manifest['commit'],source_hashes=manifest['source_hashes']))
    agent.save(str(directory/'initial.pt'))
    logger=Log(directory,agent,budget,env)
    try:
        agent.learn(env,budget,logger=logger,eval_interval=0,exact_budget=True)
        assert agent.num_timesteps==budget and env.route_override_count==0
        agent.save(str(directory/'model.pt'))
        save(directory/'training-complete.json',dict(steps=budget,episode_seeds=env.episode_seeds,model_sha=a.b.digest(directory/'model.pt')))
    finally:env.close();logger.close()
    rows=[]
    for checkpoint in sorted(directory.glob('policy-*.pt'),key=lambda p:int(p.stem.split('-')[1])):
        rows.extend(evaluate(condition,checkpoint,directory,smoke))
    result=dict(condition=condition,steps=budget,model_sha=a.b.digest(directory/'model.pt'),evaluations=len(rows))
    save(directory/'complete.json',result);save(directory/'progress.json',dict(stage='complete',**result))
    return result

def publish():
    rows=[]
    for p in sorted((OUT/'runs').glob('*/policy-*-evaluation.json')):rows.extend(read(p))
    simple=[{k:r.get(k) for k in ('condition','checkpoint','route','seed','traffic','outcome','seconds','teleports','route_overrides')} for r in rows]
    if simple:
        with DOC.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(simple[0]));w.writeheader();w.writerows(simple)
    lines=['# 관측·보상 단독 비교 실행 기록','','수정된 공통 검출기,경로안내300m,기본15,000대/h,16경로,seed11,각200,000스텝. 모두동일초기가중치에서시작하며강제보정은없다. 무거운원시자료와정책은로컬 `results/observation_reward_trials_20261114/`에보관한다.','','| 조건 | 진행 | 마지막 학습 스텝 |','|---|---|---:|']
    for c in CONDITIONS:
        p=OUT/'runs'/c/'progress.json';r=read(p) if p.exists() else {}
        lines.append(f"|{c}|{r.get('stage','대기')}|{r.get('steps','')}|")
    lines+=['','| 조건 | 체크포인트 | 교통 | 완주/전체 | 충돌 | 기타 |','|---|---|---|---:|---:|---:|']
    for condition,checkpoint,traffic in sorted({(r['condition'],r['checkpoint'],r['traffic']) for r in rows}):
        group=[r for r in rows if (r['condition'],r['checkpoint'],r['traffic'])==(condition,checkpoint,traffic)]
        arrived=sum(r['outcome']=='arrival' for r in group);collided=sum(r['outcome']=='collision' for r in group)
        lines.append(f'|{condition}|{checkpoint}|{traffic}|{arrived}/{len(group)}|{collided}|{len(group)-arrived-collided}|')
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if shutil.disk_usage(OUT).free<20*1024**3:raise RuntimeError('Insufficient free disk')
        assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==BRANCH
        for mode in ('none','reference'):
            dest=OUT/'inputs'/mode;dest.mkdir(parents=True,exist_ok=True)
            for name in ('network.net.xml','highway.rou.xml','run.sumocfg'):shutil.copy2(ROOT/'results/neighbor_fix_20261114/inputs'/mode/name,dest/name)
        save(OUT/'manifest.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes={p:a.b.digest(ROOT/p) for p in SOURCE_FILES},inputs={m:{n:a.b.digest(OUT/'inputs'/m/n) for n in ('network.net.xml','highway.rou.xml','run.sumocfg')} for m in ('none','reference')},seed=SEED,eval_seeds=EVAL_SEEDS,conditions=CONDITIONS,workers=6))
        if not args.smoke:
            assert all((OUT/'smoke'/c/'complete.json').exists() for c in CONDITIONS),'Smoke gate required'
            hashes={read(OUT/'smoke'/c/'manifest.json')['initial_tensor_sha'] for c in CONDITIONS};assert len(hashes)==1
            for c in CONDITIONS:assert read(OUT/'smoke'/c/'manifest.json')['source_hashes']==read(OUT/'manifest.json')['source_hashes'],'Smoke used different code'
        save(OUT/'status.json',dict(stage='smoke' if args.smoke else 'running',pid=os.getpid()))
        with cf.ProcessPoolExecutor(max_workers=6) as pool:
            futures={pool.submit(train,c,args.smoke):c for c in CONDITIONS}
            for future in cf.as_completed(futures):
                try:result=future.result()
                except Exception:
                    (OUT/'STOP').write_text('Peer trial failed',encoding='utf8')
                    raise
                print(result,flush=True)
                if not args.smoke:publish()
        if not args.smoke:
            hashes={read(OUT/'runs'/c/'manifest.json')['initial_tensor_sha'] for c in CONDITIONS};assert len(hashes)==1
        save(OUT/'status.json',dict(stage='smoke_complete' if args.smoke else 'complete'))
    except Exception as exc:save(OUT/'status.json',dict(stage='error',error=repr(exc)));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
