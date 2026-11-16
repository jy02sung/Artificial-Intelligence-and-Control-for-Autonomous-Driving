"""Resumable full benchmark; no teacher, no safety override, no early rejection."""
import argparse,collections,json,os,ctypes,time
from pathlib import Path
import numpy as np
import torch
from utils.curriculum_data import OUT,ROOT,s,save,digest,context
from utils.merge_adapter_env import make_merge_env

def run(name,step,smoke=False):
    torch.set_num_threads(1)
    source=OUT/'runs'/name;cp=source/f'policy-{step}.pt';meta=json.loads((source/'manifest.json').read_text())
    for f,h in meta['source_hashes'].items():assert digest(ROOT/f)==h,f
    folder=OUT/'evaluations'/(f'{name}-{step}'+('-smoke' if smoke else ''));folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    lock=folder/'active.lock'
    if lock.exists():
        pid=int(lock.read_text())
        handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid) if os.name=='nt' else None
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            raise RuntimeError(f'Existing live evaluator PID {pid}')
        if os.name!='nt':
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Existing live evaluator PID {pid}')
        lock.rename(folder/f'stale-{pid}-{int(time.time())}.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        signature=dict(checkpoint_sha=digest(cp),mode='mean_argmax',seeds=[79101,79102,79103],cases=2 if smoke else 64,smoke=smoke,evaluation_code_sha=digest(Path(__file__)))
        if (folder/'manifest.json').exists():assert json.loads((folder/'manifest.json').read_text())==signature
        else:save(folder/'manifest.json',signature)
        partial=folder/'partial.json';rows=json.loads(partial.read_text()) if partial.exists() else []
        done_cases={(r['traffic'],r['route'],r['seed']) for r in rows}
        agent=s.a.b.PPO(obs_dim=63,act_dim=2,**meta['hp']);agent.load(str(cp))
        for none in (True,False):
            mode='none' if none else 'reference';env=make_merge_env(evaluation=True,none=none)
            try:
                if smoke:env.max_steps=50
                for route in (['route_13'] if smoke else sorted(env.routes)):
                    for seed in ([79101] if none or smoke else [79101,79102,79103]):
                        if (mode,route,seed) in done_cases:continue
                        obs,_=env.reset(seed=seed,options={'route':route});tail=collections.deque(maxlen=25);counts=collections.Counter();first_risk=None;first_brake=None
                        while True:
                            action=agent.predict(obs,deterministic=True);ctx=context(env,action);t=env.step_count*env.step_length
                            risk=ctx['merge_risk']>0 or ctx['ttc'] is not None and ctx['ttc']<5
                            if risk and first_risk is None:first_risk=t
                            if risk and action[0]<0 and first_brake is None:first_brake=t
                            with torch.no_grad():mu,sd,logits,_=agent.policy(torch.tensor(obs).unsqueeze(0))
                            probs=logits.detach().softmax(-1)[0].tolist()
                            row=dict(t=t,context=ctx,action=action.tolist(),lane_probability_left_keep_right=probs,obs=obs.tolist())
                            obs,_,done,cut,info=env.step(action)
                            v=s.a.b.traci.vehicle;alive=env.ego_id in v.getIDList()
                            row['after_speed']=v.getSpeed(env.ego_id) if alive else None
                            row['actual_acceleration']=(row['after_speed']-ctx['speed'])/env.step_length if alive else None
                            counts['risk_states']+=int(risk);counts['risk_brake_attempts']+=int(risk and action[0]<0)
                            counts['risk_actual_decelerations']+=int(risk and alive and row['actual_acceleration']<0)
                            if ctx['constrained']:
                                counts['decision_steps']+=1;counts['correct_lane_steps']+=int(ctx['needed']==0);counts['toward_requests']+=int(ctx['needed']*action[1]>0)
                            tail.append(row)
                            if done or cut:break
                        outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit'
                        assert env.route_override_count==0
                        rows.append(dict(traffic=mode,route=route,seed=seed,outcome=outcome,seconds=env.step_count*env.step_length,route_overrides=0,metrics=dict(counts),first_diagnostic_risk=first_risk,first_brake_at_risk=first_brake,tail=list(tail) if outcome!='arrival' else []))
                        save(partial,rows);save(folder/'progress.json',dict(completed=len(rows),target=2 if smoke else 64,pid=os.getpid(),last=dict(route=route,seed=seed,outcome=outcome)))
            finally:env.close()
        assert len(rows)==(2 if smoke else 64)
        save(folder/'complete.json',dict(**signature,outcomes={m:dict(collections.Counter(r['outcome'] for r in rows if r['traffic']==m)) for m in ('none','reference')}))
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.name,a.step,a.smoke)
