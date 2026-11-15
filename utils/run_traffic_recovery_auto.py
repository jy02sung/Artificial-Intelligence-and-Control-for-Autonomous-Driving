"""Restart preserved experiments; analyze each batch and select reward follow-ups."""
import os,json,sys,subprocess,collections
from pathlib import Path
from utils import run_traffic_route_recovery as recovery
s=recovery.s;t=s.t;ROOT=s.ROOT
BASE=ROOT/'results/traffic_recovery_auto_20261116'
CONFIG=Path(os.environ.get('IC_RECOVERY_CONFIG',str(BASE/'round-01/config.json')))

class FrequentLog(s.Log):
    def log_train(self,step,metrics,verbose=False):
        super().log_train(step,metrics,verbose)
        tmp=self.directory/'recovery.tmp'
        s.a.b.torch.save(dict(policy=self.agent.policy.state_dict(),optimizer=self.agent.optimizer.state_dict(),steps=step),tmp)
        os.replace(tmp,self.directory/'recovery.pt')

class AdaptiveEnv(recovery.RouteRecoveryEnv):
    def __init__(self,*args,**kwargs):
        self.adaptive_config=t.read(CONFIG)
        super().__init__(*args,**kwargs)
    def step(self,action):
        obs,reward,done,cut,info=super().step(action)
        parts=info['route_reward_parts'];cfg=self.adaptive_config
        route=sum(parts.get(k,0) for k in ('selection','maintenance','departure','opportunity'))
        lane=parts.get('lane',0)
        extra=(cfg['route_scale']-1)*route+(cfg['lane_scale']-1)*lane
        self.reward_counts['adaptive_extra_total']+=extra
        self.reward_counts['bonus_total']+=extra
        info['route_selection_reward']+=extra
        return obs,reward+extra,done,cut,info

def configure():
    cfg=t.read(CONFIG)
    s.OUT=CONFIG.parent;s.t.OUT=s.OUT
    s.SOURCE=ROOT/cfg['source']
    s.DOC=ROOT/'docs/experiments/ic-traffic-recovery-auto-results-20261116.md';t.DOC=s.DOC
    s.t.TrialEnv=AdaptiveEnv;s.Log=FrequentLog
    s.t.EVAL_SEEDS=tuple(cfg['eval_seeds']);s.EVAL_SEEDS=s.t.EVAL_SEEDS
    s.SOURCE_FILES=(*s.SOURCE_FILES,'utils/run_traffic_recovery_auto.py');t.SOURCE_FILES=s.SOURCE_FILES

# Spawned workers need the same environment and source configuration.
if CONFIG.exists():configure()

def analyze(directory):
    rows=[]
    for p in directory.glob('runs/*/policy-*-evaluation.json'):
        data=t.read(p);g=[r for r in data if r['traffic']=='reference']
        if len(g)!=48:continue
        count=collections.Counter(r['outcome'] for r in g)
        rows.append(dict(condition=p.parent.name,model=str(p.with_name(p.name.replace('-evaluation.json','.pt')).relative_to(ROOT)),arrival=count['arrival'],collision=count['collision'],other=48-count['arrival']-count['collision'],n=48))
    assert rows
    # Safety-qualified candidates first; then maximize completion and reduce unresolved outcomes.
    best=max(rows,key=lambda r:(r['collision']/r['n']<.05,r['arrival'],-r['other'],-r['collision']))
    report=dict(all=rows,best=best,development_goal=best['other']==0 and best['collision']/best['n']<.05)
    t.save(directory/'decision.json',report)
    doc=ROOT/'docs/experiments/ic-traffic-recovery-auto-results-20261116.md'
    lines=['# 교통 복구 실험 자동 분석','',f'현재 배치: {directory.name}. 모든 수치는 개발 평가이며 최종 검증과 구분한다.','', '|조건/모델|완주|충돌|기타|','|---|---:|---:|---:|']
    lines += [f"|{r['condition']}/{Path(r['model']).name}|{r['arrival']}/48|{r['collision']}|{r['other']}|" for r in rows]
    lines+=['', '선택 후보: '+best['model']]
    doc.write_text('\n'.join(lines)+'\n',encoding='utf8')
    return report

def supervise():
    BASE.mkdir(parents=True,exist_ok=True)
    fd=os.open(BASE/'active.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    source='results/traffic_safety_warmstart_20261115/runs/lane/policy-151552.pt'
    route_scale=lane_scale=1
    try:
        for number in range(1,4):
            d=BASE/f'round-{number:02d}';d.mkdir(exist_ok=True)
            cfg=dict(source=source,route_scale=route_scale,lane_scale=lane_scale,eval_seeds=[59101,59102,59103],reason='restart after power loss' if number==1 else reason)
            t.save(d/'config.json',cfg)
            env=os.environ.copy();env['IC_RECOVERY_CONFIG']=str(d/'config.json')
            for mode in ('smoke','train'):
                command=[sys.executable,'-X','utf8','-u','-m','utils.run_traffic_recovery_auto','--batch']
                if mode=='smoke':command.append('--smoke')
                t.save(BASE/'status.json',dict(stage=mode,round=number,pid=os.getpid()))
                with (d/(mode+'.stdout.log')).open('w',encoding='utf8') as out,(d/(mode+'.stderr.log')).open('w',encoding='utf8') as err:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err,check=True)
            result=analyze(d);best=result['best']
            if result['development_goal']:
                t.save(BASE/'status.json',dict(stage='development_goal_pending_fresh_seed_validation',round=number,best=best));return
            source=best['model']
            if best['collision']/48>=.05:
                lane_scale*=2;reason='collision target missed: double dangerous lane-change penalty only'
            else:
                route_scale*=2;reason='safety sample meets target but unresolved exits remain: double route reward components only'
            t.save(BASE/'status.json',dict(stage='next_round_selected',round=number,best=best,next_reason=reason))
        t.save(BASE/'status.json',dict(stage='needs_trace_analysis_after_three_batches',reason='Do not keep increasing coefficients without new failure evidence',best=best))
    except Exception as exc:
        t.save(BASE/'status.json',dict(stage='error',error=repr(exc)));raise
    finally:(BASE/'active.lock').unlink(missing_ok=True)

if __name__=='__main__':
    if '--batch' in sys.argv:
        sys.argv.remove('--batch');s.main()
    else:supervise()
