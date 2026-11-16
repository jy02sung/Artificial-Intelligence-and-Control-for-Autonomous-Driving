"""Resumable 50k A/B pilots. Teacher data never enters the on-policy buffer."""
import argparse,json,os,random,subprocess,time,ctypes
from pathlib import Path
import numpy as np
import torch
from utils.curriculum_data import OUT,ROOT,BASE,save,digest,source_agent
from utils.curriculum_env import CurriculumEnv
from utils.curriculum_ppo import KeepPPO

FILES=['utils/curriculum_data.py','utils/curriculum_env.py','utils/curriculum_ppo.py','utils/run_curriculum_keep.py','utils/timeout_rollout.py']

def atomic_torch(path,state):
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)

def run(args):
    torch.set_num_threads(1)
    random.seed(args.seed)
    folder=OUT/'runs'/args.name;folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'active.lock'
    if lock.exists():
        pid=int(lock.read_text())
        handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid) if os.name=='nt' else None
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            raise RuntimeError(f'Existing live run PID {pid}')
        if os.name!='nt':
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Existing live run PID {pid}')
        lock.rename(folder/f'stale-{pid}-{int(time.time())}.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    env=None
    try:
        reference,base_meta=source_agent()
        hp=dict(base_meta['hp'],seed=args.seed)
        agent=KeepPPO(obs_dim=63,act_dim=2,**hp);agent.load(str(BASE))
        keep=dict(np.load(OUT/'keep.npz',allow_pickle=False));safe=dict(np.load(OUT/'safe.npz',allow_pickle=False))
        assert len(keep['obs']) and {'cruise','preparation','lane_change'}<=set(keep['category']), 'Keep coverage incomplete'
        if args.condition=='B':assert len(safe['obs'])>0,'No validated safe demonstrations'
        agent.set_supervision(reference.policy,keep,safe);agent.safe_weight=.1 if args.condition=='B' else 0.
        for k,v in reference.policy.state_dict().items():assert torch.equal(v,agent.policy.state_dict()[k]),k
        hashes={**base_meta['source_hashes'],**{f:digest(ROOT/f) for f in FILES}}
        manifest=dict(condition=args.condition,seed=args.seed,parent_sha=digest(BASE),source_hashes=hashes,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),hp=hp,actor_lr=3e-5,critic_head_lr=3e-4,shared_trunk_lr=3e-5,keep_weight=agent.keep_weight,safe_weight=agent.safe_weight,target_kl=agent.target_kl,observation_dim=63,normalization='Fixed baseline scaling/clipping; no learned running statistics',budget=args.budget,rollout=args.rollout,keep_sha=digest(OUT/'keep.npz'),safe_sha=digest(OUT/'safe.npz'),evaluation_mode='mean_acceleration_and_argmax_lane',training_mode='on_policy_sampling',reward=base_meta['reward'],simulation=base_meta['simulation'],action_safety=base_meta['action_safety'])
        checkpoint=folder/'resume.pt'
        env=CurriculumEnv(args.seed)
        if checkpoint.exists():
            old=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
            for k in ['condition','seed','parent_sha','source_hashes','hp','keep_sha','safe_sha','rollout','keep_weight','safe_weight']:assert manifest[k]==old[k],('resume mismatch',k)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            agent.policy.load_state_dict(state['policy']);agent.optimizer.load_state_dict(state['optimizer']);agent.num_timesteps=state['steps'];agent.aux_rng.bit_generator.state=state['aux_rng']
            obs=env.load_state_dict(state['environment'])
            torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
            ep_stats=state['ep_stats'];updates=state['updates']
            save(folder/'resume-verification.json',dict(steps=agent.num_timesteps,observation_exact=True,all_vehicle_fingerprint_exact=True))
        else:
            save(folder/'manifest.json',manifest);agent.save(str(folder/'policy-0.pt'))
            obs,_=env.reset();ep_stats=[0.,0];updates=0
        limit=min(args.budget,args.stop_at if args.stop_at is not None else args.budget)
        started=time.time()
        while agent.num_timesteps<limit:
            for f,h in hashes.items():assert digest(ROOT/f)==h,('source changed',f)
            obs,last,finished=agent.collect_rollout(env,obs,ep_stats=ep_stats,steps=min(args.rollout,limit-agent.num_timesteps))
            adv,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
            metrics=agent.update(adv,ret);updates+=1
            state=dict(policy=agent.policy.state_dict(),optimizer=agent.optimizer.state_dict(),steps=agent.num_timesteps,updates=updates,environment=env.state_dict(),ep_stats=ep_stats,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),aux_rng=agent.aux_rng.bit_generator.state)
            atomic_torch(checkpoint,state)
            atomic_torch(folder/f'policy-{agent.num_timesteps}.pt',agent.policy.state_dict())
            record=dict(steps=agent.num_timesteps,updates=updates,metrics=metrics,mixture=dict(env.counts),exposure=dict(env.exposure),episodes=len(env.episodes),wall_seconds=time.time()-started,route_overrides=env.route_override_count)
            with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n')
            save(folder/'progress.json',dict(stage='training',pid=os.getpid(),target=args.budget,**record))
            print({'steps':agent.num_timesteps,'kl':metrics['rollout_kl_final'],'keep':metrics['keep_loss'],'safe':metrics['safe_loss'],'risk':dict(env.exposure)},flush=True)
            assert env.route_override_count==0
            assert all(np.isfinite(v) for v in metrics.values() if v is not None)
            if (folder/'STOP_AFTER_UPDATE').exists():break
        save(folder/'episodes.json',env.episodes)
        stage='pilot_training_complete_pending_evaluation' if agent.num_timesteps>=args.budget else 'checkpoint_saved_resume_available'
        save(folder/'progress.json',dict(stage=stage,pid=os.getpid(),steps=agent.num_timesteps,target=args.budget,next_command=f'python -X utf8 -m utils.run_curriculum_keep --condition {args.condition} --name {args.name} --seed {args.seed} --budget {args.budget} --rollout {args.rollout}',route_overrides=env.route_override_count))
    finally:
        if env is not None:env.close()
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['A','B'],default='A');p.add_argument('--name',default='A-seed11');p.add_argument('--seed',type=int,default=11);p.add_argument('--budget',type=int,default=50000);p.add_argument('--rollout',type=int,default=2048);p.add_argument('--stop-at',type=int);run(p.parse_args())
