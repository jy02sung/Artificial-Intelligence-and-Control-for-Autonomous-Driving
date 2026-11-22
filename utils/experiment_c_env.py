"""Experiment C environment. Only the new reward reads route-specific lane targets."""
from collections import Counter
from utils.experiment_b import environment as baseline
from utils.experiment_b import m
from pathlib import Path

OUT=m.ROOT/"results"/"experiment_c_v1"
ARMS=("C0","C1","C2","C3")
SEEDS=(11,22,33)
STEPS=tuple(range(0,1000001,50000))
SELECT=(830000,)
HOLD=(940000,940100,940200,940300)
DROPS={"844484877","206426298","86198708"}

def constraint(env,snap):
    edge,lane,path,idx,_,_,_,_,distance=snap
    if edge.startswith(":") or idx<0:return None
    nav=env.navigation
    for j in range(idx,len(path)-1):
        if distance>400:break
        src,dst=path[j:j+2]
        targets=env.links[src][dst]
        if targets and len(targets)<env.counts[src]:
            possible=set(targets)
            certain=set(targets)
            for k in range(j-1,idx-1,-1):
                links=nav.links[path[k],path[k+1]]
                possible={a for a,b in links if b in possible}
                uncertain={a for a,b in links if b not in certain}
                certain={a for a,b in links if b in certain}-uncertain
            return (j,src,dst),certain,possible!=certain or not certain
        distance+=nav.link_lengths.get((src,dst),0.)+nav.lengths[dst]

def make_c(stage,arm,seed,folder,audit=False):
    new=arm in ("C2","C3")
    env=baseline(stage,"B1",seed,folder)
    env.physical_link_horizon=400. if arm=="C3" else env.visibility
    env.use_prep_reward=not new
    original_reset,original_step=env.reset,env.step
    def reset(*args,**kwargs):
        env.c_best={}
        env.c_totals=Counter()
        env.c_journal=[]
        env.c_drop={}
        env.c_commands=Counter()
        return original_reset(*args,**kwargs)
    def step(action):
        pre=env.snapshot
        edge,lane,path,idx,_,_,_,speed,rem=pre
        con=constraint(env,pre) if new else None
        before=len(env.credits.paid)
        if con:
            key,targets,ambiguous=con
            if ambiguous:env.c_totals["ambiguous_steps"]+=1
            else:
                old=min(abs(lane-t) for t in targets)
                env.c_best[key]=min(old,env.c_best.get(key,old))
        prev_pass=env.events["passes"]
        obs,r,done,cut,info=original_step(action)
        post=env.snapshot
        changed=bool(post and post[0]==edge and post[1]!=lane)
        prep=0.
        crossing=0.
        if new:
            if con and not ambiguous and changed and not done:
                now=min(abs(post[1]-t) for t in targets)
                levels=max(0,env.c_best[key]-now)
                prep=.3*levels
                env.c_best[key]=min(now,env.c_best[key])
                if levels:env.c_totals["split_prep" if (key[1],key[2]) in env.signs else "drop_prep"]+=prep
            if not info["collided"]:
                crossing=.25*(env.events["passes"]-prev_pass)
            env.events["prep_reward"]+=prep
            env.c_totals["pass_reward"]+=crossing
            r+=prep+crossing
        else:
            prep=.3*(len(env.credits.paid)-before)
        if audit:
            if edge==path[0] and env.episode_route=="route_16":
                env.c_commands[str(info["lane_change"])]+=1
                env.c_commands["actual"]+=int(changed)
            if edge in DROPS:
                ds=env.c_drop.setdefault(edge,dict(entered_lane=lane,needed=lane not in env.links[edge][path[idx+1]],passed=False,wrong_end=False))
                ds["wrong_end"] |= bool(lane not in env.links[edge][path[idx+1]] and rem<=1 and speed<=.1)
            if post and not post[0].startswith(":"):
                for road,ds in env.c_drop.items():
                    j=path.index(road)
                    if post[3]==j+1 and post[0]==path[j+1]:ds["passed"]=True
            if changed or crossing or prep:
                env.c_journal.append(dict(step=env.step_count,edge=edge,lane=lane,after=post[1] if post else None,
                    distance=rem,command=info["lane_change"],prep=prep,passed_bonus=crossing))
        info["c_prep"],info["c_pass"]=prep,crossing
        return obs,r,done,cut,info
    env.reset,env.step=reset,step
    return env
