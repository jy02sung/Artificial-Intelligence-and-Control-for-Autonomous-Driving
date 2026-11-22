"""D uses C3 policy inputs and records the history of each physical route constraint."""
from functools import lru_cache
from utils.experiment_c_env import make_c
from utils.experiment_d_probe import OUT

def environment(stage,arm,seed,folder,audit=False):
    env=make_c(stage,"C3",seed,folder,audit=audit)
    reset_c,step_c=env.reset,env.step
    @lru_cache(maxsize=None)
    def project(route,index):
        path=env.routes[route]
        for j in range(index,len(path)-1):
            src,dst=path[j:j+2]
            targets=env.links[src][dst]
            if targets and len(targets)<env.counts[src]:
                possible=certain=set(targets)
                for k in range(j-1,index-1,-1):
                    links=env.navigation.links[path[k],path[k+1]]
                    possible={a for a,b in links if b in possible}
                    uncertain={a for a,b in links if b not in certain}
                    certain={a for a,b in links if b in certain}-uncertain
                if certain and certain==possible:
                    offset=sum(env.navigation.link_lengths.get((path[k],path[k+1]),0.)+env.navigation.lengths[path[k+1]] for k in range(index,j))
                    return (j,src,dst),frozenset(certain),offset
                break
            # Beyond a fork with multiple choices the approach is no longer unique.
            if len(env.links[src])>1:break
        return None
    def reset(*args,**kwargs):
        env.d_ready=set()
        env.d_departures={}
        env.d_cycles=[]
        return reset_c(*args,**kwargs)
    def step(action):
        edge,lane,path,idx,_,_,_,_,rem=env.snapshot
        eligible=not edge.startswith(":") and idx>=0
        con=project(env.episode_route,idx) if eligible and (arm=="D1" or audit) else None
        if con:
            key,targets,offset=con
            distance=rem+offset
            if arm=="D1":
                now=min(abs(lane-t) for t in targets)
                env.c_best[key]=min(env.c_best.get(key,now),now)
            if audit and lane in targets:env.d_ready.add(key)
        obs,reward,done,cut,info=step_c(action)
        post=env.snapshot
        if con and post and post[0]==edge and post[1]!=lane:
            after=post[1]
            if arm=="D1" and distance>400:
                now=min(abs(after-t) for t in targets)
                env.c_best[key]=min(env.c_best[key],now)
            if audit:
                if lane in targets and after not in targets:
                    env.d_departures[key]=dict(step=env.step_count,edge=edge,distance=distance,lane=lane,after=after)
                if lane not in targets and after in targets and key in env.d_departures:
                    env.d_cycles.append(dict(constraint=key,departure=env.d_departures.pop(key),step=env.step_count,edge=edge,distance=distance,
                        lane=lane,after=after,prep=info["c_prep"]))
        return obs,reward,done,cut,info
    env.reset,env.step=reset,step
    return env
