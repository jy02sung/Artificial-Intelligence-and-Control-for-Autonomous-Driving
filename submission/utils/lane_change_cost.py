import math
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET
def split_lane(lane):
    edge, index = lane.rsplit("_", 1)
    return [edge, int(index)]


def measurements(obs, direction, visibility, vmax):
    i = 1 + 4 * (2-direction)
    a,b,c,d = map(float, obs[i:i+4])
    missing_f = a >= 1-1e-6 and b >= 1-1e-6
    missing_r = c <= -1+1e-6 and d <= -1+1e-6
    fg,fc = (None,None) if missing_f else (a*visibility,-b*vmax)
    rg,rc = (None,None) if missing_r else (-c*visibility,d*vmax)
    ttc = [gap/v for gap,v in ((fg,fc),(rg,rc)) if gap is not None and v>0]
    overlap = direction!=0 and bool(obs[37 if direction==1 else 36]>.5)
    speed = float(obs[0])*vmax
    tight = (overlap or (fg is not None and fg<=3+.45*speed+max(0,fc))
             or (rg is not None and rg<=3+.50*max(0,speed+rc)+max(0,rc)))
    return dict(front_gap=fg, rear_gap=rg, front_closing=fc, rear_closing=rc,
                ttc=min(ttc,default=None), overlap=overlap, tight=bool(tight))
import utils.ppo_rollout as p
import utils.kl_ppo as old
from env.sumo_env import traci
import numpy as np


def invalid(edge,lane,count,cmd,obs):
    return bool(cmd and (obs[38]>.5 or (cmd==-1 and lane==0) or (cmd==1 and lane==count-1)))


def dangerous(m):
    return m["overlap"] or any(gap is not None and (gap<distance or
           (closing>.5 and gap/closing<1.5)) for gap,closing,distance in [
               (m["front_gap"],m["front_closing"],6.),(m["rear_gap"],m["rear_closing"],10.)])


def start_trace(thunk,path):
    start=traci.start
    with patch.object(traci,"start",lambda args:start(args+["--lanechange-output",str(path)])):
        return thunk()


def finish_trace(env,path,last):
    end=traci.simulation.getTime()
    env.close()
    moves=[]
    for change in ET.parse(path).getroot().iter("change"):
        if change.get("id")==env.ego_id:
            step=last+1-round((end-float(change.get("time")))/env.sim.step_length)
            a,b=map(split_lane,(change.get("from"),change.get("to")))
            if a[0]==b[0] and a[1]!=b[1]:
                moves.append((step,a,b))
    return moves


class Episodes(p.Episodes):
    def extra_move_cost(self, measurement):
        return 0.

    def __init__(self,env,arm,path,*,training_seed,warm=False):
        super().__init__(env,arm,path,training_seed=training_seed)
        self.costed=arm.startswith("S3")
        self.risk=arm in ("S3-B","S3-C")
        self.xml=Path(env.sim.cfg_path).parent/"ego-changes.xml"
        self.stats=Counter()
        if self.costed:
            self.count=200000 if warm else 100000
            self.transform=p.lateral_transform(env)
        if self.risk:
            old.driving.COLLISION_REWARD=-16.

    def reset(self):
        reset=super().reset
        return start_trace(reset,self.xml) if self.risk else reset()

    def drive(self,action):
        if not self.costed:
            return super().drive(action)
        env=self.env
        cmd=env.sim._quantize_lane_change(action[1])
        edge,lane=env.snapshot[:2]
        count=env.counts.get(edge) or traci.edge.getLaneNumber(edge)
        obs=self.obs
        bad=invalid(edge,lane,count,cmd,obs)
        nxt,reward,done,cut,info=super().drive(action)
        charge=False
        extra=0.
        if self.risk:
            direction=0
            if info["collided"]:
                last=len(self.actions)
                for t,a,b in finish_trace(env,self.xml,last):
                    if t==last:
                        direction=int(np.sign(b[1]-a[1]))
            elif env.snapshot and env.snapshot[0]==edge:
                direction=int(np.sign(env.snapshot[1]-lane))
            if direction:
                m=measurements(obs,direction,env.visibility,env.vmax)
                charge=dangerous(m)
                extra=self.extra_move_cost(m)
        self.stats.update(invalid_commands=int(bad),risk_moves=int(charge))
        return nxt,reward-.02*bad-1.5*charge-extra,done,cut,info


def verified_update(agent,ep,adv,ret,ref=None):
    param=next(agent.policy.trunk.parameters())
    def count():
        return int(agent.optimizer.state.get(param,{}).get("step",0))
    before=count()
    metrics=agent.update(adv,ret,reference_policy=ref,reference_coef=.08 if ref is not None else 0.,
                         stratified_reference=ref is not None)
    actual=count()-before
    assert actual==10*math.ceil(len(ret)/64)
    metrics.update(minibatches=actual,lr=agent.optimizer.param_groups[0]["lr"])
    return metrics
