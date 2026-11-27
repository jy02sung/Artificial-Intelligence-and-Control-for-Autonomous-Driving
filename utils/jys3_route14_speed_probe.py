"""Change only longitudinal output for one second before the original R14 cut-in."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import jys3_gae98_cutin_replay as replay

C,L=replay.C,replay.L
ROOT=replay.ROOT.parent/"jys3_route14_speed_probe"


def run(case):
    C.torch.set_num_threads(1)
    row=case["original"]
    event=case["event"]
    model=C.final.network(replay.POLICY)
    hashes={}
    results={}
    with C.final.p.roll.baseline.simulator() as env,C.torch.inference_mode(),TemporaryDirectory() as tmp:
        transform=C.p.lateral_transform(env)
        for lead,shift in [(0,0),(5,-.25),(15,-.25),(5,.25),(15,.25)]:
            start=event["tick"]-lead
            xml=Path(tmp)/"moves.xml"
            env.ego_depart_lane=str(row["lane"])
            obs,_=C.start_trace(lambda:env.reset(seed=row["seed"],options={"route":"route_14"}),xml)
            assert replay.digest(obs)==row["initial_sha256"] and start>=1
            paid=set()
            applied=0
            for t in range(1,1001):
                x=transform(obs[:50])
                action=model.det_action(C.torch.from_numpy(x).unsqueeze(0))[0].numpy().copy()
                if not shift:
                    hashes[t]=replay.digest(x)
                elif t==start:
                    assert replay.digest(x)==hashes[t]
                if shift and start<=t<start+5:
                    adjusted=float(C.np.clip(float(action[0])+shift,-1.,1.))
                    applied+=int(adjusted!=float(action[0]))
                    action[0]=adjusted
                obs,_,done,cut,info=env.step(action)
                paid.update(env.sim.paid_forks)
                if done or cut:
                    break
            moves=C.finish_trace(env,xml,t)
            if not shift:
                assert (t,info["outcome"],len(moves))==(row["steps"],"collision",row["moves"])
            results[f"{lead}/{shift}"]=dict(outcome=info["outcome"],steps=t,changed=applied,
                                             passed=event["next_fork"] in paid)
    L.save(ROOT/f"{row['seed']}.json",results)
    return results


if __name__=="__main__":
    assert replay.mid.ft.sha(replay.POLICY)==replay.mid.ARMS["lambda98"]
    cases=[x for path in (replay.ROOT/"cases").glob("*.json") if (x:=L.read(path))["original"]["route"]==14]
    assert len(cases)==7
    with ProcessPoolExecutor(max_workers=4) as pool:
        cases=list(pool.map(run,cases))
    print({key:Counter(r[key]["outcome"] for r in cases) for key in cases[0]},flush=True)
