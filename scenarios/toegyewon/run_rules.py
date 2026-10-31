"""Run the v2 marked road with the same supervisor used by data collection."""
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import traci
from build import HERE,binary
from collect import snapshot
from rule_policy import RulePolicy

def run(gui=False,seed=300,until=400,dataset=None,episode=0):
    name=datetime.now(timezone(timedelta(hours=9))).strftime('preview_rules_%Y%m%d_%H%M%S')
    out=HERE/'results'/name;out.mkdir(parents=True,exist_ok=False)
    net=HERE/'rules_v2/network.net.xml';routes=HERE/'rules_v2/routes.rou.xml'
    if dataset:
        manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
        seed=manifest['episodes'][episode]['seed'];net=dataset/'toegyewon.net.xml';routes=dataset/f'episode_{episode:02d}.rou.xml'
    args=[binary('sumo-gui' if gui else 'sumo'),'-n',str(net),'-r',str(routes),'--seed',str(seed),
          '--step-length','.2','--no-step-log','true','--collision.check-junctions','true','--time-to-teleport','120']
    if gui:args+=['--start','--gui-settings-file',str(HERE/'view.xml'),'--additional-files',str(HERE/'rules_v2/markings.add.xml'),'--window-size','1200,850','--delay','30']
    traci.start(args);policy=RulePolicy(seed,out/'events.jsonl');states={}
    if gui:
        traci.gui.setZoom('View #0',400)
        traci.gui.setOffset('View #0',570,650)
    try:
        while traci.simulation.getTime()<until and traci.simulation.getMinExpectedNumber():
            d=policy.before_step(states);traci.simulationStep();nxt=snapshot()
            policy.after_step(states,nxt,d);states=nxt
        traci.simulation.saveState(str(out/'state.xml.gz'))
        (out/'summary.json').write_text(json.dumps(policy.counts,indent=2),encoding='utf-8')
    finally:
        policy.close();traci.close(wait=False)
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gui',action='store_true');p.add_argument('--seed',type=int,default=300)
    p.add_argument('--until',type=float,default=400);p.add_argument('--dataset',type=Path);p.add_argument('--episode',type=int,default=0)
    a=p.parse_args();run(a.gui,a.seed,a.until,a.dataset,a.episode)
