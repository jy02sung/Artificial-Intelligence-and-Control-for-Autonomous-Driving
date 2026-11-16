"""Offline probability of sampling negative acceleration at recorded risky states."""
import hashlib
import json
import math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/junction_lookahead_trials_20261116'


def main():
    models={'source':ROOT/'results/traffic_recovery_auto_20261116/round-02/runs/route/policy-200000.pt',
            'ahead150_51200':OUT/'runs/ahead150/policy-51200.pt'}
    output={}
    for label,cp in models.items():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        sd=state['log_std'].exp().item()
        source=OUT/f'braking-replay-{label}.json'
        trace=json.loads(source.read_text())['trace']
        rows=[]
        for x in trace:
            if x['risk']>0 and x['t']<11.4:
                mu=x['action'][0]
                probability=.5*math.erfc(mu/(sd*math.sqrt(2)))
                rows.append(dict(t=x['t'],mu=mu,std=sd,brake_sample_probability=probability,risk=x['risk']))
        output[label]=dict(checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha=hashlib.sha256(cp.read_bytes()).hexdigest(),
                           source_trace_sha=hashlib.sha256(source.read_bytes()).hexdigest(),std=sd,states=rows,
                           interpretation='Conditional training sampling probabilities at recorded deterministic states; not an episode collision estimate.')
        print(label,sd,len(rows),min(x['brake_sample_probability'] for x in rows),max(x['brake_sample_probability'] for x in rows))
    (OUT/'braking-exploration-audit.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
