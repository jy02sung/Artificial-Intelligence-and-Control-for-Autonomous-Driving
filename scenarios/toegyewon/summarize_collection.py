"""Report simulation warnings alongside successes; never silently drop failures."""
import argparse
from collections import Counter
import json
from pathlib import Path

def summarize(folder):
    m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    warnings=[]
    for ep in m['episodes']:
        text=(folder/f"episode_{ep['episode']:02d}.warnings.log").read_text(encoding='utf-8',errors='replace')
        warnings.append({'episode':ep['episode'],
            'background_teleport_warnings':sum('Teleporting vehicle' in line for line in text.splitlines()),
            'emergency_braking_warnings':sum('performs emergency braking' in line for line in text.splitlines()),
            'collision_warning_lines':sum('collision' in line.lower() for line in text.splitlines())})
    report={'dataset':folder.name,'complete':m['complete'],'transitions':m.get('transitions'),
        'episodes':len(m['episodes']),'outcomes':dict(Counter(e['end_reason'] for e in m['episodes'])),
        'ego_collisions':sum(e['ego_collisions'] for e in m['episodes']),
        'ego_teleports':sum(e['ego_teleports'] for e in m['episodes']),
        'warnings':warnings,'warning_source':'SUMO per-episode warnings.log, includes pre-ego warmup',
        'suggested_initial_imitation_episodes':[e['episode'] for e,w in zip(m['episodes'],warnings)
            if e['end_reason']=='arrival' and not w['background_teleport_warnings'] and not w['collision_warning_lines']],
        'note':'Suggestion is only a first filter, not proof of expert-quality driving. Actions are realized kinematics, not policy commands.'}
    (folder/'quality_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    print(json.dumps(summarize(p.parse_args().folder),indent=2))
