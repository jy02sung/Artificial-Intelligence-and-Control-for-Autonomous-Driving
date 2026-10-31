"""Traffic-only merge diagnostics. No ego, imitation data or AI training."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from build import HERE,binary,write_xml
from comparison_setup import freeze_demand

CONFIG=json.loads((HERE/'traffic_config.json').read_text(encoding='utf-8'))
OUT=HERE/'traffic'

def make_routes(scale,seed):
    root=ET.parse(HERE/'rules_v2/routes.rou.xml').getroot()
    for tag in ['vehicle','vType']:
        for e in list(root.findall(tag)):
            if tag=='vehicle' or e.get('id')=='ego_baseline':root.remove(e)
    for ident,model in [('car_idm','IDM'),('car_eidm','EIDM'),('car_acc','ACC')]:
        root.insert(0,ET.Element('vType',id=ident,carFollowModel=model,accel='2',decel='4.5',tau='1.5',minGap='2.5',length='5',color='0.2,0.7,0.7'))
    for t in root.findall('vType'):
        t.set('laneChangeModel','LC2013');t.set('lcCooperative','1')
    d=root.find('vTypeDistribution');d.set('vTypes',' '.join(CONFIG['background_distribution']))
    d.set('probabilities',' '.join(map(str,CONFIG['background_distribution'].values())))
    for f in root.findall('flow'):
        f.set('end',str(CONFIG['demand_end_s']));f.set('vehsPerHour',str(float(f.get('vehsPerHour'))*scale))
    freeze_demand(root,seed)
    return root

def build():
    OUT.mkdir(exist_ok=True)
    nodes=ET.Element('nodes')
    for ident in CONFIG['merge_nodes']:ET.SubElement(nodes,'node',id=ident,type='zipper')
    write_xml(nodes,OUT/'merges.nod.xml')
    net=ET.parse(HERE/'rules_v2/network.net.xml').getroot()
    incoming={e.get('id') for e in net.findall('edge') if e.get('to') in CONFIG['merge_nodes']}
    links=ET.Element('connections')
    for c in net.findall('connection'):
        if c.get('from') in incoming:
            attrs={k:c.get(k) for k in ['from','to','fromLane','toLane']}
            attrs['visibilityDistance']=str(CONFIG['merge_visibility_m'])
            ET.SubElement(links,'connection',attrs)
    write_xml(links,OUT/'merges.con.xml')
    with (OUT/'build.log').open('w',encoding='utf-8') as log:
        subprocess.run([binary('netconvert'),'-s',str(HERE/'rules_v2/network.net.xml'),
            '--node-files',str(OUT/'merges.nod.xml'),'--connection-files',str(OUT/'merges.con.xml'),
            '-o',str(OUT/'network.net.xml')],stdout=log,stderr=subprocess.STDOUT,check=True)
    rebuilt=ET.parse(OUT/'network.net.xml').getroot()
    # netconvert also recomputes unrelated, manually shaped weaving junctions.
    # Transplant ONLY complete generated merge blocks (including their conflict matrices).
    # Never invent/edit response bits; all local junction logic comes from netconvert.
    prefixes=tuple(':'+n+'_' for n in CONFIG['merge_nodes'])
    def affected(e):
        if e.tag=='junction':return e.get('id') in CONFIG['merge_nodes'] or e.get('id','').startswith(prefixes)
        if e.tag=='edge':return e.get('id','').startswith(prefixes)
        if e.tag=='connection':return e.get('from') in incoming or e.get('from','').startswith(prefixes)
        return False
    merged=copy.deepcopy(net)
    for tag in ['edge','junction','connection']:
        existing=[e for e in merged if e.tag==tag and affected(e)]
        index=list(merged).index(existing[0])
        for e in existing:merged.remove(e)
        for offset,e in enumerate(e for e in rebuilt if e.tag==tag and affected(e)):
            merged.insert(index+offset,copy.deepcopy(e))
    rebuilt=merged
    write_xml(rebuilt,OUT/'network.net.xml')
    # Check non-internal road identity, lane counts and every existing route connection.
    old={e.get('id'):e for e in net.findall('edge') if e.get('function')!='internal'}
    new={e.get('id'):e for e in rebuilt.findall('edge') if e.get('function')!='internal'}
    assert set(old)==set(new)
    assert all(len(e.findall('lane'))==len(new[k].findall('lane')) for k,e in old.items())
    connections=lambda root:{tuple(c.get(k) for k in ['from','to','fromLane','toLane']) for c in root.findall('connection') if not c.get('from').startswith(':')}
    assert connections(net)==connections(rebuilt),'Unexpected lane connectivity change'
    audit={'merge_types':{i:rebuilt.find(f"junction[@id='{i}']").get('type') for i in CONFIG['merge_nodes']},
           'external_connections_preserved':True,
           'lane_length_changes_m':{k:max(abs(float(a.get('length'))-float(b.get('length'))) for a,b in zip(e.findall('lane'),new[k].findall('lane'))) for k,e in old.items()}}
    assert set(audit['merge_types'].values())=={'zipper'}
    assert all(v==0 for v in audit['lane_length_changes_m'].values())
    for k,e in old.items():
        assert ET.tostring(e)==ET.tostring(new[k]),k
    (OUT/'network_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    write_xml(make_routes(1.,1000),OUT/'routes.rou.xml')
    cfg=ET.Element('configuration');i=ET.SubElement(cfg,'input')
    ET.SubElement(i,'net-file',value='network.net.xml');ET.SubElement(i,'route-files',value='routes.rou.xml')
    ET.SubElement(i,'additional-files',value='../rules_v2/markings.add.xml')
    t=ET.SubElement(cfg,'time');ET.SubElement(t,'step-length',value=str(CONFIG['step_length_s']))
    ET.SubElement(t,'end',value=str(CONFIG['demand_end_s']+CONFIG['drain_s']))
    p=ET.SubElement(cfg,'processing');ET.SubElement(p,'time-to-teleport',value='-1')
    ET.SubElement(p,'collision.check-junctions',value='true')
    r=ET.SubElement(cfg,'random_number');ET.SubElement(r,'seed',value='1000')
    g=ET.SubElement(cfg,'gui_only');ET.SubElement(g,'gui-settings-file',value='../view.xml')
    write_xml(cfg,OUT/'traffic.sumocfg')
    return audit

def trial(variant,scale,seed,root):
    out=root/f'{variant}_{scale}_{seed}';out.mkdir()
    routes=make_routes(scale,seed);write_xml(routes,out/'routes.rou.xml')
    net=OUT/'network.net.xml' if variant=='zipper' else HERE/'rules_v2/network.net.xml'
    with (out/'run.log').open('w',encoding='utf-8') as log:
        subprocess.run([binary('sumo'),'-n',str(net),'-r',str(out/'routes.rou.xml'),
            '--step-length',str(CONFIG['step_length_s']),'--end',str(CONFIG['demand_end_s']+CONFIG['drain_s']),
            '--seed',str(seed),'--time-to-teleport','-1','--collision.check-junctions','true',
            '--no-step-log','--duration-log.disable','true','--tripinfo-output',str(out/'trips.xml'),
            '--tripinfo-output.write-unfinished','true','--tripinfo-output.write-undeparted','true',
            '--summary-output',str(out/'summary.xml'),'--summary-output.period','10',
            '--collision-output',str(out/'collisions.xml')],stdout=log,stderr=subprocess.STDOUT,check=True)
    trips=ET.parse(out/'trips.xml').getroot().findall('tripinfo')
    collisions=ET.parse(out/'collisions.xml').getroot().findall('collision')
    text=(out/'run.log').read_text(encoding='utf-8')
    groups={}
    for route in routes.findall('route'):
        name=route.get('id');a=[v for v in trips if v.get('id').rsplit('.',1)[0]==name]
        done=[v for v in a if float(v.get('arrival'))>=0]
        groups[name]={'demand':len(a),'arrived':len(done),'unfinished':len(a)-len(done),
            'mean_duration_s':float(np.mean([float(v.get('duration')) for v in done])) if done else None,
            'mean_waiting_s':float(np.mean([float(v.get('waitingTime')) for v in a])) if a else 0,
            'max_waiting_s':max([float(v.get('waitingTime')) for v in a],default=0)}
    summary=ET.parse(out/'summary.xml').getroot().findall('step')[-1]
    result={'variant':variant,'scale':scale,'seed':seed,'demand':len(routes.findall('vehicle')),
        'arrived':sum(float(v.get('arrival'))>=0 for v in trips),'undeparted':sum(float(v.get('depart'))<0 for v in trips),
        'unfinished':sum(float(v.get('arrival'))<0 for v in trips),'collisions':len(collisions),
        'teleport_warnings':text.count('Teleporting vehicle'),'emergency_braking_warnings':text.count('performs emergency braking'),
        'last_summary_time_s':float(summary.get('time')),'routes':groups}
    assert len(trips)==result['demand'],(len(trips),result['demand'])
    result['passed']=result['unfinished']==0 and result['collisions']==0 and result['teleport_warnings']==0
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(variant,scale,seed,'arrived',result['arrived'],'/',result['demand'],'collisions',result['collisions'],flush=True)
    return result

def validate(quick=False):
    from datetime import datetime
    root=HERE/'results'/datetime.now().strftime('traffic_stability_%Y%m%d_%H%M%S');root.mkdir()
    seeds=CONFIG['validation_seeds'][:1] if quick else CONFIG['validation_seeds']
    jobs=[(v,s,k) for v in ['priority','zipper'] for s in CONFIG['validation_scales'] for k in seeds]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda j:trial(*j,root),jobs))
    report={'config':CONFIG,'results':results,'raw_results':str(root),'all_zipper_passed':all(r['passed'] for r in results if r['variant']=='zipper')}
    (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('REPORT',root,flush=True)
    if not report['all_zipper_passed']:
        raise RuntimeError('Traffic validation failed; inspect the saved report before using this network.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');p.add_argument('--quick',action='store_true');p.add_argument('--gui',action='store_true');a=p.parse_args()
    build()
    if a.validate:validate(a.quick)
    if a.gui:subprocess.Popen([binary('sumo-gui'),'-c',str(OUT/'traffic.sumocfg')])
