from collections import Counter,defaultdict
from pathlib import Path
import csv,json,math,statistics,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'local_holdout_1600'
def records(kind):
    return [json.loads(p.read_bytes()) for p in sorted((OUT/kind).glob('*.json'))]
def dump(name,items):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in items for k in x)))
        w.writeheader(); w.writerows(items)
def interval(n,k):
    p=k/n;z=1.959964
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return round(100*(center-half),2),round(100*(center+half),2)
def edge(lane):return lane.rsplit('_',1)[0]
def laneidx(lane):return int(lane.rsplit('_',1)[1])
def main():
    rows=records('fresh'); baseline=records('replicate'); failures=records('replay')
    ref=json.loads((ROOT/'results/mcw_runs/eval_mcw2_occlusion_12288_80ep/eval_16routes.json').read_bytes())['rows']
    plan=json.loads((OUT/'plan.json').read_bytes())
    refidx={(x['route'],x['seed']):x for x in ref}
    assert len(baseline)==80 and all((r['outcome'],r['steps'])==(refidx[r['route'],r['seed']]['outcome'],refidx[r['route'],r['seed']]['steps']) for r in baseline)
    assert len(set(r['seed'] for r in rows))==1600 and set((r['route'],r['seed']) for r in rows)==set(map(tuple,plan['fresh']))
    assert not {r['seed'] for r in rows}.intersection(r['seed'] for r in baseline)
    assert all(n==100 for n in Counter(r['route'] for r in rows).values())
    byseed={r['seed']:r for r in rows}
    assert {r['seed'] for r in failures}=={r['seed'] for r in rows if r['outcome']!='arrival' and 'steps' in r}
    assert all((r['outcome'],r['steps'],r['contacts'])==(byseed[r['seed']]['outcome'],byseed[r['seed']]['steps'],byseed[r['seed']]['contacts']) for r in failures)
    dump('episodes.csv',[{k:v for k,v in r.items() if k!='contacts'} for r in rows])
    route_stats=[]
    for route in sorted(set(r['route'] for r in rows)):
        group=[r for r in rows if r['route']==route]
        cnt=Counter(r['outcome'] for r in group);lo,hi=interval(100,cnt['collision'])
        route_stats.append(dict(route=route,arrival=cnt['arrival'],collision=cnt['collision'],timeout=cnt['timeout'],
                            other=100-sum(cnt[k] for k in ('arrival','collision','timeout')),collision_low=lo,collision_high=hi))
    dump('routes.csv',route_stats)
    links=defaultdict(set)
    scenario=ROOT/'scenarios'/'hv_high15000_entryprep'
    net=ET.parse(scenario/'network.net.xml')
    for node in net.findall('connection'):
        links[node.get('from'),node.get('to')].add(int(node.get('fromLane')))
    routes={n.get('id'):n.get('edges').split() for n in ET.parse(scenario/'highway.rou.xml').findall('route')}
    ctrlfile=OUT/'matched_controls.json'
    controls={r['incident_seed']:r for r in json.loads(ctrlfile.read_bytes())} if ctrlfile.exists() else {}
    events=[]
    for r in failures:
        hist=r['history']; f=hist[-1]; road=f['edge']; lane=laneidx(f['lane']); cmd=f['cmd']
        path=routes[r['route']]; nxt=path[path.index(road)+1] if road in path[:-1] else None
        allowed=links[road,nxt] if nxt else set()
        incidents=r.get('contacts') or [{}]
        for c in incidents:
            role='collider' if c.get('collider')=='ego' else 'victim' if c.get('victim')=='ego' else ''
            other='victim' if role=='collider' else 'collider'
            fl=f.get('foe_lane','')
            same_edge=bool(fl and edge(fl)==road)
            delta=laneidx(fl)-lane if same_edge else None
            toward=bool(cmd and delta==cmd)
            ctrl=controls.get(r['seed'],{}); g=ctrl.get('frame') or {}
            changes=sum(1 for a,b in zip(hist[-11:-1],hist[-10:]) if a['edge']==b['edge'] and a['lane']!=b['lane'])
            events.append(dict(route=r['route'],seed=r['seed'],outcome=r['outcome'],seconds=round(r['steps']*.2,1),
                contact_type=c.get('type',''),contact_lane=c.get('lane',''),contact_pos=c.get('pos'),ego_role=role,partner=c.get(other,''),partner_type=c.get(other+'Type',''),
                pre_edge=road,pre_lane=f['lane'],remaining_m=round(f['remaining'],2),speed_mps=round(f['speed'],2),accel_raw=round(f['accel'],3),
                lane_cmd=cmd,cmd_applied=f['applied'],speed_2s_before=round(hist[-11]['speed'],2) if len(hist)>=11 else None,
                previous_changes_2s=changes,next_road=nxt,connected_lanes=' '.join(map(str,sorted(allowed))),current_connected=(lane in allowed) if allowed else None,
                command_into_connected=(lane+cmd in allowed) if allowed else None,foe_lane=fl,foe_speed=f.get('foe_speed'),foe_distance=f.get('foe_distance'),
                geometric_occlusion=f.get('geometric_occlusion'),occluded_in_last_1s=any(q.get('geometric_occlusion') for q in hist[-5:]),
                same_edge_before=same_edge,foe_lane_offset=delta,command_toward_partner=toward,foe_longitudinal_offset=(round(f['foe_pos']-f['pos'],2) if same_edge else None),
                target_front_obs_m=(round(f['obs'][1+4*(2-cmd)]*120,1) if cmd else None),
                target_rear_obs_m=(round(-f['obs'][3+4*(2-cmd)]*120,1) if cmd else None),
                control_seed=ctrl.get('control_seed'),control_pos_error=g.get('pos_difference'),control_lane=g.get('lane'),control_speed=g.get('speed'),control_cmd=g.get('cmd'),
                control_connected=(laneidx(g['lane']) in allowed) if g and allowed else None))
    dump('incidents.csv',events)
    summary=dict(n=1600,counts=Counter(r['outcome'] for r in rows),routes=route_stats,replication=80,replays=len(failures),contact_rows=len(events),
                 collision_types=Counter(e['contact_type'] for e in events),contact_edges=Counter(edge(e['contact_lane']) for e in events if e['contact_lane']))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary))
if __name__=='__main__':main()
