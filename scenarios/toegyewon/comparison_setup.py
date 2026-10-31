"""Matched experimental inputs; no background trajectory replay or model training."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET

MODELS=('Krauss','IDM','EIDM','ACC')
COMMON=dict(accel='2.6',decel='4.5',emergencyDecel='9',tau='1.2',minGap='2.5',
            length='5',width='1.8',speedFactor='1',speedDev='0',sigma='0.5',
            laneChangeModel='LC2013',lcStrategic='1',lcCooperative='1',lcSpeedGain='1',lcKeepRight='1')

def configure_ego(root,model):
    if model not in MODELS:raise ValueError(model)
    t=root.find("vType[@id='ego_baseline']")
    t.attrib.update(COMMON);t.set('carFollowModel',model)
    return dict(t.attrib)

def freeze_demand(root,seed):
    distribution=root.find("vTypeDistribution[@id='background']")
    types=distribution.get('vTypes').split();weights=list(map(float,distribution.get('probabilities').split()))
    # Fix individual desired-speed factors for every model, rather than drawing during SUMO loading.
    for t in root.findall('vType'):
        t.set('speedFactor','1');t.set('speedDev','0')
    vehicles=list(root.findall('vehicle'))
    for v in vehicles:root.remove(v)
    for flow in list(root.findall('flow')):
        begin=float(flow.get('begin'));end=float(flow.get('end'));period=3600/float(flow.get('vehsPerHour'))
        for i in range(math.ceil((end-begin)/period)):
            depart=begin+i*period
            if depart>=end:break
            ident=f"{flow.get('id')}.{i}"
            draw=int.from_bytes(hashlib.sha256(f'type:{seed}:{ident}'.encode()).digest()[:8],'big')/2**64*sum(weights)
            selected=types[-1]
            for typ,weight in zip(types,weights):
                draw-=weight
                if draw<0:selected=typ;break
            vehicles.append(ET.Element('vehicle',id=ident,type=selected,route=flow.get('route'),
                depart=f'{depart:.6f}'.rstrip('0').rstrip('.') if depart else '0',
                departLane=flow.get('departLane','best'),departSpeed=flow.get('departSpeed','max')))
        root.remove(flow)
    for v in sorted(vehicles,key=lambda v:(float(v.get('depart')),v.get('id'))):root.append(v)

def demand_hash(root):
    fields={tag:[dict(sorted(v.attrib.items())) for v in root.findall(tag)] for tag in ['route','vehicle']}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()

def split_for(rep):
    if not 0<=rep<5:raise ValueError('Matched study uses exactly 5 repeats per condition')
    return 'train' if rep<3 else 'validation' if rep==3 else 'test'
